"""Local-only evidence runner: preserve every sample, never invoke production."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .core import BoardSpec, DetectorConfig, MultiBoardTracker
from .mechanics import AssemblySpec, SlidingBoardTracker
from .coordinator import DynamicMechanicalTracker
from .media import extract_frames, fresh_output, local_video, outside_repository, probe_media, source_frames
from .preview import write_preview
from .models import FrameRef


def run_video(video_path: Path, config_path: Path, output_dir: Path) -> Path:
    video = local_video(video_path)
    output = outside_repository(output_dir)
    if output.exists():
        raise FileExistsError("Output must be a fresh directory; nothing is overwritten")
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    interval = configuration.get("sample_interval_ms", 1000)
    duration_limit = configuration.get("max_duration_ms", 600000)
    if type(interval) is not int or interval <= 0 or type(duration_limit) is not int or duration_limit <= 0:
        raise ValueError("Sampling interval/duration limit must be positive integer milliseconds")
    boards = [BoardSpec(b["board_id"], b["label"], tuple(b["roi"]), b.get('assembly_id'), b.get('layer')) for b in configuration["boards"]]
    detector = DetectorConfig(**configuration.get("detector", {}))
    mechanical = 'assemblies' in configuration
    if not mechanical and any(b.assembly_id is not None for b in boards):
        raise ValueError('Physical boards require explicit assembly viewports')
    assemblies = [AssemblySpec(a['assembly_id'],tuple(a.get('search_region', a.get('viewport', ())))) for a in configuration.get('assemblies',[])]
    mode = configuration.get('tracking_mode', 'automatic_tracking')
    if mode not in ('automatic_tracking', 'oracle_geometry'):
        raise ValueError('tracking_mode must be automatic_tracking or oracle_geometry')
    if mode == 'automatic_tracking' and configuration.get('position_keyframes'):
        raise ValueError('Automatic tracking rejects position_keyframes; use explicit oracle_geometry mode')
    apertures = {a['assembly_id']:a.get('physical_aperture') for a in configuration.get('assemblies',[])}
    tracker = ((DynamicMechanicalTracker(boards,assemblies,detector,configuration.get('camera'),apertures) if mode == 'automatic_tracking' else SlidingBoardTracker(boards,assemblies,detector)) if mechanical else MultiBoardTracker(boards,detector))
    probe = probe_media(video, duration_limit)
    if not mechanical and any(not b.fits(probe["width"], probe["height"]) for b in boards):
        raise ValueError("An ROI lies outside the source frame")
    samples = source_frames(video, probe, interval, duration_limit)
    manual_warnings = configuration.get("reconfirm_at_ms", [])
    if not isinstance(manual_warnings, list) or any(type(v) is not int or v < 0 for v in manual_warnings):
        raise ValueError("reconfirm_at_ms must contain nonnegative millisecond timestamps")
    checkpoints = configuration.get('position_keyframes',[])
    if not isinstance(checkpoints,list) or (checkpoints and not mechanical):
        raise ValueError('position_keyframes require mechanical assemblies')
    manual_by_time = {}
    sampled_times = {s['timestamp_ms'] for s in samples}
    for point in checkpoints:
        t=point.get('timestamp_ms')
        if type(t) is not int or t not in sampled_times or t in manual_by_time or not isinstance(point.get('positions'),dict):
            raise ValueError('Manual checkpoints need unique exact sampled timestamps and positions; none are silently shifted')
        manual_by_time[t]=point['positions']
    output = fresh_output(output)
    extraction = extract_frames(video, probe, samples, output / "frames")
    for board in boards:
        (output / "crops" / board.board_id).mkdir(parents=True)
        if mechanical:
            (output / "masks" / board.board_id).mkdir(parents=True)
    frames, frame_lookup = [], {}
    for index, sample in enumerate(samples):
        frame_id = f"f{index:04d}"
        image_path = f"frames/{frame_id}.png"
        image = cv2.imread(str(output / image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not read an extracted sample")
        expected_shape = (sample.get("height", probe["height"]), sample.get("width", probe["width"]))
        if image.shape[:2] != expected_shape:
            raise RuntimeError("Extracted image geometry does not match source-frame metadata")
        crop_paths = {b.board_id:f'crops/{b.board_id}/{frame_id}.png' for b in boards}
        before_event_count = len(tracker.events)
        kwargs = {'manual_positions':manual_by_time.get(sample['timestamp_ms'])} if mechanical else {}
        record = tracker.observe(image, frame_id, sample["timestamp_ms"], image_path, crop_paths,
                                 time_status=sample["time_status"],
                                 force_reconfirm=any(sample["timestamp_ms"] >= value and (not (mechanical and mode == 'automatic_tracking') or index == 0 or samples[index-1]['timestamp_ms'] < value) for value in manual_warnings), **kwargs)
        for b,state in zip(boards,record['boards']):
            crop_roi=state.get('crop_roi') if mechanical else (list(b.roi) if b.fits(image.shape[1],image.shape[0]) else None)
            if crop_roi:
                x,y,w,h=crop_roi
                if not cv2.imwrite(str(output / crop_paths[b.board_id]),image[y:y+h,x:x+w]):
                    raise RuntimeError('Could not persist screen-position crop evidence')
                if mechanical:
                    mask=np.zeros((h,w),np.uint8)
                    for sx,sy,sw,sh in state['visible_regions']:
                        mask[sy-y:sy-y+sh,sx-x:sx-x+sw]=255
                    mask_path=f'masks/{b.board_id}/{frame_id}.png'
                    if not cv2.imwrite(str(output/mask_path),mask):
                        raise RuntimeError('Could not persist visibility-mask evidence')
                    state['mask_path']=mask_path
                    tracker.position_history[-len(boards)+boards.index(b)]['mask_path']=mask_path
            else:
                state['crop_path']=None
        record.update({key: sample[key] for key in ("native_pts", "native_time_base", "pts_origin", "source_index", "pts_time_ms")})
        record['frame_ref']=FrameRef(frame_id,sample['timestamp_ms'],sample['native_pts'],sample['native_time_base'],sample['source_index'],image_path,probe.get('sha256'),probe.get('video_stream_index',0)).to_dict()
        for event in tracker.events[before_event_count:]:
            previous = frame_lookup.get(event["before_frame_id"])
            event["source_interval_ms"] = [previous["pts_time_ms"] if previous else sample["pts_time_ms"], sample["pts_time_ms"]]
            event["source_frame"] = {key: sample[key] for key in ("native_pts", "native_time_base", "pts_origin", "source_index", "pts_time_ms")}
            if mechanical:
                before_state=next((s for s in previous.get('boards',[]) if s['board_id']==event['board_id']),None) if previous else None
                after_state=next(s for s in record['boards'] if s['board_id']==event['board_id'])
                event['before_mask_path']=before_state.get('mask_path') if before_state else None
                event['after_mask_path']=after_state.get('mask_path')
        frames.append(record)
        frame_lookup[frame_id] = record
    probe["sampled_frame_count"] = len(frames)
    manifest = {
        "schema_version": ("blackboard-mechanical-mvp-v3" if mode == "automatic_tracking" else "blackboard-mechanical-mvp-v2") if mechanical else "blackboard-mvp-v1", "status": "complete_visual_observations_only",
        "source": probe, "frame_extraction": extraction,
        "config": {**configuration, "tracking_mode":mode, "boards": [b.to_dict() for b in boards]},
        "evaluation_category":mode if mechanical else "legacy_fixed_roi",
        "camera_history":getattr(tracker,"camera_history",[]),
        "aggregation_status":getattr(tracker,"aggregation_status","legacy_regression_only"),
        "frames": frames, "events": tracker.events, "versions": tracker.versions,
        "writing_runs": tracker.writing_runs,
        "position_history":tracker.position_history if mechanical else [],
        "scope": {"audio_analysis": False, "formula_recognition": False, "vision_api": False,
                  "teaching_or_proof_order_inferred": False, "all_sampled_frames_retained": True,
                  "automatic_physical_identity_guaranteed":False,"mechanical_model":mechanical,
                  "ordinary_content_admission":not (mechanical and mode == "automatic_tracking")},
        "warnings": [("Mechanical masks represent panel/window geometry, not person segmentation." if mechanical else "Legacy fixed screen ROIs are not physical sliding-board identities; map assemblies/layers manually before mechanical analysis."),
                     "Events are visual candidates, not confirmed writing or teaching order.",
                     "Equal/overlapping observation intervals do not establish cross-board order.",
                     "Mechanical opposed motion is a prior, never identity evidence; hidden content is not erased.",
                     "Manual position checkpoints confirm geometry/identity; synthetic truth is not automatic tracking success.",
                     "Motion-time residual content is deferred until a reliably aligned stationary observation.",
                     "No previous board versions or unscreened sampled frames were deleted."],
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(manifest, output)
    return path
