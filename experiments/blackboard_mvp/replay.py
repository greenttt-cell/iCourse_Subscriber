"""Replay immutable local samples. No video decode, model, network or database."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .core import BoardSpec, DetectorConfig
from .mechanics import AssemblySpec
from .coordinator import DynamicMechanicalTracker
from .media import fresh_output, outside_repository
from .preview import write_preview
from .models import FrameRef


def safe_member(root, value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Evidence must be a safe relative local file')
    relative = Path(value)
    if relative.is_absolute() or any(v in ('.','..') for v in relative.parts):
        raise ValueError('Evidence path escapes source directory')
    resolved = (root/relative).resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError('Evidence path is not a regular source member')
    return resolved


def replay_manifest(input_manifest, output_dir, config_path=None):
    source_path = Path(input_manifest).expanduser().resolve(strict=True)
    source_root = source_path.parent
    implementation_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    original_bytes = source_path.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    previous = json.loads(original_bytes)
    configuration = json.loads(Path(config_path).read_text()) if config_path else copy.deepcopy(previous['config'])
    if configuration.get('position_keyframes') or configuration.get('tracking_mode') == 'oracle_geometry':
        raise ValueError('Automatic replay refuses oracle geometry and position keyframes')
    configuration['tracking_mode'] = 'automatic_tracking'
    boards = [BoardSpec(b['board_id'],b['label'],tuple(b['roi']),b['assembly_id'],b['layer']) for b in configuration['boards']]
    assemblies = [AssemblySpec(a['assembly_id'],tuple(a.get('search_region',a.get('viewport',())))) for a in configuration['assemblies']]
    apertures = {a['assembly_id']:a.get('physical_aperture') for a in configuration['assemblies']}
    tracker = DynamicMechanicalTracker(boards,assemblies,DetectorConfig(**configuration.get('detector',{})),configuration.get('camera'),apertures)
    source_frames = previous['frames']
    if not source_frames or len(source_frames)>10000:
        raise ValueError('Replay needs 1..10000 sampled source frames')
    # Validate all references and time mappings before creating any new result.
    inputs=[]; seen=set(); previous_time=None
    required=('native_pts','native_time_base','pts_origin','source_index','pts_time_ms')
    for record in source_frames:
        fid=record['frame_id']; t=record['timestamp_ms']
        if not isinstance(fid,str) or len(fid)>80 or not fid.replace('_','').isalnum() or fid in seen:
            raise ValueError('Invalid or duplicate frame ID')
        if type(t) is not int or t<0 or (previous_time is not None and t<=previous_time):
            raise ValueError('Replay source timestamps must strictly increase')
        if any(key not in record for key in required):
            raise ValueError('Replay requires preserved native PTS metadata')
        inputs.append(safe_member(source_root,record['image_path']))
        seen.add(fid);previous_time=t
    output=outside_repository(Path(output_dir))
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError('Replay output must be independent of original results')
    output=fresh_output(output)
    (output/'frames').mkdir()
    for board in boards:
        (output/'crops'/board.board_id).mkdir(parents=True)
        (output/'masks'/board.board_id).mkdir(parents=True)
    frames=[]; copies=[]; frame_lookup={}
    for index,(old,src) in enumerate(zip(source_frames,inputs)):
        fid=old['frame_id']; image_path=f'frames/{fid}.png'
        dest=output/image_path
        shutil.copy2(src,dest)
        src_digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=src_digest:
            raise RuntimeError('Source evidence copy integrity failure')
        image=cv2.imread(str(dest),cv2.IMREAD_COLOR)
        if image is None:raise RuntimeError('Could not decode existing source sample')
        paths={b.board_id:f'crops/{b.board_id}/{fid}.png' for b in boards}
        begin=len(tracker.events)
        record=tracker.observe(image,fid,old['timestamp_ms'],image_path,paths,time_status=old.get('time_status','unverified'))
        for key in required:record[key]=copy.deepcopy(old[key])
        record['frame_ref']=FrameRef(fid,old['timestamp_ms'],old['native_pts'],old['native_time_base'],old['source_index'],image_path,previous.get('source',{}).get('sha256'),previous.get('source',{}).get('video_stream_index',0)).to_dict()
        for state in record['boards']:
            roi=state['crop_roi']
            if roi:
                x,y,w,h=roi
                if not cv2.imwrite(str(output/paths[state['board_id']]),image[y:y+h,x:x+w]):raise RuntimeError('Crop save failed')
                mask=np.zeros((h,w),np.uint8)
                for sx,sy,sw,sh in state['visible_regions']:mask[sy-y:sy-y+sh,sx-x:sx-x+sw]=255
                mp=f"masks/{state['board_id']}/{fid}.png"
                if not cv2.imwrite(str(output/mp),mask):raise RuntimeError('Mask save failed')
                state['mask_path']=mp
                state['mask_semantics']='known_mechanical_visibility_only_unknown_is_black_not_erased'
            else:state['crop_path']=None
            for hstate in tracker.position_history[-len(boards):]:
                if hstate['board_id']==state['board_id']:hstate.update(copy.deepcopy(state))
        for event in tracker.events[begin:]:
            previous_record=frame_lookup.get(event.get('before_frame_id'))
            event['source_interval_ms']=[previous_record['pts_time_ms'] if previous_record else record['pts_time_ms'],record['pts_time_ms']]
            event['source_frame']={key:copy.deepcopy(record[key]) for key in required}
        frames.append(record);frame_lookup[fid]=record
        copies.append({'frame_id':fid,'source_image_path':old['image_path'],'copied_image_path':image_path,'sha256':src_digest})
        if index % 20==0 or index==len(inputs)-1:print(f'Replayed {index+1}/{len(inputs)} samples',flush=True)
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=original_sha:
        raise RuntimeError('Original manifest changed externally during replay')
    manifest={'schema_version':'blackboard-mechanical-mvp-v3','status':'complete_geometry_observations_content_disabled',
              'evaluation_category':('real_screen_recording_automatic_tracking' if configuration.get('review_metadata',{}).get('source_kind')=='local_screen_recording' else 'offline_automatic_replay_source_kind_unverified'),
              'source':copy.deepcopy(previous['source']),'config':configuration,'frames':frames,
              'events':tracker.events,'versions':tracker.versions,'writing_runs':[],
              'aggregation_status':'not_implemented_in_s0_s2','position_history':tracker.position_history,
              'camera_history':tracker.camera_history,
              'replay':{'input_manifest':str(source_path),'input_manifest_sha256':original_sha,'all_source_samples_copied':True,
                        'source_samples':copies,'implementation_source_sha256':implementation_sha,'video_reextracted':False,'original_manifest_unchanged':True},
              'scope':{'all_sampled_frames_retained':True,'ordinary_content_admission':False,'vision_api':False,'audio_analysis':False,
                       'person_segmentation':False,'formula_recognition':False,'teaching_or_proof_order_inferred':False},
              'warnings':['S0–S2 geometry/identity only. Ordinary content comparison is disabled.',
                          'Positions may be measured, predicted or unknown; physical identity is independently evaluated.',
                          'Initial ROIs may be visible fragments, not complete board extents; physical aperture remains unknown unless manually supplied.',
                          'Local screen-recording timestamps are not original-course timestamps.',
                          'Operational tracking coverage is not accuracy; unavailable ground truth is reported explicitly.']}
    from .evaluation import evaluate_tracking
    manifest['tracking_metrics']=evaluate_tracking(manifest)
    path=output/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'tracking-metrics.json').write_text(json.dumps(manifest['tracking_metrics'],ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_preview(manifest,output)
    return path
