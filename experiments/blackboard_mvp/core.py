"""Per-board visual observations. No OCR, audio, teaching-order inference or API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    board_id: str
    label: str
    roi: tuple[int, int, int, int]
    assembly_id: str | None = None
    layer: str | None = None

    def __post_init__(self):
        if not isinstance(self.board_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.board_id):
            raise ValueError("board_id must be a short path-safe identifier")
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 100:
            raise ValueError("Each board requires a human-readable location label")
        if len(self.roi) != 4 or any(type(v) is not int for v in self.roi):
            raise ValueError("ROI must be integer [x,y,width,height]")
        x, y, w, h = self.roi
        if min(w, h) <= 0 or (self.assembly_id is None and min(x, y) < 0):
            raise ValueError("ROI coordinates/dimensions are invalid")
        if (self.assembly_id is None) != (self.layer is None):
            raise ValueError("assembly_id and layer must be provided together")
        if self.assembly_id is not None:
            if not isinstance(self.assembly_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.assembly_id) or self.layer not in ("outer", "inner"):
                raise ValueError("Invalid assembly identity/layer")

    def fits(self, width, height):
        x, y, w, h = self.roi
        return min(x,y) >= 0 and x + w <= width and y + h <= height

    def to_dict(self):
        result = asdict(self)
        result["roi"] = list(self.roi)
        return result


@dataclass(frozen=True)
class DetectorConfig:
    pixel_threshold: int = 25
    min_changed_pixels: int = 8
    min_component_pixels: int = 2
    registration_threshold: float = 0.25
    run_gap_ms: int = 5000
    motion_shift_pixels: float = 3.0
    interference_ratio: float = 0.20

    def __post_init__(self):
        for name in ("pixel_threshold", "min_changed_pixels", "min_component_pixels", "run_gap_ms"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pixel_threshold > 255 or not 0 < self.registration_threshold < 1:
            raise ValueError("Invalid detector thresholds")
        if not 0 < self.interference_ratio < 1 or self.motion_shift_pixels <= 0:
            raise ValueError("Invalid motion/interference thresholds")


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _coherent_shift(before, after, mask=None):
    """Conservative feature-flow warning, not physical board identification."""
    points = cv2.goodFeaturesToTrack(before, maxCorners=200, qualityLevel=0.02,
                                    minDistance=5, mask=mask)
    if points is None or len(points) < 6:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        before, after, points, None, winSize=(31, 31), maxLevel=4,
    )
    if moved is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if good.sum() < 6 or good.mean() < 0.6:
        return None
    deltas = moved.reshape(-1, 2)[good] - points.reshape(-1, 2)[good]
    median = np.median(deltas, axis=0)
    coherent = np.linalg.norm(deltas - median, axis=1) <= 2.0
    if coherent.mean() < 0.8:
        return None
    return float(np.linalg.norm(median))


class MultiBoardTracker:
    def __init__(self, boards: list[BoardSpec], config: DetectorConfig | None = None):
        if not boards or len({b.board_id for b in boards}) != len(boards):
            raise ValueError("Provide a nonempty list of unique board IDs")
        self.boards = list(boards)
        self.config = config or DetectorConfig()
        self.events: list[dict] = []
        self.versions: list[dict] = []
        self.writing_runs: list[dict] = []
        self._states: dict[str, dict] = {}
        self._previous_frame = None
        self._previous_time = None
        self._frame_ids: set[str] = set()
        self._shape = None
        self._active_runs: dict[str, dict] = {}

    def _event(self, board, frame_id, timestamp, crop_path, kind, reasons, uncertainty,
               regions, version_id, previous_version_id, previous):
        return {
            "event_id": f"{board.board_id}:{frame_id}", "board_id": board.board_id,
            "label": board.label, "kind": kind, "observed_at_ms": timestamp,
            "interval_ms": [previous["timestamp_ms"] if previous else timestamp, timestamp],
            "before_frame_id": previous["frame_id"] if previous else None,
            "before_crop_path": previous["crop_path"] if previous else None,
            "after_frame_id": frame_id, "after_crop_path": crop_path,
            "change_regions": regions, "selected_reasons": reasons,
            "uncertainty": uncertainty, "version_id": version_id,
            "previous_version_id": previous_version_id,
            "simultaneous_group": None, "order_status": "observation_only",
            "roi_identity_basis": "manual_ROI_with_heuristic_warnings",
        }

    def _version(self, board, frame_id, timestamp, crop_path, status, visibility, previous_id):
        number = sum(v["board_id"] == board.board_id for v in self.versions)
        version = {
            "version_id": f"{board.board_id}:v{number:04d}", "board_id": board.board_id,
            "label": board.label, "frame_id": frame_id, "timestamp_ms": timestamp,
            "crop_path": crop_path, "previous_version_id": previous_id,
            "content_status": status, "visibility": visibility, "roi_status": "confirmed",
            "content_confirmation": "unverified_visual_observation",
        }
        self.versions.append(version)
        return version["version_id"]

    def _regions(self, before, after):
        delta = cv2.absdiff(before, after)
        if delta.ndim == 3:
            delta = np.max(delta, axis=2)
        # No opening/whole-board percentage gate: preserve tiny minus signs.
        mask = (delta >= self.config.pixel_threshold).astype(np.uint8)
        count = int(mask.sum())
        if count < self.config.min_changed_pixels:
            return [], count
        components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = [{"bbox": [int(x), int(y), int(w), int(h)], "changed_pixels": int(area)}
                   for x, y, w, h, area in stats[1:components]
                   if area >= self.config.min_component_pixels]
        return regions, count

    def _global_warning(self, frame):
        if frame.shape != self._shape or frame.shape != self._previous_frame.shape:
            return "frame_geometry_changed"
        before, after = _gray(self._previous_frame), _gray(frame)
        phase_shift, response = cv2.phaseCorrelate(before.astype(np.float32), after.astype(np.float32))
        if response >= max(0.3, self.config.registration_threshold) and math.hypot(*phase_shift) > self.config.motion_shift_pixels:
            return "possible_camera_motion"
        outside = np.full(before.shape, 255, dtype=np.uint8)
        for board in self.boards:
            x, y, w, h = board.roi
            outside[y:y+h, x:x+w] = 0
        shift = _coherent_shift(before, after, outside)
        if shift is not None and shift > self.config.motion_shift_pixels:
            return "possible_camera_motion"
        outside_count = int(np.count_nonzero(outside))
        if outside_count:
            outside_ratio = np.count_nonzero((cv2.absdiff(before, after) >= self.config.pixel_threshold)
                                             & (outside > 0)) / outside_count
            ratios = []
            for board in self.boards:
                x, y, w, h = board.roi
                ratios.append(float(np.mean(cv2.absdiff(before[y:y+h, x:x+w], after[y:y+h, x:x+w])
                                            >= self.config.pixel_threshold)))
            if outside_ratio > 0.65 and sum(v > 0.45 for v in ratios) >= max(1, math.ceil(len(ratios)*0.75)):
                return "possible_scene_change"
        return None

    def observe(self, frame, frame_id, timestamp_ms, image_path, crop_paths,
                time_status="verified_pts", force_reconfirm=False):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR frame")
        if frame_id in self._frame_ids:
            raise ValueError("Frame IDs must be unique")
        if type(timestamp_ms) is not int or (self._previous_time is not None and timestamp_ms <= self._previous_time):
            raise ValueError("Observation timestamps must increase strictly")
        height, width = frame.shape[:2]
        if self._shape is None:
            if any(not b.fits(width, height) for b in self.boards):
                raise ValueError("An initial ROI lies outside the video frame")
            self._shape = frame.shape
        warning = "manual_reconfirmation_requested" if force_reconfirm else (
            self._global_warning(frame) if self._previous_frame is not None else None)
        new_events, frame_boards = [], []
        for board in self.boards:
            old = self._states.get(board.board_id)
            crop_path = crop_paths.get(board.board_id)
            previous_id = old["version_id"] if old else None
            invalid_reason = warning or ("roi_out_of_frame" if not board.fits(width, height) else None)
            # Latch identity warnings; neither an old resolution nor a similar
            # image automatically confirms the same physical board again.
            if old and old["roi_status"] == "needs_reconfirmation":
                invalid_reason = old["invalid_reason"]
            if invalid_reason:
                if old and old["roi_status"] != "needs_reconfirmation":
                    new_events.append(self._event(
                        board, frame_id, timestamp_ms, crop_path, "roi_needs_reconfirmation",
                        [invalid_reason], ["physical_board_identity_unconfirmed", "manual_review_required"],
                        [], None, previous_id, old,
                    ))
                self._states[board.board_id] = {
                    **(old or {}), "roi_status": "needs_reconfirmation", "invalid_reason": invalid_reason,
                    "frame_id": frame_id, "timestamp_ms": timestamp_ms, "crop_path": crop_path,
                    "version_id": previous_id,
                }
                self._active_runs.pop(board.board_id, None)
                frame_boards.append({"board_id": board.board_id, "crop_path": crop_path,
                                     "version_id": previous_id, "roi_status": "needs_reconfirmation",
                                     "visibility": "identity_unconfirmed"})
                continue
            x, y, w, h = board.roi
            pixels = frame[y:y+h, x:x+w].copy()
            current = _gray(pixels)
            visibility = old.get("visibility", "unknown") if old else "unknown"
            version_id = previous_id
            event = None
            if old is None:
                version_id = self._version(board, frame_id, timestamp_ms, crop_path,
                                           "initial_existing", "unknown", None)
                event = self._event(board, frame_id, timestamp_ms, crop_path, "initial_observation",
                                    ["already_present_at_clip_start"], ["appearance_time_before_clip_unknown"],
                                    [], version_id, None, None)
                baseline = pixels.copy()
                baseline_evidence = {"frame_id": frame_id, "crop_path": crop_path, "timestamp_ms": timestamp_ms}
            else:
                shift = _coherent_shift(old["gray"], current)
                if shift is not None and shift > self.config.motion_shift_pixels:
                    event = self._event(board, frame_id, timestamp_ms, crop_path, "roi_needs_reconfirmation",
                                        ["possible_board_motion"], ["physical_board_identity_unconfirmed", "manual_review_required"],
                                        [], None, previous_id, old)
                    self._states[board.board_id] = {**old, "roi_status": "needs_reconfirmation",
                        "invalid_reason": "possible_board_motion", "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms, "crop_path": crop_path}
                    new_events.append(event)
                    self._active_runs.pop(board.board_id, None)
                    frame_boards.append({"board_id": board.board_id, "crop_path": crop_path,
                        "version_id": previous_id, "roi_status": "needs_reconfirmation", "visibility": "identity_unconfirmed"})
                    continue
                regions, changed = self._regions(old["pixels"], pixels)
                baseline = old["baseline"]
                baseline_evidence = old["baseline_evidence"]
                accumulated = False
                if not regions and visibility != "possible_interference":
                    regions, changed = self._regions(baseline, pixels)
                    accumulated = bool(regions)
                if regions:
                    ratio = changed / current.size
                    stable_regions, _ = self._regions(baseline, pixels)
                    restored = visibility == "possible_interference" and not stable_regions
                    visibility = "possible_interference" if ratio >= self.config.interference_ratio and not restored else "unknown"
                    kind = "visibility_restored_candidate" if restored else "visual_change_candidate"
                    version_id = self._version(board, frame_id, timestamp_ms, crop_path,
                        "visibility_observation" if restored else "unverified_visual_change", visibility, previous_id)
                    uncertainty = ["possible_interference", "meaning_unverified", "not_speech_or_proof_order"]
                    if time_status != "verified_pts":
                        uncertainty.append("source_time_unverified")
                    reasons = ["per_board_local_pixel_change"]
                    if visibility == "possible_interference":
                        reasons.append("large_visual_change_requires_review")
                    if restored:
                        reasons.append("matches_pre_interference_observation")
                    event = self._event(board, frame_id, timestamp_ms, crop_path, kind,
                                        reasons, uncertainty, regions, version_id, previous_id, old)
                    if accumulated:
                        event["selected_reasons"].append("accumulated_change_since_reference")
                        event["uncertainty"].append("change_accumulated_over_multiple_samples")
                        event["interval_ms"][0] = baseline_evidence["timestamp_ms"]
                        event["previous_sample_frame_id"] = old["frame_id"]
                        event["previous_sample_crop_path"] = old["crop_path"]
                        event["before_frame_id"] = baseline_evidence["frame_id"]
                        event["before_crop_path"] = baseline_evidence["crop_path"]
                    if visibility != "possible_interference":
                        baseline = pixels.copy()
                        baseline_evidence = {"frame_id": frame_id, "crop_path": crop_path, "timestamp_ms": timestamp_ms}
                else:
                    self._active_runs.pop(board.board_id, None)
            if event:
                new_events.append(event)
                if event["kind"] == "visual_change_candidate" and visibility != "possible_interference":
                    run = self._active_runs.get(board.board_id)
                    if not run or timestamp_ms - run["interval_ms"][1] > self.config.run_gap_ms:
                        run = {"run_id": f"{board.board_id}:run{len(self.writing_runs):04d}",
                               "board_id": board.board_id, "event_ids": [],
                               "interval_ms": list(event["interval_ms"]),
                               "interpretation": "continuous_visual_change_candidate",
                               "uncertainty": ["not_confirmed_writing_or_explanation_order"]}
                        self.writing_runs.append(run)
                        self._active_runs[board.board_id] = run
                    run["event_ids"].append(event["event_id"])
                    run["interval_ms"][1] = timestamp_ms
                else:
                    self._active_runs.pop(board.board_id, None)
            self._states[board.board_id] = {
                "gray": current.copy(), "pixels": pixels, "baseline": baseline,
                "baseline_evidence": baseline_evidence, "frame_id": frame_id,
                "timestamp_ms": timestamp_ms, "crop_path": crop_path, "version_id": version_id,
                "roi_status": "confirmed", "visibility": visibility,
            }
            frame_boards.append({"board_id": board.board_id, "crop_path": crop_path,
                                 "version_id": version_id, "roi_status": "confirmed", "visibility": visibility})
        if len(new_events) > 1:
            group = f"unordered:{frame_id}"
            for event in new_events:
                event["simultaneous_group"] = group
                event["order_status"] = "undetermined_within_interval"
                event["uncertainty"].append("cross_board_order_within_sample_unknown")
        self.events.extend(new_events)
        self._frame_ids.add(frame_id)
        self._previous_frame = frame.copy()
        self._previous_time = timestamp_ms
        return {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "pts_time_ms": timestamp_ms,
                "time_status": time_status, "image_path": image_path, "boards": frame_boards,
                "event_ids": [e["event_id"] for e in new_events], "selected": bool(new_events)}
