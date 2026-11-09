"""JSON state contracts for local camera/physical-board tracking.

Heuristic confidence is evidence, never a calibrated identity probability.  Native
source clocks do not change when a geometry/identity/comparison epoch changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_roi(value, name="roi"):
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(v) for v in value):
        raise ValueError(f"{name} requires four finite coordinates")
    if value[2] <= 0 or value[3] <= 0:
        raise ValueError(f"{name} must have positive dimensions")


def _epoch(value):
    if type(value) is not int or value < 0:
        raise ValueError("epoch must be a nonnegative integer")


@dataclass(frozen=True)
class FrameRef:
    frame_id: str
    timestamp_ms: int
    native_pts: int | None = None
    native_time_base: str | None = None
    source_index: int | None = None
    image_path: str | None = None
    source_sha256: str | None = None
    stream_index: int = 0

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id is required")
        if type(self.timestamp_ms) is not int or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if self.native_pts is not None and type(self.native_pts) is not int:
            raise ValueError("native_pts is an integer or unknown")
        if (self.native_pts is None) != (self.native_time_base is None):
            raise ValueError("native_pts/time_base must be provided together")
        if self.native_time_base is not None:
            from fractions import Fraction
            try:
                valid = Fraction(self.native_time_base) > 0
            except (ValueError, TypeError, ZeroDivisionError):
                valid = False
            if not valid:
                raise ValueError("native_time_base must be a positive rational")
        if self.source_index is not None and (type(self.source_index) is not int or self.source_index < 0):
            raise ValueError("source_index must be nonnegative")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TransformRef:
    transform_id: str
    from_space: str
    to_space: str
    matrix: list[list[float]]
    geometry_epoch: int = 0
    valid_domain: list | None = None
    error_px: float | None = None

    def __post_init__(self):
        _epoch(self.geometry_epoch)
        if not self.transform_id or not self.from_space or not self.to_space:
            raise ValueError("transform identity/direction is required")
        if len(self.matrix) != 3 or any(len(r) != 3 for r in self.matrix) or not all(_number(x) for r in self.matrix for x in r):
            raise ValueError("transform requires a finite 3x3 matrix")
        a,b,c = self.matrix
        determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(determinant) < 1e-12:
            raise ValueError("transform must be invertible")
        if self.valid_domain is not None:
            if not isinstance(self.valid_domain, (list, tuple)) or not self.valid_domain or not all(isinstance(p, (list, tuple)) and len(p) == 2 and all(_number(x) for x in p) for p in self.valid_domain):
                raise ValueError("valid_domain must contain finite point pairs")
        if self.error_px is not None and (not _number(self.error_px) or self.error_px < 0):
            raise ValueError("error must be finite/nonnegative or unknown")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BoardGeometry:
    board_id: str
    frame_id: str
    measurement_status: str
    measured_screen_roi: list | None = None
    predicted_screen_roi: list | None = None
    prediction_uncertainty: float | None = None
    last_verified_geometry: dict | None = None
    motion_status: str = "unresolved"
    geometry_epoch: int = 0

    def __post_init__(self):
        if not self.board_id or not self.frame_id:
            raise ValueError("board/frame identity required")
        _epoch(self.geometry_epoch)
        if self.measurement_status not in {"observed", "partially_observed", "predicted_only", "unavailable"}:
            raise ValueError("unknown geometry status")
        validate_roi(self.measured_screen_roi, "measured_screen_roi")
        validate_roi(self.predicted_screen_roi, "predicted_screen_roi")
        measured = self.measurement_status in {"observed", "partially_observed"}
        if measured != (self.measured_screen_roi is not None):
            raise ValueError("only an actual measurement may populate measured_screen_roi")
        if self.measurement_status == "predicted_only" and self.predicted_screen_roi is None:
            raise ValueError("predicted_only requires a prediction")
        if self.prediction_uncertainty is not None and (not _number(self.prediction_uncertainty) or self.prediction_uncertainty < 0):
            raise ValueError("prediction uncertainty must be finite and nonnegative")
        if self.motion_status not in {"stationary", "moving", "unresolved"}:
            raise ValueError("invalid motion state")

    def to_dict(self):
        out = asdict(self)
        r = self.measured_screen_roi
        out['measured_screen_polygon'] = None if r is None else [[r[0],r[1]], [r[0]+r[2],r[1]], [r[0]+r[2],r[1]+r[3]], [r[0],r[1]+r[3]]]
        return out


@dataclass(frozen=True)
class BoardIdentity:
    board_id: str
    association_status: str
    identity_epoch: int = 0
    confirmation_basis: str | None = None
    confidence_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _epoch(self.identity_epoch)
        if not self.board_id or self.association_status not in {"confirmed", "tentative", "uncertain", "historical_unmapped"}:
            raise ValueError("invalid physical identity state")
        if self.association_status == 'confirmed' and self.confirmation_basis not in {'manual_initial', 'manual_checkpoint', 'direct_visual'}:
            raise ValueError("mechanical prediction/position alone cannot confirm physical identity")
        for value in self.confidence_evidence.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("confidence evidence must not contain NaN/Infinity")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ComparisonEligibility:
    eligible: bool
    reason_codes: tuple[str, ...] = ()
    comparison_epoch: int = 0
    baseline_id: str | None = None

    def __post_init__(self):
        _epoch(self.comparison_epoch)
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be bool")
        if self.eligible and (self.reason_codes or self.baseline_id is None):
            raise ValueError("eligible comparison requires a baseline and no rejection reasons")

    def to_dict(self):
        out = asdict(self)
        out['reason_codes'] = list(self.reason_codes)
        return out


def comparison_gate(*, camera_reliability, identity_status, measurement_status,
                    baseline_ready=False, content_enabled=False,
                    visibility_status="unknown", comparison_epoch=0, baseline_id=None):
    """Fail closed; S0-S2 callers leave content_enabled false deliberately."""
    reasons = []
    if camera_reliability != 'verified': reasons.append('camera_geometry_unreliable')
    if identity_status != 'confirmed': reasons.append('physical_identity_unconfirmed')
    if measurement_status not in {'observed', 'partially_observed'}: reasons.append('no_current_geometry_measurement')
    if visibility_status not in {'visible', 'partially_visible', 'partially_occluded'}: reasons.append('visibility_not_comparable')
    if not baseline_ready or baseline_id is None: reasons.append('comparison_baseline_not_ready')
    if not content_enabled: reasons.append('content_comparison_disabled_s0_s2')
    return ComparisonEligibility(not reasons, tuple(reasons), comparison_epoch, baseline_id).to_dict()
