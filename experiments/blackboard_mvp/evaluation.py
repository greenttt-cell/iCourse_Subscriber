"""Local evaluation; self-reported tracking coverage is not tracking accuracy.

Ground truth is evaluation-only and must never be passed to an automatic tracker.
Its optional form is::

    {"frames": {frame_id: {board_id: {
        "observable": True, "screen_roi": [x, y, w, h],
        "associated_physical_board_id": "independently annotated identity"
    }}}, "reacquisition_windows": [{"board_id": "...",
        "recoverable_since_ms": 123, "deadline_ms": 456}]}

The associated identity annotates what the reported measurement *actually* sees,
not the identity the algorithm claims. Missing labels stay unknown. No identity
truth is inferred from nearest screen position or opposed mechanical motion.
"""
from __future__ import annotations

import math
from statistics import median


def _valid_roi(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(type(x) in (int, float) and math.isfinite(x) for x in value)
            and value[2] > 0 and value[3] > 0)


def _accepted(state):
    return (state.get('identity_status') == 'confirmed'
            and _valid_roi(state.get('measured_screen_roi'))
            and state.get('camera_reliability') == 'verified')


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    point = (len(values) - 1) * fraction
    lower, upper = math.floor(point), math.ceil(point)
    return values[lower] + (values[upper] - values[lower]) * (point - lower)


def _distribution(values):
    return {'samples': len(values), 'median_ms': median(values) if values else None,
            'p95_ms': _percentile(values, .95), 'max_ms': max(values) if values else None}


def _unknown_spans(observations, clip_end):
    """Bounds describe sampling uncertainty, not continuous physical truth."""
    spans = []
    start = None
    for i, (timestamp, frame_id, accepted, _) in enumerate(observations):
        if not accepted and start is None:
            start = i
        if accepted and start is not None:
            spans.append(_span(observations, start, i - 1, i, clip_end))
            start = None
    if start is not None:
        spans.append(_span(observations, start, len(observations) - 1, None, clip_end))
    return spans


def _span(observations, start, last, recovered, clip_end):
    first_time, first_id = observations[start][:2]
    last_time, last_id = observations[last][:2]
    previous = observations[start - 1][0] if start else first_time
    end = observations[recovered][0] if recovered is not None else clip_end
    return {'first_unknown_frame_id': first_id, 'last_unknown_frame_id': last_id,
            'first_unknown_at_ms': first_time, 'last_unknown_at_ms': last_time,
            'previous_accepted_at_ms': observations[start - 1][0] if start else None,
            'recovered_at_ms': observations[recovered][0] if recovered is not None else None,
            'recovered_frame_id': observations[recovered][1] if recovered is not None else None,
            'unknown_duration_lower_bound_ms': max(0, last_time - first_time),
            'unknown_duration_upper_bound_ms': max(0, end - previous) if end is not None else None,
            'sample_hold_estimate_ms': max(0, end - first_time) if end is not None else None,
            'tail_after_last_observation_unverified': recovered is None,
            'reacquisition_observed_gap_ms': observations[recovered][0] - first_time if recovered is not None else None}


def evaluate_tracking(manifest, ground_truth=None):
    """Return JSON-safe coverage, unknown spans and only supported GT metrics.

    ``operational_accepted_coverage`` counts confirmed identity + a finite measured
    ROI + verified camera. It is an internal acceptance rate, never accuracy.
    Ground truth contains no defaults that can turn unknown labels into correct
    detections. Reacquisition windows must come from independent annotations.
    """
    frames = manifest.get('frames', [])
    ids = [b['board_id'] for b in manifest.get('config', {}).get('boards', [])]
    if not ids:
        ids = list(dict.fromkeys(b['board_id'] for f in frames for b in f.get('boards', [])))
    if len(set(ids)) != len(ids):
        raise ValueError('Evaluation board IDs must be unique')
    times = [f['timestamp_ms'] for f in frames]
    if any(type(t) is not int or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Evaluation timestamps must increase strictly')
    if len({f['frame_id'] for f in frames}) != len(frames):
        raise ValueError('Evaluation frame IDs must be unique')
    truth = ground_truth or {}
    truth_frames = truth.get('frames', {})
    clip_end = manifest.get('source', {}).get('duration_ms')
    if type(clip_end) is not int or (times and clip_end < times[-1]):
        clip_end = None
    per_board = {}
    for board_id in ids:
        observations = []
        labels = []
        for frame in frames:
            matching = [b for b in frame.get('boards', []) if b['board_id'] == board_id]
            if len(matching) > 1:
                raise ValueError('A frame may contain a physical board at most once')
            state = dict(matching[0]) if matching else {}
            state.setdefault('camera_reliability', frame.get('camera', {}).get('reliability'))
            observations.append((frame['timestamp_ms'], frame['frame_id'], _accepted(state), state))
            labels.append(truth_frames.get(frame['frame_id'], {}).get(board_id, {}))
        accepted = sum(o[2] for o in observations)
        measured = sum(_valid_roi(o[3].get('measured_screen_roi')) for o in observations)
        spans = _unknown_spans(observations, clip_end)
        observable = [(o, g) for o, g in zip(observations, labels) if g.get('observable') is True]
        position_errors = []
        identity_errors = []
        for o, g in zip(observations, labels):
            if not o[2]:
                continue
            if g.get('observable') is True and _valid_roi(g.get('screen_roi')):
                # Maximum edge error tests both position and unchanged panel size.
                actual, expected = o[3]['measured_screen_roi'], g['screen_roi']
                a = [actual[0], actual[1], actual[0] + actual[2], actual[1] + actual[3]]
                e = [expected[0], expected[1], expected[0] + expected[2], expected[1] + expected[3]]
                position_errors.append(max(abs(x - y) for x, y in zip(a, e)))
            actual_id = g.get('associated_physical_board_id')
            if isinstance(actual_id, str) and actual_id:
                identity_errors.append(actual_id != board_id)
        windows = [w for w in truth.get('reacquisition_windows', []) if w.get('board_id') == board_id]
        latencies, missed_windows, unassessable_windows = [], 0, 0
        for window in windows:
            start, end = window.get('recoverable_since_ms'), window.get('deadline_ms')
            if type(start) is not int or type(end) is not int or end < start:
                raise ValueError('Reacquisition truth requires a valid independently labelled interval')
            in_window = [(o, g) for o, g in zip(observations, labels) if start <= o[0] <= end]
            # A claimed confirmed output is not proof of correct reacquisition.
            # Require an independent annotation of its actually observed identity.
            labelled = [(o, g) for o, g in in_window
                        if isinstance(g.get('associated_physical_board_id'), str)]
            if not labelled:
                unassessable_windows += 1
                continue
            candidates = [o[0] for o, g in labelled
                          if o[2] and g['associated_physical_board_id'] == board_id]
            if candidates:
                latencies.append(min(candidates) - start)
            else:
                missed_windows += 1
        upper = [s['unknown_duration_upper_bound_ms'] for s in spans]
        estimate = [s['sample_hold_estimate_ms'] for s in spans]
        gaps = [s['reacquisition_observed_gap_ms'] for s in spans if s['reacquisition_observed_gap_ms'] is not None]
        per_board[board_id] = {
            'sample_count': len(observations), 'accepted_samples': accepted,
            'measured_samples': measured, 'unknown_samples': len(observations) - accepted,
            'operational_accepted_coverage': accepted / len(observations) if observations else None,
            'measurement_coverage': measured / len(observations) if observations else None,
            'coverage_is_accuracy': False,
            'gt_observable_samples': len(observable) if ground_truth is not None else None,
            'gt_observable_accepted_coverage': (sum(o[2] for o, _ in observable) / len(observable)) if observable else None,
            'identity_error_rate': sum(identity_errors) / len(identity_errors) if identity_errors else None,
            'identity_labelled_accepted_samples': len(identity_errors),
            'position_error': {'metric': 'maximum screen-rectangle edge error in pixels', 'samples': len(position_errors),
                               'median_px': median(position_errors) if position_errors else None,
                               'p95_px': _percentile(position_errors, .95),
                               'max_px': max(position_errors) if position_errors else None},
            'unknown_duration_ms': {'lower_bound_ms': sum(s['unknown_duration_lower_bound_ms'] for s in spans),
                                    'upper_bound_ms': sum(upper) if all(x is not None for x in upper) else None,
                                    'sample_hold_estimate_ms': sum(estimate) if all(x is not None for x in estimate) else None,
                                    'continuous_physical_truth_verified': False},
            'unknown_spans': spans,
            'reacquisition_observed_gap': {**_distribution(gaps), 'is_ground_truth_latency': False},
            'reacquisition_latency': {**_distribution(latencies), 'annotated_windows': len(windows),
                                      'unrecovered_windows': missed_windows,
                                      'unassessable_windows': unassessable_windows,
                                      'requires_independently_confirmed_identity': True},
            'unavailable_metrics': [name for name, available in [('identity_error_rate', bool(identity_errors)),
                                    ('position_error', bool(position_errors)),
                                    ('reacquisition_latency', len(windows) > unassessable_windows)] if not available],
        }
    return {'evaluation_kind': 'automatic_tracking_observation_audit',
            'ground_truth_supplied': ground_truth is not None,
            'coverage_definition': 'confirmed physical identity AND finite measured screen ROI AND verified camera; self-report, not accuracy',
            'unknown_duration_definition': 'sampling bounds plus sample-hold estimate; tail to source duration is unverified',
            'per_board': per_board,
            'warnings': ['No ground-truth identity or position is inferred from screen location or opposed motion.',
                         'An old legacy screen_roi is not accepted as a new measured_screen_roi.',
                         'Operational coverage is not identity correctness or geometric accuracy.',
                         'No labelled reacquisition opportunity means true reacquisition latency is unknown.']}
