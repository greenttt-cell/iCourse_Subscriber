"""Independent acceptance tests, separate from oracle/legacy geometry fixtures.

Only initialization is supplied to automatic tracking. Evaluation-only truth never
enters trackers. These tests do not interpret self-reported coverage as accuracy.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blackboard_mvp.evaluation import evaluate_tracking
from blackboard_mvp.core import BoardSpec
from blackboard_mvp.mechanics import AssemblySpec
from blackboard_mvp.coordinator import DynamicMechanicalTracker
from blackboard_mvp.mechanical_synthetic import assembly_config, mechanical_board_config, positions, render_mechanical, _board_surface
from blackboard_mvp.synthetic import write_video
from blackboard_mvp.runner import run_video


def board(roi=None, identity='confirmed', camera='verified', bid='a'):
    return {'board_id': bid, 'measured_screen_roi': roi if roi is not None else [10, 20, 30, 40],
            'identity_status': identity, 'camera_reliability': camera}


def manifest(states, times=None, duration=None):
    times = list(range(0, len(states) * 1000, 1000)) if times is None else times
    return {'config': {'boards': [{'board_id': 'a'}]},
            'source': {'duration_ms': duration},
            'frames': [{'frame_id': f'f{i}', 'timestamp_ms': t, 'boards': [s] if s is not None else []}
                       for i, (t, s) in enumerate(zip(times, states))]}


class EvaluationMetricTests(unittest.TestCase):
    def test_no_ground_truth_never_reports_accuracy_or_true_latency(self):
        value = evaluate_tracking(manifest([board(), board(identity='identity_uncertain'), board()]))
        got = value['per_board']['a']
        self.assertEqual(got['operational_accepted_coverage'], 2 / 3)
        self.assertFalse(got['coverage_is_accuracy'])
        self.assertIsNone(got['identity_error_rate'])
        self.assertIsNone(got['position_error']['median_px'])
        self.assertIsNone(got['reacquisition_latency']['median_ms'])
        self.assertEqual(got['reacquisition_observed_gap']['median_ms'], 1000)
        self.assertFalse(got['reacquisition_observed_gap']['is_ground_truth_latency'])
        self.assertEqual(set(got['unavailable_metrics']), {'identity_error_rate', 'position_error', 'reacquisition_latency'})

    def test_prediction_and_legacy_screen_roi_cannot_count_as_measurements(self):
        predicted = board()
        predicted['predicted_screen_roi'] = predicted.pop('measured_screen_roi')
        legacy = {'board_id': 'a', 'screen_roi': [10, 20, 30, 40], 'identity_status': 'confirmed',
                  'camera_reliability': 'verified'}
        got = evaluate_tracking(manifest([predicted, legacy]))['per_board']['a']
        self.assertEqual(got['measurement_coverage'], 0)
        self.assertEqual(got['operational_accepted_coverage'], 0)

    def test_missing_camera_or_unreliable_camera_is_not_verified(self):
        missing = board()
        del missing['camera_reliability']
        got = evaluate_tracking(manifest([missing, board(camera='degraded'), board(camera='unobservable')]))['per_board']['a']
        self.assertEqual(got['accepted_samples'], 0)
        self.assertEqual(got['measured_samples'], 3)

    def test_frame_camera_is_available_when_board_does_not_repeat_it(self):
        m = manifest([board()]); del m['frames'][0]['boards'][0]['camera_reliability']
        m['frames'][0]['camera'] = {'reliability': 'verified'}
        self.assertEqual(evaluate_tracking(m)['per_board']['a']['accepted_samples'], 1)

    def test_unknown_duration_reports_sampling_bounds_and_tail_uncertainty(self):
        got = evaluate_tracking(manifest([board(), board(identity='identity_uncertain'),
                                         board(identity='identity_uncertain'), board()], duration=3200))['per_board']['a']
        self.assertEqual(got['unknown_duration_ms']['lower_bound_ms'], 1000)
        self.assertEqual(got['unknown_duration_ms']['upper_bound_ms'], 3000)
        self.assertEqual(got['unknown_duration_ms']['sample_hold_estimate_ms'], 2000)
        self.assertFalse(got['unknown_duration_ms']['continuous_physical_truth_verified'])
        trailing = evaluate_tracking(manifest([board(), None], duration=1400))['per_board']['a']
        self.assertEqual(trailing['unknown_duration_ms']['sample_hold_estimate_ms'], 400)
        self.assertTrue(trailing['unknown_spans'][0]['tail_after_last_observation_unverified'])
        no_end = evaluate_tracking(manifest([board(), None]))['per_board']['a']
        self.assertIsNone(no_end['unknown_duration_ms']['upper_bound_ms'])

    def test_independent_identity_labels_detect_error_without_position_guess(self):
        m = manifest([board(), board(roi=[12, 23, 30, 40])])
        truth = {'frames': {
            'f0': {'a': {'observable': True, 'screen_roi': [10, 20, 30, 40], 'associated_physical_board_id': 'a'}},
            'f1': {'a': {'observable': True, 'screen_roi': [10, 20, 30, 40], 'associated_physical_board_id': 'different_physical_board'}}}}
        before = copy.deepcopy(truth)
        got = evaluate_tracking(m, truth)['per_board']['a']
        self.assertEqual(got['identity_error_rate'], .5)
        self.assertEqual(got['position_error']['median_px'], 1.5)
        self.assertEqual(got['position_error']['max_px'], 3)
        self.assertEqual(got['gt_observable_accepted_coverage'], 1)
        self.assertEqual(truth, before)

    def test_unknown_ground_truth_does_not_become_success(self):
        got = evaluate_tracking(manifest([board(), board()]), {'frames': {'f0': {'a': {'observable': False}}}})['per_board']['a']
        self.assertEqual(got['gt_observable_samples'], 0)
        self.assertIsNone(got['gt_observable_accepted_coverage'])
        self.assertIsNone(got['identity_error_rate'])
        self.assertIsNone(got['position_error']['median_px'])

    def test_false_identity_when_target_is_hidden_still_counts_as_error(self):
        truth = {'frames': {'f0': {'a': {'observable': False,
                                        'associated_physical_board_id': 'different_visible_board'}}}}
        got = evaluate_tracking(manifest([board()]), truth)['per_board']['a']
        self.assertEqual(got['gt_observable_samples'], 0)
        self.assertEqual(got['identity_error_rate'], 1)
        self.assertEqual(got['identity_labelled_accepted_samples'], 1)
        self.assertIsNone(got['position_error']['median_px'])

    def test_reacquisition_latency_requires_correct_independently_labelled_identity(self):
        truth = {'frames': {
            'f1': {'a': {'observable': True, 'associated_physical_board_id': 'other'}},
            'f2': {'a': {'observable': True, 'associated_physical_board_id': 'a'}}},
            'reacquisition_windows': [{'board_id': 'a', 'recoverable_since_ms': 500, 'deadline_ms': 2000}]}
        got = evaluate_tracking(manifest([board(identity='identity_uncertain'), board(), board()]), truth)['per_board']['a']
        self.assertEqual(got['reacquisition_latency']['median_ms'], 1500)
        self.assertEqual(got['identity_error_rate'], .5)
        no_labels = {'reacquisition_windows': truth['reacquisition_windows']}
        unknown = evaluate_tracking(manifest([board(), board(), board()]), no_labels)['per_board']['a']
        self.assertIsNone(unknown['reacquisition_latency']['median_ms'])
        self.assertEqual(unknown['reacquisition_latency']['unassessable_windows'], 1)
        self.assertIn('reacquisition_latency', unknown['unavailable_metrics'])

    def test_finite_positive_measured_roi_required(self):
        got = evaluate_tracking(manifest([board(roi=[10, float('nan'), 30, 40]),
                                         board(roi=[10, 20, 0, 40]), board(roi=[True, 20, 30, 40])]))['per_board']['a']
        self.assertEqual(got['accepted_samples'], 0)
        json.dumps(got, allow_nan=False)

    def test_rejects_nonmonotone_timestamps_duplicate_frames_or_boards(self):
        with self.assertRaises(ValueError):
            evaluate_tracking(manifest([board(), board()], times=[0, 0]))
        m = manifest([board(), board()]); m['frames'][1]['frame_id'] = 'f0'
        with self.assertRaises(ValueError):
            evaluate_tracking(m)
        m = manifest([board()]); m['frames'][0]['boards'].append(board())
        with self.assertRaises(ValueError):
            evaluate_tracking(m)

    def test_all_unknown_remains_zero_coverage_not_automatic_pass(self):
        got = evaluate_tracking(manifest([None, None, None], duration=2500))['per_board']['a']
        self.assertEqual(got['operational_accepted_coverage'], 0)
        self.assertEqual(got['unknown_samples'], 3)
        self.assertIsNone(got['identity_error_rate'])
        self.assertNotIn('passed', got)


def automatic_fixture():
    specs = mechanical_board_config()
    boards = [BoardSpec(b['board_id'], b['label'], tuple(b['roi']), b['assembly_id'], b['layer']) for b in specs]
    assemblies = [AssemblySpec(a['assembly_id'], tuple(a['viewport'])) for a in assembly_config()]
    return DynamicMechanicalTracker(boards, assemblies)


def automatic_frame(left=0, right=0, anchor_variation=0, obscured=()):
    frame = render_mechanical(positions(left, right), obscured_boards=obscured)
    outside = np.ones(frame.shape[:2], bool)
    for spec in assembly_config():
        x, y, w, h = spec['viewport']; outside[y:y+h, x:x+w] = False
    texture = np.random.default_rng(7581).integers(30, 210, size=frame.shape, dtype=np.uint8)
    # Explicit static-camera support, independent of panel texture/motion. Only
    # anchor brightness varies during recovery; physical panel pixels do not.
    values = np.clip(texture.astype(np.int16) + anchor_variation, 0, 255).astype(np.uint8)
    frame[outside] = values[outside]
    return frame


def automatic_observe(tracker, frame, index):
    fid = f'f{index:04d}'
    return tracker.observe(frame, fid, index * 1000, f'frames/{fid}.png',
                           {b.board_id: f'crops/{b.board_id}/{fid}.png' for b in tracker.boards})


def get_state(record, board_id):
    return next(s for s in record['boards'] if s['board_id'] == board_id)


class AutomaticCoordinatorSafetyTests(unittest.TestCase):
    def test_clean_automatic_motion_has_dynamic_measurements_not_all_unknown(self):
        tracker = automatic_fixture()
        records = [automatic_observe(tracker, automatic_frame(left=offset), i)
                   for i, offset in enumerate([0, 20, 35, 20, 0])]
        metrics = evaluate_tracking({'config': {'boards': [b.to_dict() for b in tracker.boards]}, 'frames': records})
        for b in tracker.boards:
            self.assertGreaterEqual(metrics['per_board'][b.board_id]['operational_accepted_coverage'], .95)
        y_values = [get_state(r, 'left_outer')['measured_screen_roi'][1] for r in records]
        self.assertGreater(len(set(y_values)), 1)
        self.assertTrue(any(e['kind'] == 'board_motion_event' for e in tracker.events))
        self.assertFalse(any(e['kind'] == 'board_content_change_event' for e in tracker.events))
        self.assertFalse(tracker.writing_runs)
        self.assertTrue(all(not s['comparison_eligibility']['eligible'] for r in records for s in r['boards']))

    def test_transient_camera_failure_keeps_searching_and_can_recover(self):
        tracker = automatic_fixture()
        automatic_observe(tracker, automatic_frame(), 0)
        before = copy.deepcopy(tracker.versions)
        bad = automatic_observe(tracker, np.zeros((480, 800, 3), np.uint8), 1)
        self.assertNotEqual(bad['camera']['reliability'], 'verified')
        records = [automatic_observe(tracker, automatic_frame(anchor_variation=i), i + 1) for i in range(1, 5)]
        self.assertTrue(all(r['camera']['search_attempted'] for r in records))
        self.assertEqual(records[-1]['camera']['reliability'], 'verified')
        self.assertTrue(any(s['identity_status'] == 'confirmed' for s in records[-1]['boards']))
        self.assertEqual(tracker.versions[:len(before)], before)
        self.assertFalse(any(e['kind'] == 'board_content_change_event' for e in tracker.events))
        self.assertEqual(len(tracker.position_history), 6 * 4)

    def test_identity_recovery_starts_new_baseline_without_overwriting_history(self):
        tracker = automatic_fixture()
        first = automatic_observe(tracker, automatic_frame(), 0)
        original_versions = copy.deepcopy(tracker.versions)
        hidden = automatic_observe(tracker, automatic_frame(obscured=('left_outer', 'left_inner')), 1)
        lost = get_state(hidden, 'left_outer')
        self.assertEqual(lost['identity_status'], 'identity_uncertain')
        self.assertIsNone(lost['measured_screen_roi'])
        self.assertIsNotNone(lost['predicted_screen_roi'])
        recovered = automatic_observe(tracker, automatic_frame(), 2)
        final = get_state(recovered, 'left_outer')
        initial = get_state(first, 'left_outer')
        self.assertEqual(final['identity_status'], 'confirmed')
        self.assertGreater(final['comparison_epoch'], initial['comparison_epoch'])
        self.assertNotEqual(final['baseline_id'], initial['baseline_id'])
        self.assertEqual(tracker.versions[:len(original_versions)], original_versions)
        self.assertFalse(any(e['kind'] == 'board_content_change_event' for e in tracker.events))
        self.assertEqual(len(tracker.position_history), 12)

    def test_automatic_coordinator_rejects_future_ground_truth(self):
        tracker = automatic_fixture()
        automatic_observe(tracker, automatic_frame(), 0)
        with self.assertRaises(ValueError):
            tracker.observe(automatic_frame(left=20), 'f0001', 1000, 'frames/f0001.png', {},
                            manual_positions=positions(20))

    def test_unknown_physical_aperture_is_not_fabricated_from_search_window(self):
        tracker = automatic_fixture()
        record = automatic_observe(tracker, automatic_frame(), 0)
        self.assertTrue(all(s['visibility'] == 'unknown' for s in record['boards']))
        self.assertTrue(all(s['visible_fraction'] is None and s['occlusion_ratio'] is None for s in record['boards']))
        self.assertTrue(all(s['visible_regions'] == [] for s in record['boards']))


class AutomaticVideoToolchainTests(unittest.TestCase):
    def test_ffmpeg_video_automatic_large_displacements_without_future_truth(self):
        # Four nonoverlapping physical panels, with enough travel to keep all
        # four observable during the clean 310px / 240px fixture. Truth is used
        # by this renderer and by the evaluator only, never the tracker config.
        initial = [dict(board_id=f'{side}_{layer}', label=f'{side} {layer}',
                        assembly_id=f'{side}_assembly', layer=layer, roi=[x, y, 300, 180])
                   for side, x in [('left', 40), ('right', 440)]
                   for layer, y in [('outer', 50), ('inner', 900)]]
        assemblies = [{'assembly_id': f'{side}_assembly', 'search_region': [x, 20, 340, 1150]}
                      for side, x in [('left', 20), ('right', 420)]]
        background = np.random.default_rng(349701).integers(20, 220, size=(1200, 800, 3), dtype=np.uint8)
        samples, truths = [], []
        for index, (left, right) in enumerate([(0, 0), (310, 0), (240, 240), (0, 240), (0, 0)]):
            image = background.copy()
            for assembly in assemblies:
                x, y, w, h = assembly['search_region']; image[y:y+h, x:x+w] = (25, 30, 25)
            truth = {}
            for b in initial:
                x, y, w, h = b['roi']; shift = left if b['assembly_id'] == 'left_assembly' else right
                y += shift if b['layer'] == 'outer' else -shift
                image[y:y+h, x:x+w] = _board_surface(b['board_id'])
                truth[b['board_id']] = [x, y, w, h]
            samples.append(SimpleNamespace(frame=image))
            truths.append(truth)
        with tempfile.TemporaryDirectory(prefix='icourse-auto-video-') as directory:
            root = Path(directory)
            source = root / 'automatic large motion.avi'
            write_video(source, samples, fps=4)
            config = {'tracking_mode': 'automatic_tracking', 'sample_interval_ms': 1000,
                      'boards': initial, 'assemblies': assemblies,
                      'camera': {'anchor_regions': [[0, 0, 800, 20], [0, 1170, 800, 30], [360, 20, 60, 1150]]}}
            self.assertNotIn('position_keyframes', config)
            configuration = root / 'initial-only.json'; configuration.write_text(json.dumps(config))
            output = root / 'automatic-review'
            result = json.loads(run_video(source, configuration, output).read_text())
            self.assertEqual(len(result['frames']), 5)
            self.assertEqual(len(list((output / 'frames').glob('*.png'))), 5)
            self.assertEqual(result['source']['metadata_source'], 'ffprobe')
            self.assertIn('ffmpeg', result['source']['tools'])
            self.assertTrue(all(f['native_pts'] is not None and f['native_time_base'] for f in result['frames']))
            self.assertEqual([f['timestamp_ms'] for f in result['frames']], [0, 1000, 2000, 3000, 4000])
            ground_truth = {'frames': {}}
            for frame, truth in zip(result['frames'], truths):
                frame_truth = {}
                for measured in frame['boards']:
                    roi = measured['measured_screen_roi']
                    associated = None
                    if roi is not None:
                        cx, cy = roi[0] + roi[2] / 2, roi[1] + roi[3] / 2
                        # Exact synthetic occupancy labels, independent of the
                        # algorithm's claimed identity, not nearest real-video ROI.
                        associated = next((bid for bid, (x, y, w, h) in truth.items()
                                           if x <= cx < x+w and y <= cy < y+h), 'background')
                    bid = measured['board_id']
                    frame_truth[bid] = {'observable': True, 'screen_roi': truth[bid],
                                        'associated_physical_board_id': associated}
                ground_truth['frames'][frame['frame_id']] = frame_truth
            metrics = evaluate_tracking(result, ground_truth)
            for b in initial:
                actual = metrics['per_board'][b['board_id']]
                self.assertGreaterEqual(actual['gt_observable_accepted_coverage'], .95)
                self.assertEqual(actual['identity_error_rate'], 0)
                self.assertLessEqual(actual['position_error']['median_px'], 2)
                self.assertLessEqual(actual['position_error']['p95_px'], 4)
            self.assertFalse(any(e['kind'] == 'board_content_change_event' for e in result['events']))
            self.assertTrue((output / 'review.html').is_file())
            self.assertEqual(result['writing_runs'], [])


class CoordinatorContractRegressionTests(unittest.TestCase):
    """Mock/oracle-assisted state-flow tests, NOT automatic tracking accuracy."""

    @staticmethod
    def camera_result(reliability='verified', transform=None):
        return {'reliability': reliability,
                'measured_transform_to_scene': np.eye(3).tolist() if transform is None and reliability == 'verified' else transform,
                'geometry_epoch': 0, 'scene_id': 0, 'search_attempted': True,
                'reason_codes': ['test_fixture_independent_camera_result']}

    def test_pure_camera_translation_changes_screen_roi_without_mechanical_motion_event(self):
        tracker = automatic_fixture()
        image = automatic_frame()
        shifted = cv2.warpAffine(image, np.array([[1, 0, 3], [0, 1, 4]], dtype=float), (800, 480))
        known_geometry = self.camera_result(transform=[[1, 0, -3], [0, 1, -4], [0, 0, 1]])
        # Camera geometry is an oracle fixture here; board appearance matching is
        # real. This tests coordinator semantics, not camera estimation accuracy.
        with patch.object(tracker.camera, 'observe', side_effect=[self.camera_result(), known_geometry]):
            first = automatic_observe(tracker, image, 0)
            versions = copy.deepcopy(tracker.versions)
            second = automatic_observe(tracker, shifted, 1)
        for b in tracker.boards:
            before, after = get_state(first, b.board_id), get_state(second, b.board_id)
            self.assertEqual(after['identity_status'], 'confirmed')
            self.assertEqual(after['measured_screen_roi'], [before['measured_screen_roi'][0] + 3,
                                                          before['measured_screen_roi'][1] + 4,
                                                          *before['measured_screen_roi'][2:]])
            self.assertEqual(after['motion_status'], 'stationary')
            self.assertLessEqual(float(np.linalg.norm(after['delta_scene_px'])), .75)
        self.assertFalse(any(e['kind'] == 'board_motion_event' for e in tracker.events))
        self.assertFalse(any(e['kind'] == 'board_content_change_event' for e in tracker.events))
        self.assertEqual(tracker.versions, versions)

    def test_first_unverified_camera_later_verified_still_initializes_missing_baseline_once(self):
        tracker = automatic_fixture()
        image = automatic_frame()
        rows = tracker.tracker.observe(image, 'upstream-reference', 0, self.camera_result())
        # Keep raw identity and comparison_epoch unchanged across the camera
        # transition. The coordinator must notice that no baseline exists yet.
        upstream = []
        for index in range(3):
            current = copy.deepcopy(rows)
            for state in current:
                state['frame_id'] = f'f{index:04d}'
                state['timestamp_ms'] = index * 1000
                state['comparison_epoch'] = 0
                state['identity_status'] = 'confirmed'
            upstream.append(current)
        cameras = [self.camera_result('degraded'), self.camera_result(), self.camera_result()]
        with patch.object(tracker.camera, 'observe', side_effect=cameras), \
                patch.object(tracker.tracker, 'observe', side_effect=upstream):
            first = automatic_observe(tracker, image, 0)
            self.assertEqual(tracker.versions, [])
            self.assertTrue(all(s['baseline_id'] is None for s in first['boards']))
            second = automatic_observe(tracker, image, 1)
            saved_versions = copy.deepcopy(tracker.versions)
            third = automatic_observe(tracker, image, 2)
        self.assertEqual(len(saved_versions), 4)
        self.assertEqual(tracker.versions, saved_versions)
        for b in tracker.boards:
            mid, final = get_state(second, b.board_id), get_state(third, b.board_id)
            self.assertIsNotNone(mid['baseline_id'])
            self.assertEqual(final['baseline_id'], mid['baseline_id'])
            self.assertEqual(final['comparison_epoch'], 0)
            self.assertEqual(final['identity_status'], 'confirmed')
        self.assertEqual(sum(e['kind'] == 'content_baseline_initialized' for e in tracker.events), 4)
        self.assertFalse(any(e['kind'] == 'board_content_change_event' for e in tracker.events))


if __name__ == '__main__':
    unittest.main()
