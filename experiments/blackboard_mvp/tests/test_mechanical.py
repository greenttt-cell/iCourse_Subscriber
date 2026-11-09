"""Paired-panel geometry and evidence semantics; synthetic inputs only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from blackboard_mvp.core import BoardSpec
from blackboard_mvp.mechanics import AssemblySpec, SlidingBoardTracker, intersect, subtract
from blackboard_mvp.mechanical_synthetic import (
    assembly_config, mechanical_board_config, mechanical_samples, positions,
    render_mechanical, write_mechanical_demo,
)


def make_tracker():
    boards = [BoardSpec(b['board_id'], b['label'], tuple(b['roi']), b['assembly_id'], b['layer'])
              for b in mechanical_board_config()]
    assemblies = [AssemblySpec(a['assembly_id'], tuple(a['viewport'])) for a in assembly_config()]
    return SlidingBoardTracker(boards, assemblies)


def observe(tracker, timestamp, pos=None, frame=None, manual=True):
    pos = positions() if pos is None else pos
    frame = render_mechanical(pos) if frame is None else frame
    frame_id = f'f{timestamp:06d}'
    return tracker.observe(frame, frame_id, timestamp, f'frames/{frame_id}.png',
                           {b.board_id: f'crops/{b.board_id}/{frame_id}.png' for b in tracker.boards},
                           manual_positions=pos if manual else None)


def state(record, bid):
    return next(b for b in record['boards'] if b['board_id'] == bid)


def contents(tracker, bid=None):
    return [e for e in tracker.events if e['kind'] == 'board_content_change_event'
            and (bid is None or e['board_id'] == bid)]


class MechanicalTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = make_tracker()

    def test_initial_identity_is_physical_and_assembly_layer_explicit(self):
        record = observe(self.tracker, 0)
        self.assertEqual(len(record['boards']), 4)
        self.assertEqual({b['assembly_id'] for b in record['boards']}, {'left_assembly', 'right_assembly'})
        self.assertEqual(len(self.tracker.versions), 4)
        self.assertTrue(all(v['content_status'] == 'initial_existing' for v in self.tracker.versions))
        self.assertTrue(all(b['identity_status'] == 'confirmed' for b in record['boards']))
        self.assertTrue(all(b['visibility'] == 'visible' for b in record['boards']))

    def test_a_downward_outer_and_upward_inner_are_motion_not_content(self):
        observe(self.tracker, 0)
        record = observe(self.tracker, 1000, positions(40))
        motions = [e for e in self.tracker.events if e['kind'] == 'board_motion_event']
        self.assertEqual({e['board_id']: e['delta_y'] for e in motions}, {'left_outer': 40, 'left_inner': -40})
        self.assertFalse(contents(self.tracker))
        self.assertEqual(state(record, 'left_inner')['motion_status'], 'moving')
        self.assertEqual(len(self.tracker.versions), 4)
        for e in motions:
            self.assertEqual(e['interval_ms'], [0, 1000])
            self.assertIsNotNone(e['before_frame_id'])
            self.assertIsNotNone(e['after_crop_path'])
            self.assertFalse(e['mechanical_prior']['identity_evidence'])

    def test_b_upward_outer_and_downward_inner_keep_independent_ids(self):
        observe(self.tracker, 0)
        observe(self.tracker, 1000, positions(40))
        observe(self.tracker, 2000, positions(20))
        motions = [e for e in self.tracker.events if e['kind'] == 'board_motion_event' and e['observed_at_ms'] == 2000]
        self.assertEqual({e['board_id']: e['delta_y'] for e in motions}, {'left_outer': -20, 'left_inner': 20})
        self.assertFalse(contents(self.tracker))

    def test_c_large_partial_occlusion_reports_only_remaining_visible_strip(self):
        observe(self.tracker, 0)
        record = observe(self.tracker, 1000, positions(85))
        inner = state(record, 'left_inner')
        self.assertEqual(inner['visibility'], 'partially_occluded')
        self.assertAlmostEqual(inner['visible_fraction'], 20 / 180)
        self.assertEqual(inner['visible_regions'], [[40, 315, 300, 20]])
        self.assertGreater(inner['occlusion_ratio'], .85)
        self.assertFalse(contents(self.tracker, 'left_inner'))

    def test_d_complete_occlusion_pauses_comparison_and_hidden_edit_is_unobservable(self):
        observe(self.tracker, 0)
        before = copy.deepcopy(self.tracker.versions)
        p = positions(95)
        record = observe(self.tracker, 1000, p, render_mechanical(p, hidden_edit_boards=('left_inner',)))
        inner = state(record, 'left_inner')
        self.assertEqual(inner['visibility'], 'occluded')
        self.assertEqual(inner['visible_fraction'], 0)
        self.assertEqual(inner['visible_regions'], [])
        self.assertEqual(inner['comparison_status'], 'paused_occluded')
        observe(self.tracker, 2000, p, render_mechanical(p, hidden_edit_boards=('left_inner',)))
        self.assertFalse(contents(self.tracker, 'left_inner'))
        self.assertEqual(self.tracker.versions, before)

    def test_e_f_reappearance_uses_original_board_id_and_preserves_history(self):
        initial = observe(self.tracker, 0)
        initial_version = state(initial, 'left_inner')['version_id']
        old_versions = copy.deepcopy(self.tracker.versions)
        observe(self.tracker, 1000, positions(95))
        partial = observe(self.tracker, 2000, positions(60))
        restored = observe(self.tracker, 3000, positions())
        self.assertEqual(state(partial, 'left_inner')['board_id'], 'left_inner')
        self.assertEqual(state(restored, 'left_inner')['version_id'], initial_version)
        self.assertEqual(state(restored, 'left_inner')['identity_status'], 'confirmed')
        self.assertEqual(state(restored, 'left_inner')['visibility'], 'visible')
        self.assertEqual(self.tracker.versions, old_versions)
        history = [h for h in self.tracker.position_history if h['board_id'] == 'left_inner']
        self.assertEqual([h['visibility'] for h in history], ['visible', 'occluded', 'partially_occluded', 'visible'])
        self.assertTrue(any(e.get('reappearance_confirmed') for e in self.tracker.events))

    def test_g_writing_during_motion_is_deferred_then_residual_keeps_broad_interval(self):
        observe(self.tracker, 0)
        p = positions(35)
        moving = observe(self.tracker, 1000, p, render_mechanical(p, minus_boards=('left_outer',)))
        self.assertFalse(contents(self.tracker, 'left_outer'))
        self.assertEqual(state(moving, 'left_outer')['comparison_status'], 'deferred_during_motion')
        observe(self.tracker, 2000, p, render_mechanical(p, minus_boards=('left_outer',)))
        changes = contents(self.tracker, 'left_outer')
        self.assertEqual(len(changes), 1)
        event = changes[0]
        self.assertEqual(event['interval_ms'], [0, 2000])
        self.assertEqual(event['comparison_coordinate_space'], 'physical_board_local')
        self.assertEqual(event['compensation_status'], 'translation_compensated')
        self.assertIn('change_time_within_evidence_interval_unknown', event['uncertainty'])
        self.assertEqual(event['change_regions'][0]['changed_pixels'], 12)

    def test_h_components_move_independently(self):
        observe(self.tracker, 0)
        observe(self.tracker, 1000, positions(40))
        observe(self.tracker, 2000, positions(40, 45))
        motions = [e for e in self.tracker.events if e['kind'] == 'board_motion_event' and e['observed_at_ms'] == 2000]
        self.assertEqual({e['board_id'] for e in motions}, {'right_outer', 'right_inner'})
        self.assertFalse(contents(self.tracker))

    def test_i_feature_failure_never_merges_physical_ids(self):
        initial = observe(self.tracker, 0)
        p = positions(20)
        record = observe(self.tracker, 1000, p,
                         render_mechanical(p, obscured_boards=('left_outer', 'left_inner')), manual=False)
        self.assertEqual({b['board_id'] for b in record['boards']}, {b['board_id'] for b in initial['boards']})
        self.assertEqual(state(record, 'left_outer')['identity_status'], 'identity_uncertain')
        self.assertEqual(state(record, 'left_inner')['identity_status'], 'identity_uncertain')
        self.assertFalse(contents(self.tracker, 'left_outer'))
        self.assertFalse(contents(self.tracker, 'left_inner'))
        self.assertEqual(len(self.tracker.versions), 4)

    def test_j_hidden_pixels_are_excluded_from_stable_partial_comparison(self):
        p = positions(85)
        observe(self.tracker, 0, p)
        plain = render_mechanical(p)
        edited = render_mechanical(p, hidden_edit_boards=('left_inner',))
        # The changed mark is physically behind the outer board.
        self.assertTrue(np.array_equal(plain, edited))
        record = observe(self.tracker, 1000, p, edited)
        self.assertEqual(state(record, 'left_inner')['comparison_status'], 'shared_visible_only')
        self.assertFalse(contents(self.tracker, 'left_inner'))

    def test_visible_small_symbol_while_partially_occluded_is_preserved(self):
        p = positions(60)
        observe(self.tracker, 0, p)
        frame = render_mechanical(p)
        x, y, _, _ = p['left_inner']
        # The lower physical-local strip is visible; 12px exceeds local threshold.
        cv2.line(frame, (x+80, y+164), (x+91, y+164), (240, 240, 230), 1)
        observe(self.tracker, 1000, p, frame)
        event = contents(self.tracker, 'left_inner')[0]
        self.assertEqual(event['change_regions'][0]['bbox'], [80, 164, 12, 1])
        self.assertLess(event['compared_pixel_count'], 300 * 180)
        self.assertTrue(all(r[1] >= 70 for r in event['compared_visible_regions']))

    def test_persistent_hidden_edit_is_only_candidate_after_reappearance_with_wide_interval(self):
        observe(self.tracker, 0)
        hidden = positions(95)
        observe(self.tracker, 1000, hidden,
                render_mechanical(hidden, hidden_edit_boards=('left_inner',)))
        self.assertFalse(contents(self.tracker, 'left_inner'))
        restored = positions()
        edited = render_mechanical(restored, hidden_edit_boards=('left_inner',))
        observe(self.tracker, 2000, restored, edited)
        self.assertFalse(contents(self.tracker, 'left_inner'))
        observe(self.tracker, 3000, restored, edited)
        event = contents(self.tracker, 'left_inner')[0]
        self.assertEqual(event['interval_ms'], [0, 3000])
        self.assertEqual(event['before_frame_id'], 'f000000')
        self.assertIn('change_time_within_evidence_interval_unknown', event['uncertainty'])
        self.assertEqual(event['comparison_coordinate_space'], 'physical_board_local')
        # The old observation is immutable and still points before the hiding.
        self.assertEqual(self.tracker.versions[0]['content_status'], 'initial_existing')

    def test_first_ever_revealed_pixels_do_not_claim_content_was_newly_written(self):
        hidden = positions(95)
        initial = observe(self.tracker, 0, hidden)
        self.assertIsNone(state(initial, 'left_inner')['version_id'])
        shown = positions()
        edited = render_mechanical(shown, hidden_edit_boards=('left_inner',))
        observe(self.tracker, 1000, shown, edited)
        record = observe(self.tracker, 2000, shown, edited)
        self.assertFalse(contents(self.tracker, 'left_inner'))
        versions = [v for v in self.tracker.versions if v['board_id'] == 'left_inner']
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['content_status'], 'first_observed_after_occlusion')
        self.assertEqual(state(record, 'left_inner')['board_id'], 'left_inner')

    def test_auto_outer_match_is_appearance_based_not_opposed_motion_identity(self):
        observe(self.tracker, 0, manual=False)
        record = observe(self.tracker, 1000, positions(20), manual=False)
        outer = state(record, 'left_outer')
        self.assertEqual(outer['screen_roi'], positions(20)['left_outer'])
        self.assertEqual(outer['identity_status'], 'confirmed')
        self.assertEqual(outer['tracking_source'], 'appearance_feature_translation')
        # Withholding distinctive inner evidence must never let linkage establish
        # its identity, even while the outer's movement has been measured.
        t = make_tracker()
        observe(t, 0, manual=False)
        record = observe(t, 1000, positions(20),
                         render_mechanical(positions(20), obscured_boards=('left_inner',)), manual=False)
        inner = state(record, 'left_inner')
        self.assertEqual(inner['identity_status'], 'identity_uncertain')
        self.assertFalse(inner['mechanical_prior']['identity_evidence'])
        self.assertFalse(contents(t, 'left_inner'))

    def test_uncertain_reappearance_requires_direct_evidence_or_manual_confirmation(self):
        observe(self.tracker, 0)
        observe(self.tracker, 1000, positions(95))
        uncertain = observe(self.tracker, 2000, positions(),
                            render_mechanical(positions(), obscured_boards=('left_inner',)), manual=False)
        self.assertEqual(state(uncertain, 'left_inner')['identity_status'], 'identity_uncertain')
        self.assertFalse(contents(self.tracker, 'left_inner'))
        confirmed = observe(self.tracker, 3000, positions())
        self.assertEqual(state(confirmed, 'left_inner')['identity_status'], 'confirmed')
        self.assertEqual(state(confirmed, 'left_inner')['version_id'], 'left_inner:v0000')

    def test_opposed_motion_is_tolerant_and_never_forced_to_exact_equality(self):
        observe(self.tracker, 0)
        p = positions(40)
        p['left_inner'][1] += 2
        record = observe(self.tracker, 1000, p)
        self.assertEqual(state(record, 'left_inner')['screen_roi'][1], 202)
        prior = state(record, 'left_outer')['mechanical_prior']
        self.assertEqual(prior['residual_pixels'], 2)
        self.assertEqual(prior['status'], 'consistent')
        self.assertFalse(prior['identity_evidence'])

    def test_inconsistent_motion_prior_is_reviewed_without_forcing_positions(self):
        observe(self.tracker, 0)
        p = positions(40)
        p['left_inner'][1] = 230
        record = observe(self.tracker, 1000, p)
        self.assertEqual(state(record, 'left_inner')['screen_roi'][1], 230)
        # Direct human confirmation remains identity authority; the noisy
        # mechanical prior reports inconsistency without overriding that input.
        outer = state(record, 'left_outer')
        self.assertEqual(outer['identity_status'], 'confirmed')
        self.assertEqual(outer['mechanical_prior']['status'], 'inconsistent_requires_review')
        self.assertFalse(outer['mechanical_prior']['identity_evidence'])
        self.assertFalse(contents(self.tracker))

    def test_automatic_motion_failure_keeps_original_evidence_and_no_content_version(self):
        observe(self.tracker, 0)
        before = copy.deepcopy(self.tracker.versions)
        p = positions(35)
        frame = render_mechanical(p, obscured_boards=('left_outer',))
        record = observe(self.tracker, 1000, p, frame, manual=False)
        review = [e for e in self.tracker.events if e['board_id'] == 'left_outer' and e['observed_at_ms'] == 1000]
        self.assertTrue(review)
        self.assertTrue(all(e['before_crop_path'] and e['after_crop_path'] for e in review))
        self.assertEqual(state(record, 'left_outer')['comparison_status'], 'paused_identity_uncertain')
        self.assertEqual(self.tracker.versions, before)

    def test_all_observations_and_positions_remain_append_only(self):
        samples = mechanical_samples()
        for sample in samples:
            observe(self.tracker, sample.timestamp_ms, sample.positions, sample.frame,
                    manual=sample.timestamp_ms != 9000)
        self.assertEqual(len(self.tracker.position_history), len(samples) * 4)
        for bid in {b.board_id for b in self.tracker.boards}:
            history = [p for p in self.tracker.position_history if p['board_id'] == bid]
            self.assertEqual([h['timestamp_ms'] for h in history], list(range(0, 12000, 1000)))
        self.assertEqual(self.tracker.versions[0]['content_status'], 'initial_existing')
        events = self.tracker.events
        self.assertEqual([e['observed_at_ms'] for e in events], sorted(e['observed_at_ms'] for e in events))
        multi = [e for e in events if e['observed_at_ms'] == 1000]
        self.assertTrue(all(e['order_status'] == 'undetermined_within_interval' for e in multi))

    def test_unknown_manual_board_and_size_change_are_rejected(self):
        frame = render_mechanical(positions())
        with self.assertRaises(ValueError):
            self.tracker.observe(frame, 'a', 0, 'frames/a.png', {}, manual_positions={'other': [0, 0, 1, 1]})
        invalid = positions()
        invalid['left_outer'][2] += 1
        with self.assertRaises(ValueError):
            self.tracker.observe(frame, 'a', 0, 'frames/a.png', {}, manual_positions=invalid)

    def test_assemblies_require_one_inner_and_one_outer_without_fixed_side_count(self):
        bs = [b for b in self.tracker.boards if b.assembly_id == 'left_assembly']
        tracker = SlidingBoardTracker(bs, [AssemblySpec('left_assembly', (20, 35, 340, 400))])
        self.assertEqual(len(tracker.boards), 2)
        with self.assertRaises(ValueError):
            SlidingBoardTracker(bs[:1], [AssemblySpec('left_assembly', (20, 35, 340, 400))])

    def test_invalid_mechanical_identifiers_layers_viewports_and_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            AssemblySpec('../unsafe', (0, 0, 20, 20))
        with self.assertRaises(ValueError):
            AssemblySpec('safe', (0, 0, 0, 20))
        with self.assertRaises(ValueError):
            BoardSpec('b', '板', (0, 0, 20, 20), 'a', 'front')
        boards = self.tracker.boards
        assemblies = list(self.tracker.assemblies.values())
        with self.assertRaises(ValueError):
            SlidingBoardTracker(boards + [boards[0]], assemblies)
        with self.assertRaises(ValueError):
            SlidingBoardTracker(boards, assemblies + [assemblies[0]])
        bad = make_tracker()
        bad.assemblies['left_assembly'] = AssemblySpec('left_assembly', (0, 0, 5000, 5000))
        with self.assertRaises(ValueError):
            observe(bad, 0)


class MechanicalGeometryTests(unittest.TestCase):
    def test_rectangle_subtraction_does_not_keep_hidden_overlap(self):
        parts = subtract([0, 0, 100, 100], [20, 30, 60, 40])
        self.assertEqual(sum(r[2] * r[3] for r in parts), 7600)
        self.assertTrue(all(intersect(r, [20, 30, 60, 40]) is None for r in parts))
        self.assertEqual(subtract([0, 0, 100, 100], [0, 0, 100, 100]), [])


class MechanicalToolchainTests(unittest.TestCase):
    def test_synthetic_demo_through_real_ffmpeg_ffprobe_and_manifest(self):
        from blackboard_mvp.runner import run_video
        with tempfile.TemporaryDirectory(prefix='icourse-mechanical-toolchain-') as root:
            root = Path(root)
            fixture = write_mechanical_demo(root / 'input')
            manifest_path = run_video(Path(fixture['video']), Path(fixture['config']), root / 'review')
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest['frames']), 12)
            self.assertEqual(len(list((root / 'review' / 'frames').glob('*.png'))), 12)
            self.assertEqual(len(list((root / 'review' / 'crops').glob('*/*.png'))), 48)
            self.assertTrue(all(f['native_pts'] is not None for f in manifest['frames']))
            self.assertEqual([f['timestamp_ms'] for f in manifest['frames']], list(range(0, 12000, 1000)))
            self.assertEqual(state(manifest['frames'][4], 'left_inner')['visibility'], 'occluded')
            self.assertEqual(state(manifest['frames'][6], 'left_inner')['version_id'], 'left_inner:v0000')
            self.assertFalse(any(e['kind'] == 'board_content_change_event' and e['board_id'] == 'left_inner'
                                 for e in manifest['events']))
            self.assertTrue((root / 'review' / 'review.html').is_file())
            self.assertTrue(all((root / 'review' / f['image_path']).is_file() for f in manifest['frames']))
            with self.assertRaises(FileExistsError):
                run_video(Path(fixture['video']), Path(fixture['config']), root / 'review')


if __name__ == '__main__':
    unittest.main()
