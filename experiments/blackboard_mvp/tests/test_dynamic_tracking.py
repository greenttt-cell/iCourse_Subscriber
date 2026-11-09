"""Automatic tracking tests: future position truth is used only by assertions.

Oracle geometry tests are explicitly named and do not count as auto coverage.
"""
import json
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blackboard_mvp.core import BoardSpec
from blackboard_mvp.mechanics import AssemblySpec
from blackboard_mvp.mechanical_synthetic import assembly_config, mechanical_board_config, positions, render_mechanical, _board_surface
from blackboard_mvp.tracking import DynamicBoardTracker


CAMERA={'reliability':'verified','measured_transform_to_scene':np.eye(3).tolist(),'geometry_epoch':0,'scene_id':'scene-0'}


def tracker():
    bs=[BoardSpec(b['board_id'],b['label'],tuple(b['roi']),b['assembly_id'],b['layer']) for b in mechanical_board_config()]
    ass=[AssemblySpec(a['assembly_id'],tuple(a['viewport'])) for a in assembly_config()]
    return DynamicBoardTracker(bs,ass)


def record(rows,bid): return next(r for r in rows if r['board_id']==bid)


class AutomaticDynamicTrackingTests(unittest.TestCase):
    def test_returned_records_do_not_mutate_internal_trusted_or_history_state(self):
        t=tracker();rows=t.observe(render_mechanical(positions()),'a',0,CAMERA)
        row=record(rows,'left_outer')
        row['hypotheses'][0]['scene_origin'][1]=9999
        row['last_verified_screen_roi'][1]=9999
        self.assertEqual(t._states['left_outer']['_scene_origin'][1],50.)
        self.assertEqual(t._states['left_outer']['last_verified_screen_roi'][1],50)
        initial=record(t.history,'left_outer')
        self.assertEqual(initial['hypotheses'][0]['scene_origin'][1],50.)
        self.assertEqual(initial['last_verified_screen_roi'][1],50)

    def test_rejected_cross_board_candidates_never_pollute_trusted_state(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        saved={bid:{'origin':list(s['_scene_origin']),'velocity':list(s['_velocity']),
                    'comparison_epoch':s['comparison_epoch'],'last':list(s['last_verified_screen_roi'])}
               for bid,s in t._states.items()}
        def conflicting(frame,board,target,camera_transform,outer_roi):
            x,y,w,h=board.roi
            candidate={'screen_roi':[x,y+45,w,h],'scene_origin':[float(x),float(y+45)],
                'inliers':20,'matches':20,'support_spread_px':[100.,80.],
                'translation_residual_px':0.,'appearance':{'appearance_fraction':1.},'ranking_score':20.,
                'target_points':[[float(100+i*3),float(100+i*2)] for i in range(20)],
                'reference_frame_id':'a'}
            return [candidate],{'reason':'mock_competing_evidence','matches':20}
        with patch.object(t,'_hypotheses',side_effect=conflicting):
            rows=t.observe(render_mechanical(positions()),'b',1000,CAMERA)
        for row in rows:
            bid=row['board_id'];internal=t._states[bid]
            self.assertEqual(row['identity_status'],'identity_uncertain')
            self.assertIsNone(row['measured_screen_roi'])
            self.assertEqual(row['last_verified_screen_roi'],saved[bid]['last'])
            self.assertEqual(internal['_scene_origin'],saved[bid]['origin'])
            self.assertEqual(internal['_velocity'],saved[bid]['velocity'])
            self.assertEqual(row['comparison_epoch'],saved[bid]['comparison_epoch'])
            self.assertEqual(row['identity_epoch'],0)
            self.assertFalse(row['baseline_reset_required'])
            self.assertIsNone(row['delta_scene_px'])
        # The next prediction must be based on the original trusted zero-speed
        # state rather than the rejected 45px candidate.
        blank=np.full((480,800,3),25,np.uint8)
        later=t.observe(blank,'c',2000,CAMERA)
        for row in later:self.assertEqual(row['predicted_screen_roi'],saved[row['board_id']]['last'])

    def test_large_310_and_240_pixel_motion_is_measured_without_truth_input(self):
        boards=[BoardSpec(f'{side}_{layer}',f'{side} {layer}',(x,y,300,180),side+'_assembly',layer)
                for side,x in [('left',40),('right',440)] for layer,y in [('outer',50),('inner',600)]]
        assemblies=[AssemblySpec(side+'_assembly',(x-20,20,340,900)) for side,x in [('left',40),('right',440)]]
        t=DynamicBoardTracker(boards,assemblies)
        def image(shift):
            frame=np.full((950,800,3),25,np.uint8)
            for layer in ('inner','outer'):
                for b in boards:
                    if b.layer!=layer:continue
                    x,y,w,h=b.roi
                    if b.assembly_id=='left_assembly':y+=shift if layer=='outer' else -shift
                    frame[y:y+h,x:x+w]=_board_surface(b.board_id)
            return frame
        for i,shift in enumerate([0,310,310,0,240,0]):
            rows=t.observe(image(shift),f'large-{i}',i*1000,CAMERA)
            outer=record(rows,'left_outer')
            self.assertEqual(outer['identity_status'],'confirmed',outer['tracking_evidence'])
            self.assertEqual(outer['measured_screen_roi'],[40,50+shift,300,180])

    def test_camera_translation_compensated_before_board_motion(self):
        t=tracker();initial=render_mechanical(positions())
        t.observe(initial,'a',0,CAMERA)
        # Screen moved by (3, 4), while all physical boards stay in scene.
        moved=cv2.warpAffine(initial,np.array([[1,0,3],[0,1,4]],float),(800,480))
        camera={**CAMERA,'measured_transform_to_scene':[[1,0,-3],[0,1,-4],[0,0,1]]}
        rows=t.observe(moved,'b',1000,camera)
        for r in rows:
            self.assertEqual(r['identity_status'],'confirmed',r['tracking_evidence'])
            self.assertEqual(r['motion_status'],'stationary')
            b=next(b for b in t.boards if b.board_id==r['board_id'])
            self.assertEqual(r['measured_screen_roi'],[b.roi[0]+3,b.roi[1]+4,*b.roi[2:]])

    def test_camera_scale_changes_screen_extent_without_inventing_board_motion(self):
        t=tracker();initial=render_mechanical(positions())
        t.observe(initial,'a',0,CAMERA)
        forward=np.vstack((cv2.getRotationMatrix2D((400,240),0,1.02),[0,0,1]))
        moved=cv2.warpPerspective(initial,forward,(800,480))
        camera={**CAMERA,'measured_transform_to_scene':np.linalg.inv(forward).tolist()}
        rows=t.observe(moved,'b',1000,camera)
        for r in rows:
            self.assertEqual(r['identity_status'],'confirmed',r['tracking_evidence'])
            self.assertEqual(r['motion_status'],'stationary')
            board=next(b for b in t.boards if b.board_id==r['board_id'])
            x,y,w,h=board.roi
            expected=forward @ np.array([x,y,1.])
            self.assertLessEqual(np.linalg.norm(np.array(r['measured_screen_roi'][:2])-expected[:2]),2)
            self.assertGreater(r['measured_screen_roi'][2],w)

    def test_clean_automatic_motion_coverage_and_position_error(self):
        t=tracker();correct=0;total=0;errors=[]
        # Future ground truth never enters observe; it renders the independent
        # input and evaluates outputs after tracking.
        for i,(left,right) in enumerate([(0,0),(10,0),(20,10),(30,20),(20,10),(0,0),(0,10),(10,0)]):
            truth=positions(left,right)
            rows=t.observe(render_mechanical(truth),f'f{i}',i*1000,CAMERA)
            for r in rows:
                total+=1
                if r['identity_status']=='confirmed':
                    error=float(np.linalg.norm(np.array(r['measured_screen_roi'][:2])-truth[r['board_id']][:2]))
                    errors.append(error)
                    if error<=4: correct+=1
            json.dumps(rows,allow_nan=False)
        coverage=correct/total
        self.assertGreaterEqual(coverage,.95,{'tracking_coverage':coverage,'samples':total})
        self.assertLessEqual(float(np.median(errors)),2)
        self.assertLessEqual(float(np.percentile(errors,95)),4)
        self.assertEqual(max(errors),0, 'Clean rendered integer translations should be exact')
        self.assertTrue(any(r['measured_screen_roi']!=list(next(b.roi for b in t.boards if b.board_id==r['board_id'])) for r in t.history))

    def test_lost_board_keeps_searching_and_recovers_original_id_with_new_epoch(self):
        t=tracker();initial=t.observe(render_mechanical(positions()),'a',0,CAMERA)
        p=positions(20)
        lost=t.observe(render_mechanical(p,obscured_boards=('left_outer','left_inner')),'b',1000,CAMERA)
        out=record(lost,'left_outer')
        self.assertIsNone(out['measured_screen_roi'])
        self.assertEqual(out['identity_status'],'identity_uncertain')
        self.assertIsNotNone(out['predicted_screen_roi'])
        recovered=t.observe(render_mechanical(p),'c',2000,CAMERA)
        out=record(recovered,'left_outer')
        self.assertEqual(out['identity_status'],'confirmed')
        self.assertEqual(out['measured_screen_roi'],p['left_outer'])
        self.assertEqual(out['comparison_epoch'],1)
        self.assertTrue(out['baseline_reset_required'])
        self.assertTrue(out['search_attempted'])
        self.assertEqual(record(initial,'left_outer')['comparison_epoch'],0)
        self.assertEqual(record(initial,'left_outer')['measured_screen_roi'],positions()['left_outer'])

    def test_inner_linkage_prediction_never_confirms_identity(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        p=positions(30)
        rows=t.observe(render_mechanical(p,obscured_boards=('left_inner',)),'b',1000,CAMERA)
        inner=record(rows,'left_inner')
        self.assertEqual(inner['identity_status'],'identity_uncertain')
        self.assertIsNone(inner['measured_screen_roi'])
        self.assertEqual(inner['predicted_screen_roi'],p['left_inner'])
        self.assertEqual(inner['prediction_source'],'opposed_motion_soft_prior')
        self.assertFalse(inner['mechanical_prior']['identity_evidence'])

    def test_camera_failure_does_not_stop_search_or_return_old_measurement(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        camera={**CAMERA,'reliability':'degraded','measured_transform_to_scene':None}
        p=positions(20)
        rows=t.observe(render_mechanical(p),'b',1000,camera)
        out=record(rows,'left_outer')
        self.assertTrue(out['search_attempted'])
        self.assertEqual(out['identity_status'],'confirmed')
        self.assertEqual(out['visual_association_status'],'confirmed')
        self.assertEqual(out['scene_geometry_status'],'unverified')
        self.assertEqual(out['measured_screen_roi'],p['left_outer'])
        self.assertEqual(out['last_verified_screen_roi'],p['left_outer'])
        self.assertEqual(t._states['left_outer']['_scene_origin'],[40.,50.])
        self.assertEqual(out['motion_status'],'unresolved')
        self.assertIsNone(out['delta_scene_px'])
        self.assertFalse(out['comparison_eligibility']['eligible'])
        restored=t.observe(render_mechanical(p),'c',2000,CAMERA)
        self.assertEqual(record(restored,'left_outer')['identity_status'],'confirmed')
        self.assertTrue(record(restored,'left_outer')['baseline_reset_required'])
        self.assertEqual(record(restored,'left_outer')['motion_status'],'unresolved')

    def test_camera_unknown_freezes_scene_velocity_and_mechanical_anchor_only(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        t.observe(render_mechanical(positions(10,10)),'b',1000,CAMERA)
        old_states=copy.deepcopy(t._states);old_anchors=copy.deepcopy(t._pair_anchors)
        unknown={**CAMERA,'reliability':'unobservable','measured_transform_to_scene':None}
        rows=t.observe(render_mechanical(positions(20,20)),'c',2000,unknown)
        for r in rows:
            old=old_states[r['board_id']];current=t._states[r['board_id']]
            self.assertEqual(r['identity_status'],'confirmed')
            self.assertEqual(r['visual_association_status'],'confirmed')
            self.assertEqual(r['scene_geometry_status'],'unverified')
            self.assertEqual(current['_scene_origin'],old['_scene_origin'])
            self.assertEqual(current['_velocity'],old['_velocity'])
            self.assertEqual(r['last_verified_scene_timestamp_ms'],1000)
            self.assertEqual(r['motion_status'],'unresolved')
            self.assertIsNone(r['delta_scene_px'])
            self.assertFalse(r['mechanical_prior']['measured_pair_available'])
            self.assertNotEqual(r['prediction_source'],'opposed_motion_soft_prior')
            self.assertTrue(all(h['scene_origin'] is None and h['origin_coordinate_space']=='screen_unverified_camera' for h in r['hypotheses']))
            self.assertFalse(r['comparison_eligibility']['eligible'])
        self.assertEqual(t._pair_anchors,old_anchors)
        restored=t.observe(render_mechanical(positions(30,30)),'d',3000,CAMERA)
        for r in restored:
            self.assertTrue(r['geometry_reacquired'])
            self.assertFalse(r['identity_reacquired'])
            self.assertTrue(r['baseline_reset_required'])
            self.assertEqual(r['comparison_epoch'],1)
            self.assertEqual(r['identity_epoch'],0)
            self.assertEqual(r['motion_status'],'unresolved')
            self.assertIsNone(r['delta_scene_px'])
            self.assertEqual(t._states[r['board_id']]['_velocity'],[0.,0.])

    def test_new_geometry_epoch_does_not_bridge_scene_velocity(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        t.observe(render_mechanical(positions(10)),'b',1000,CAMERA)
        camera={**CAMERA,'geometry_epoch':1}
        rows=t.observe(render_mechanical(positions(30)),'c',2000,camera)
        for r in rows:
            self.assertEqual(r['identity_status'],'confirmed')
            self.assertTrue(r['baseline_reset_required'])
            self.assertEqual(r['motion_status'],'unresolved')
            self.assertIsNone(r['delta_scene_px'])
            self.assertEqual(t._states[r['board_id']]['_velocity'],[0.,0.])

    def test_initial_unknown_camera_does_not_create_fake_scene_coordinates(self):
        t=tracker();unknown={**CAMERA,'reliability':'unobservable','measured_transform_to_scene':None}
        initial=t.observe(render_mechanical(positions()),'a',0,unknown)
        for r in initial:
            self.assertEqual(r['identity_status'],'confirmed')
            self.assertIsNone(t._states[r['board_id']]['_scene_origin'])
            self.assertIsNone(r['last_verified_scene_timestamp_ms'])
            self.assertEqual(r['motion_status'],'unresolved')
        rows=t.observe(render_mechanical(positions()),'b',1000,CAMERA)
        self.assertTrue(all(r['baseline_reset_required'] and r['geometry_reacquired'] for r in rows))

    def test_duplicate_texture_cannot_establish_two_physical_identities(self):
        t=tracker();p=positions();frame=render_mechanical(p)
        # Identical textures at two locations make association unobservable.
        outer=p['left_outer'];inner=p['left_inner'];x,y,w,h=outer;ix,iy,_,_=inner
        frame[iy:iy+h,ix:ix+w]=frame[y:y+h,x:x+w]
        t.observe(frame,'a',0,CAMERA)
        rows=t.observe(frame.copy(),'b',1000,CAMERA)
        left=[r for r in rows if r['assembly_id']=='left_assembly']
        self.assertTrue(all(r['identity_status']=='identity_uncertain' for r in left))
        self.assertEqual({r['board_id'] for r in left},{'left_outer','left_inner'})

    def test_fully_hidden_inner_retains_history_but_no_measurement(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        rows=t.observe(render_mechanical(positions(95)),'b',1000,CAMERA)
        inner=record(rows,'left_inner')
        self.assertEqual(inner['identity_status'],'identity_uncertain')
        self.assertIsNone(inner['measured_screen_roi'])
        self.assertEqual(inner['last_verified_screen_roi'],positions()['left_inner'])
        self.assertFalse(inner['comparison_eligibility']['eligible'])

    def test_moving_outer_can_preserve_small_new_symbol_without_content_claim(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        rows=t.observe(render_mechanical(positions(30),minus_boards=('left_outer',)),'b',1000,CAMERA)
        out=record(rows,'left_outer')
        self.assertEqual(out['identity_status'],'confirmed')
        self.assertEqual(out['motion_status'],'moving')
        self.assertFalse(out['comparison_eligibility']['eligible'])

    def test_initial_ids_and_output_order_remain_stable(self):
        t=tracker();rows=t.observe(render_mechanical(positions()),'a',0,CAMERA)
        self.assertEqual([r['board_id'] for r in rows],[b.board_id for b in t.boards])
        self.assertTrue(all(r['identity_status']=='confirmed' for r in rows))
        self.assertTrue(all(r['baseline_reset_required'] for r in rows))
        with self.assertRaises(ValueError):t.observe(render_mechanical(positions()),'a',1000,CAMERA)

    def test_future_ground_truth_is_rejected_by_automatic_interface(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        with self.assertRaises(ValueError):
            t.observe(render_mechanical(positions(20)),'b',1000,CAMERA,manual_positions=positions(20))

    def test_oracle_manual_checkpoint_is_labelled_separately(self):
        t=tracker();t.observe(render_mechanical(positions()),'a',0,CAMERA)
        rows=t.observe(render_mechanical(positions(20)),'b',1000,CAMERA,manual_positions=positions(20),oracle=True)
        self.assertTrue(all(r['tracking_source']=='oracle_manual_checkpoint' for r in rows))
        self.assertTrue(all(r['identity_evidence']['source']=='oracle_manual' for r in rows))
        self.assertTrue(all(r['visual_association_status']=='oracle_manual_not_automatic' for r in rows))

    def test_unknown_blank_board_is_not_automatically_confirmed(self):
        t=tracker();frame=np.full((480,800,3),50,np.uint8)
        t.observe(frame,'a',0,CAMERA)
        rows=t.observe(frame,'b',1000,CAMERA)
        self.assertTrue(all(r['identity_status']=='identity_uncertain' for r in rows))
        self.assertTrue(all(r['measured_screen_roi'] is None for r in rows))
        self.assertTrue(all(r['search_attempted'] for r in rows))

class FixedStructureIdentityTests(unittest.TestCase):
    """Automatic board matching with a supplied fixed-camera test fixture."""

    def setUp(self):
        self.boards = [
            BoardSpec(f'{side}_{layer}', f'{side} {layer}',
                      (x, y, 300, 180), f'{side}_assembly', layer)
            for side, x in [('left', 40), ('right', 440)]
            for layer, y in [('outer', 50), ('inner', 600)]
        ]
        assemblies = [
            AssemblySpec(f'{side}_assembly', (x - 20, 20, 340, 900))
            for side, x in [('left', 40), ('right', 440)]
        ]
        self.tracker = DynamicBoardTracker(self.boards, assemblies)
        random = np.random.default_rng(194)
        self.fixed_borders = {
            board.board_id: random.integers(0, 255, (180, 300, 3), np.uint8)
            for board in self.boards
        }

    def render(self, shift, textured):
        image = np.full((950, 800, 3), (90, 100, 110), np.uint8)
        truth = {}
        for board in self.boards:
            x, y, width, height = board.roi
            if board.assembly_id == 'left_assembly':
                y += shift if board.layer == 'outer' else -shift
            truth[board.board_id] = [x, y, width, height]
            image[y:y + height, x:x + width] = (
                _board_surface(board.board_id) if textured else (25, 70, 40)
            )
        # These textures belong to the screen, not the moving physical panels.
        for board in self.boards:
            x, y, width, height = board.roi
            border = np.zeros((height, width), bool)
            border[:12] = border[-12:] = True
            border[:, :12] = border[:, -12:] = True
            image[y:y + height, x:x + width][border] = self.fixed_borders[board.board_id][border]
        return image, truth

    def test_fixed_border_alone_cannot_confirm_wrong_stationary_position(self):
        initial, _ = self.render(0, textured=False)
        moving, truth = self.render(40, textured=False)
        self.assertFalse(np.array_equal(initial, moving))
        self.tracker.observe(initial, 'initial', 0, CAMERA)
        rows = self.tracker.observe(moving, 'moving', 1000, CAMERA)
        for row in rows:
            self.assertEqual(row['identity_status'], 'identity_uncertain')
            self.assertIsNone(row['measured_screen_roi'])
            self.assertEqual(row['tracking_source'], 'insufficient_interior_identity_support')
            self.assertTrue(row['search_attempted'])
            self.assertEqual(row['last_verified_frame_id'], 'initial')
        self.assertNotEqual(truth['left_outer'], list(self.boards[0].roi))

    def test_interior_texture_tracks_motion_despite_fixed_border(self):
        for index, shift in enumerate((0, 40, 0)):
            image, truth = self.render(shift, textured=True)
            rows = self.tracker.observe(image, f'frame-{index}', index * 1000, CAMERA)
            for row in rows:
                self.assertEqual(row['identity_status'], 'confirmed', row['tracking_evidence'])
                self.assertEqual(row['measured_screen_roi'], truth[row['board_id']])
                if index:
                    self.assertGreaterEqual(row['hypotheses'][0]['unique_interior_support'], 4)

    def test_interior_loss_pauses_then_reacquires_without_replacing_history(self):
        initial, _ = self.render(0, textured=True)
        self.tracker.observe(initial, 'initial', 0, CAMERA)
        history = copy.deepcopy(self.tracker.history)
        obscured, _ = self.render(40, textured=False)
        lost = self.tracker.observe(obscured, 'lost', 1000, CAMERA)
        self.assertTrue(all(row['identity_status'] == 'identity_uncertain' for row in lost))
        self.assertTrue(all(row['measured_screen_roi'] is None for row in lost))
        restored, truth = self.render(40, textured=True)
        recovered = self.tracker.observe(restored, 'recovered', 2000, CAMERA)
        for row in recovered:
            self.assertEqual(row['identity_status'], 'confirmed', row['tracking_evidence'])
            self.assertEqual(row['measured_screen_roi'], truth[row['board_id']])
            self.assertTrue(row['identity_reacquired'])
            self.assertTrue(row['baseline_reset_required'])
            self.assertEqual(row['motion_status'], 'unresolved')
            self.assertFalse(row['comparison_eligibility']['eligible'])
        self.assertEqual(self.tracker.history[:len(history)], history)
        self.assertEqual(len(self.tracker.history), 12)


if __name__=='__main__':unittest.main()
