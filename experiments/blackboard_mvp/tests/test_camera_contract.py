"""Automatic camera geometry and state contracts; no per-frame geometry truth input."""
import json
import unittest

import cv2
import numpy as np

from blackboard_mvp.camera import CameraMonitor
from blackboard_mvp.models import (BoardGeometry, BoardIdentity, ComparisonEligibility,
                                   FrameRef, TransformRef, comparison_gate)


def textured(seed=32):
    rng=np.random.default_rng(seed)
    gray=rng.integers(35,175,(360,640),dtype=np.uint8)
    gray=cv2.GaussianBlur(gray,(3,3),.5)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)


class StateContractTests(unittest.TestCase):
    def test_prediction_cannot_populate_measurement(self):
        with self.assertRaises(ValueError):
            BoardGeometry('left_inner','f1','predicted_only',[20,30,100,80],[20,30,100,80])
        state=BoardGeometry('left_inner','f1','predicted_only',None,[20,30,100,80]).to_dict()
        self.assertIsNone(state['measured_screen_roi'])
        self.assertIsNone(state['measured_screen_polygon'])
        self.assertEqual(state['predicted_screen_roi'],[20,30,100,80])

    def test_measured_state_requires_current_measurement(self):
        with self.assertRaises(ValueError):
            BoardGeometry('board','f1','observed')

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            BoardGeometry('board','f1','observed',[0,float('nan'),20,20])

    def test_prior_cannot_confirm_identity(self):
        with self.assertRaises(ValueError):
            BoardIdentity('inner','confirmed',confirmation_basis='opposed_motion_prior')
        self.assertEqual(BoardIdentity('inner','uncertain').to_dict()['board_id'],'inner')

    def test_manual_and_visual_are_explicit_identity_sources(self):
        for source in ['manual_initial','manual_checkpoint','direct_visual']:
            self.assertEqual(BoardIdentity('board','confirmed',confirmation_basis=source).association_status,'confirmed')

    def test_gate_blocks_unknown_identity_and_disabled_content(self):
        result=comparison_gate(camera_reliability='verified',identity_status='uncertain',measurement_status='observed',
                               baseline_ready=True,baseline_id='b0',visibility_status='visible')
        self.assertFalse(result['eligible'])
        self.assertIn('physical_identity_unconfirmed',result['reason_codes'])
        self.assertIn('content_comparison_disabled_s0_s2',result['reason_codes'])

    def test_gate_requires_new_baseline(self):
        result=comparison_gate(camera_reliability='verified',identity_status='confirmed',measurement_status='observed',
                               content_enabled=True,visibility_status='visible',comparison_epoch=2)
        self.assertFalse(result['eligible'])
        self.assertIn('comparison_baseline_not_ready',result['reason_codes'])

    def test_native_clock_preserved_independently_of_epoch(self):
        ref=FrameRef('f0135',135000,81000,'1/600',5000).to_dict()
        self.assertEqual(ref['native_pts'],81000)
        self.assertEqual(ref['native_time_base'],'1/600')
        with self.assertRaises(ValueError):
            FrameRef('x',0,100,None)

    def test_transform_direction_epoch_and_finite_matrix(self):
        t=TransformRef('c0','screen','scene',[[1,0,-4],[0,1,-7],[0,0,1]],3).to_dict()
        self.assertEqual(t['geometry_epoch'],3)
        self.assertEqual(t['from_space'],'screen')
        with self.assertRaises(ValueError):
            TransformRef('bad','screen','scene',[[0,0,0],[0,0,0],[0,0,0]])
        with self.assertRaises(ValueError):
            TransformRef('bad','screen','scene',[[1,0,0],[0,float('inf'),0],[0,0,1]])


class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.base=textured()
        self.monitor=CameraMonitor(excluded_regions=[[100,50,440,260]])

    def observe(self,frame,index,**kwargs):
        return self.monitor.observe(frame,f'f{index:04}',index*1000,**kwargs)

    def test_initial_distributed_anchors_and_json(self):
        state=self.observe(self.base,0)
        self.assertEqual(state['reliability'],'verified')
        self.assertGreaterEqual(state['evidence']['support_count'],3)
        json.dumps(state,allow_nan=False)

    def test_explicit_static_anchors_inside_component_search_are_not_erased(self):
        monitor=CameraMonitor(anchor_regions=[[10,10,150,100],[470,230,150,100],[470,10,150,100]],
                              excluded_regions=[[0,0,640,360]])
        state=monitor.observe(self.base,'f0',0)
        self.assertEqual(state['reliability'],'verified')
        self.assertEqual(state['evidence']['anchor_source'],'explicit_static_regions')
        self.assertGreater(np.count_nonzero(monitor._mask),1000)
        next_state=monitor.observe(self.base,'f1',1000)
        self.assertEqual(next_state['reliability'],'verified')

    def test_lk_centres_have_complete_window_footprints(self):
        state=self.observe(self.base,0)
        self.assertTrue(state['evidence']['anchor_support_window_verified'])
        for x,y in self.monitor._points.reshape(-1,2):
            self.assertGreaterEqual(self.monitor._anchor_distance[int(round(y)),int(round(x))],14)
        next_state=self.observe(self.base,1)
        self.assertEqual(next_state['evidence']['lk_max_level'],0)
        self.assertTrue(next_state['evidence']['anchor_support_window_verified'])

    def test_orb_footprints_reject_boundary_and_large_scale_patches(self):
        mask=np.full((100,100),255,np.uint8)
        distance=CameraMonitor._distance(mask)
        kp=[cv2.KeyPoint(2,50,15),cv2.KeyPoint(50,50,15),cv2.KeyPoint(50,50,100)]
        ds=np.zeros((3,32),np.uint8)
        accepted,descriptors,rejected=CameraMonitor._descriptor_footprints(kp,ds,distance)
        self.assertEqual(len(accepted),1)
        self.assertEqual(descriptors.shape,(1,32))
        self.assertEqual(rejected,2)

    def test_moving_texture_outside_narrow_static_anchors_is_not_camera_evidence(self):
        monitor=CameraMonitor(excluded_regions=[[25,0,590,360]])
        first=np.full_like(self.base,80)
        first[:,25:615]=self.base[:,25:615]
        state=monitor.observe(first,'f0',0)
        self.assertNotEqual(state['reliability'],'verified')
        second=np.full_like(self.base,80)
        second[:,25:615]=np.roll(self.base[:,25:615],20,axis=0)
        state=monitor.observe(second,'f1',1000)
        self.assertNotEqual(state['reliability'],'verified')
        self.assertIsNone(state['measured_transform_to_scene'])
        self.assertFalse(state['evidence']['anchor_support_window_verified'])

    def test_true_camera_translation_has_measured_inverse(self):
        self.observe(self.base,0)
        shifted=cv2.warpAffine(self.base,np.float32([[1,0,4],[0,1,7]]),(640,360),borderMode=cv2.BORDER_REFLECT)
        state=self.observe(shifted,1)
        self.assertEqual(state['reliability'],'verified')
        self.assertEqual(state['disturbance_type'],'camera_motion_supported')
        matrix=np.array(state['measured_transform_to_scene'])
        np.testing.assert_allclose(matrix[:2,2],[-4,-7],atol=.6)

    def test_large_brightness_change_not_permanent_camera_failure(self):
        self.observe(self.base,0)
        bright=cv2.convertScaleAbs(self.base,alpha=1,beta=40)
        state=self.observe(bright,1)
        self.assertEqual(state['reliability'],'verified')
        self.assertEqual(state['disturbance_type'],'photometric_change')
        self.assertGreater(state['evidence']['anchor_changed_fraction'],.90)
        self.assertIsNotNone(state['measured_transform_to_scene'])

    def test_local_forty_four_percent_appearance_change_not_motion_proof(self):
        self.observe(self.base,0)
        changed=self.base.copy()
        # Roughly 44% of anchor pixels change, while spatially separate fixed
        # anchors still offer actual geometry evidence.
        mask=self.monitor._mask>0
        ys,xs=np.where(mask)
        changed[ys[:int(.44*len(ys))],xs[:int(.44*len(xs))]]=15
        state=self.observe(changed,1)
        self.assertNotEqual(state['disturbance_type'],'camera_motion_supported')
        self.assertTrue(state['search_attempted'])
        for i in range(2,6): state=self.observe(cv2.convertScaleAbs(self.base,alpha=1,beta=i),i)
        self.assertEqual(state['reliability'],'verified')

    def test_one_bad_frame_continues_search_and_recovers_with_geometry(self):
        self.observe(self.base,0)
        bad=self.observe(np.zeros_like(self.base),1)
        self.assertIsNone(bad['measured_transform_to_scene'])
        for i in range(2,4):
            state=self.observe(cv2.convertScaleAbs(self.base,alpha=1,beta=i),i)
            self.assertEqual(state['reliability'],'reinitializing')
            self.assertIsNone(state['measured_transform_to_scene'])
        state=self.observe(cv2.convertScaleAbs(self.base,alpha=1,beta=4),4)
        self.assertEqual(state['reliability'],'verified')
        self.assertTrue(state['recovered'])
        self.assertEqual(state['geometry_epoch'],1)
        self.assertFalse(state['identity_inheritance_allowed'])
        self.assertEqual(len(self.monitor.history),5)
        self.assertTrue(all(s['search_attempted'] for s in self.monitor.history))

    def test_duplicate_recovery_image_is_not_three_observations(self):
        self.observe(self.base,0)
        self.observe(np.zeros_like(self.base),1)
        for i in range(2,8):
            state=self.observe(self.base,i)
        self.assertEqual(state['reliability'],'reinitializing')
        self.assertEqual(state['recovery_count'],1)
        self.assertTrue(state['search_attempted'])
        self.assertIn('duplicate_anchor_image_not_new_recovery_evidence',state['reason_codes'])

    def test_no_anchors_does_not_assume_camera_static(self):
        monitor=CameraMonitor(anchor_regions=[])
        for i in range(4):
            state=monitor.observe(self.base,f'f{i}',i*1000)
            self.assertEqual(state['reliability'],'unobservable')
            self.assertIsNone(state['measured_transform_to_scene'])

    def test_insufficient_texture_does_not_claim_recovery(self):
        monitor=CameraMonitor()
        for i in range(4):
            state=monitor.observe(np.full_like(self.base,80),f'f{i}',i*1000)
            self.assertNotEqual(state['reliability'],'verified')
            self.assertIsNone(state['measured_transform_to_scene'])

    def test_foreign_stable_scene_cannot_inherit_old_scene(self):
        self.observe(self.base,0)
        for i in range(1,5):
            state=self.observe(textured(999),i)
            self.assertNotEqual(state['reliability'],'verified')
            self.assertIsNone(state['measured_transform_to_scene'])
            self.assertFalse(state['identity_inheritance_allowed'])

    def test_resolution_change_does_not_destroy_original_reference(self):
        self.observe(self.base,0)
        state=self.observe(self.base[:300,:500],1)
        self.assertEqual(state['reliability'],'manual_review_required')
        for i in range(2,5): state=self.observe(cv2.convertScaleAbs(self.base,alpha=1,beta=i),i)
        self.assertEqual(state['reliability'],'verified')
        self.assertEqual(state['reference_frame_id'],'f0000')

    def test_force_reconfirmation_does_not_permanently_latch(self):
        self.observe(self.base,0)
        state=self.observe(self.base,1,force_reconfirm=True)
        self.assertEqual(state['reliability'],'degraded')
        for i in range(2,5): state=self.observe(cv2.convertScaleAbs(self.base,alpha=1,beta=i),i)
        self.assertEqual(state['reliability'],'verified')

    def test_repeated_pts_and_frame_id_rejected(self):
        self.observe(self.base,0)
        with self.assertRaises(ValueError): self.observe(self.base,0)
        with self.assertRaises(ValueError): self.monitor.observe(self.base,'different',0)


if __name__=='__main__':
    unittest.main()
