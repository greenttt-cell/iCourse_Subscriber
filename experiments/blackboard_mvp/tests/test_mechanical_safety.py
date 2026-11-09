"""Independent regression cases for sparse chalk, evidence and camera safety."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blackboard_mvp.core import BoardSpec
from blackboard_mvp.mechanics import AssemblySpec, SlidingBoardTracker
from blackboard_mvp.mechanical_synthetic import (assembly_config, mechanical_board_config,
                                                positions, render_mechanical)


def tracker():
    boards=[BoardSpec(b['board_id'],b['label'],tuple(b['roi']),b['assembly_id'],b['layer']) for b in mechanical_board_config()]
    return SlidingBoardTracker(boards,[AssemblySpec(a['assembly_id'],tuple(a['viewport'])) for a in assembly_config()])


def observe(t, frame, time, manual=None, force=False):
    fid=f'f{time}'
    return t.observe(frame,fid,time,f'frames/{fid}.png',
                     {b.board_id:f'crops/{b.board_id}/{fid}.png' for b in t.boards},
                     manual_positions=manual,force_reconfirm=force)


class NoFeatures:
    def detectAndCompute(self,*args):
        return [],None


class MechanicalSafetyTests(unittest.TestCase):
    def test_sparse_chalk_translation_without_features_does_not_become_writing(self):
        t=tracker()
        def sparse(p):
            image=np.full((480,800,3),170,np.uint8)
            for b in sorted(mechanical_board_config(),key=lambda b:b['layer']=='outer'):
                x,y,w,h=p[b['board_id']]
                image[y:y+h,x:x+w]=(35,72,40)
                cv2.line(image,(x+30,y+70),(x+150,y+70),(240,240,240),1)
            return image
        observe(t,sparse(positions()),0)
        with patch('blackboard_mvp.mechanics.cv2.ORB_create',return_value=NoFeatures()):
            record=observe(t,sparse(positions(5)),1000)
        self.assertEqual(next(b for b in record['boards'] if b['board_id']=='left_outer')['identity_status'],'identity_uncertain')
        self.assertFalse(any(e['kind']=='board_content_change_event' for e in t.events))
        self.assertEqual(len(t.versions),4)

    def test_camera_failure_latches_even_after_old_image_returns_or_manual_checkpoint(self):
        t=tracker(); frame=render_mechanical(positions())
        observe(t,frame,0)
        before=copy.deepcopy(t.versions)
        shifted=np.roll(frame,12,axis=1)
        observe(t,shifted,1000)
        record=observe(t,frame,2000,positions())
        self.assertTrue(all(b['identity_status']=='identity_uncertain' for b in record['boards']))
        self.assertEqual(t.versions,before)
        self.assertFalse(any(e['kind']=='board_content_change_event' for e in t.events))

    def test_frame_geometry_change_pauses_content_without_reusing_board_identity(self):
        t=tracker(); frame=render_mechanical(positions())
        observe(t,frame,0)
        record=observe(t,frame[:450,:780],1000)
        self.assertTrue(all(b['identity_status']=='identity_uncertain' for b in record['boards']))
        self.assertFalse(any(e['kind']=='board_content_change_event' for e in t.events))

    def test_out_of_frame_panel_has_no_nonexistent_crop_reference(self):
        t=tracker(); p=positions(); p['left_outer'][1]=-200
        record=observe(t,render_mechanical(p),0,p)
        board=next(b for b in record['boards'] if b['board_id']=='left_outer')
        self.assertIsNone(board['crop_roi'])
        self.assertIsNone(board['crop_path'])
        self.assertIsNone(next(h for h in t.position_history if h['board_id']=='left_outer')['crop_path'])
        self.assertFalse(any(v['board_id']=='left_outer' for v in t.versions))

    def test_pixel_reference_history_survives_changes_in_other_visible_regions(self):
        t=tracker(); p0=positions(); f0=render_mechanical(p0)
        observe(t,f0,0,p0)
        # Inner top half becomes hidden while the visible bottom is edited.
        p1=positions(50); f1=render_mechanical(p1)
        observe(t,f1,1000,p1)
        x,y,w,h=p1['left_inner']; f2=f1.copy()
        cv2.line(f2,(x+220,y+165),(x+231,y+165),(250,250,250),1)
        observe(t,f2,2000,p1)
        # A different hidden top region is only observed after reappearance.
        f3=render_mechanical(p0)
        x,y,w,h=p0['left_inner']
        cv2.line(f3,(x+220,y+165),(x+231,y+165),(250,250,250),1)
        cv2.line(f3,(x+220,y+45),(x+231,y+45),(250,250,250),1)
        observe(t,f3,3000,p0)
        observe(t,f3,4000,p0)
        events=[e for e in t.events if e['kind']=='board_content_change_event' and e['board_id']=='left_inner']
        event=events[-1]
        self.assertEqual(event['before_frame_id'],'f0')
        self.assertEqual(event['interval_ms'],[0,4000])
        self.assertEqual(event['comparison_references'][0]['frame_id'],'f0')
        self.assertEqual(event['comparison_reference_semantics'],'per_pixel_last_confirmed_visible_evidence')


if __name__=='__main__':
    unittest.main()
