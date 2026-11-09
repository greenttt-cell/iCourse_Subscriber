"""Replay integrity and native-time contracts; no course data or model calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from blackboard_mvp.replay import replay_manifest
from blackboard_mvp.mechanical_synthetic import render_mechanical, positions, assembly_config, mechanical_board_config


class ReplayEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'original';(self.source/'frames').mkdir(parents=True)
        frames=[]
        rng=np.random.default_rng(349)
        top=rng.integers(0,256,(25,800,3),dtype=np.uint8)
        for i,(timestamp,pts) in enumerate([(0,300),(1017,910),(2500,1800)]):
            image=render_mechanical(positions(0 if i==0 else 35))
            image[:25]=top
            fid=f'f{i:04d}';name=f'frames/{fid}.png'
            self.assertTrue(cv2.imwrite(str(self.source/name),image))
            frames.append({'frame_id':fid,'image_path':name,'timestamp_ms':timestamp,'native_pts':pts,
                           'native_time_base':'1/600','pts_origin':'pts','source_index':i*43,'pts_time_ms':timestamp+500,'time_status':'verified_pts'})
        self.manifest={'source':{'width':800,'height':480,'duration_ms':3000,'time_base':'1/600'},
                       'config':{'boards':mechanical_board_config(),'assemblies':assembly_config()},'frames':frames}
        self.path=self.source/'manifest.json';self.save()
    def save(self):self.path.write_text(json.dumps(self.manifest))
    def test_replay_retains_every_source_byte_and_native_clock_without_overwrite(self):
        before={str(p.relative_to(self.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()}
        output=self.root/'replay';result=json.loads(replay_manifest(self.path,output).read_text())
        self.assertEqual(len(result['frames']),3);self.assertEqual(len(result['position_history']),12)
        for old,new in zip(self.manifest['frames'],result['frames']):
            for key in ['frame_id','timestamp_ms','native_pts','native_time_base','pts_origin','source_index','pts_time_ms']:
                self.assertEqual(old[key],new[key])
            self.assertEqual((self.source/old['image_path']).read_bytes(),(output/new['image_path']).read_bytes())
            for board in new['boards']:
                if board['crop_roi']:
                    x,y,w,h=board['crop_roi'];source=cv2.imread(str(output/new['image_path']))
                    np.testing.assert_array_equal(cv2.imread(str(output/board['crop_path'])),source[y:y+h,x:x+w])
                    # Unknown physical aperture is not a verified visible area.
                    self.assertIsNone(board['visible_fraction'])
                    self.assertFalse(np.any(cv2.imread(str(output/board['mask_path']),0)))
                self.assertFalse(board['comparison_eligibility']['eligible'])
        self.assertFalse(any(e['kind']=='board_content_change_event' for e in result['events']))
        self.assertEqual(before,{str(p.relative_to(self.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()})
        self.assertTrue((output/'review.html').is_file())
        with self.assertRaises(FileExistsError):replay_manifest(self.path,output)
    def test_automatic_replay_rejects_future_ground_truth_before_creating_output(self):
        self.manifest['config']['position_keyframes']=[{'timestamp_ms':1017,'positions':positions(35)}];self.save()
        with self.assertRaisesRegex(ValueError,'oracle|keyframes'):replay_manifest(self.path,self.root/'no-output')
        self.assertFalse((self.root/'no-output').exists())
    def test_explicit_oracle_input_does_not_silently_become_auto(self):
        self.manifest['config']['tracking_mode']='oracle_geometry';self.save()
        with self.assertRaisesRegex(ValueError,'oracle'):replay_manifest(self.path,self.root/'no-output')
    def test_missing_pts_does_not_fabricate_fps_time(self):
        del self.manifest['frames'][1]['native_pts'];self.save()
        with self.assertRaisesRegex(ValueError,'PTS'):replay_manifest(self.path,self.root/'no-output')
    def test_external_or_traversal_frame_reference_rejected(self):
        self.manifest['frames'][0]['image_path']='../outside.png';self.save()
        with self.assertRaises(ValueError):replay_manifest(self.path,self.root/'no-output')
    def test_original_output_ancestor_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError,'independent'):replay_manifest(self.path,self.root)
