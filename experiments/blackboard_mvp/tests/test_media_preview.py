"""Real local FFmpeg/FFprobe plus portable offline HTML safety checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blackboard_mvp.media import probe_media, source_frames, write_probe_preview
from blackboard_mvp.preview import write_preview
from blackboard_mvp.synthetic import write_demo


class MediaPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='icourse-media-preview-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_probe_and_first_frame_manual_roi_preview(self):
        fixture = write_demo(self.root/'input')
        video = Path(fixture['video'])
        probe = probe_media(video)
        self.assertEqual((probe['width'], probe['height']), (640,400))
        self.assertEqual(probe['metadata_source'],'ffprobe')
        self.assertEqual(probe['duration_ms'],8000)
        self.assertEqual(len(probe['sha256']),64)
        self.assertTrue(probe['time_base'])
        page = write_probe_preview(video,self.root/'probe')
        self.assertTrue(page.is_file())
        self.assertTrue((page.parent/'frames/f0000.png').is_file())
        self.assertEqual(len(list((page.parent/'frames').glob('*.png'))),1)
        text = page.read_text()
        self.assertIn('手动',text)
        self.assertIn('connect-src',text)
        self.assertNotIn('<script src=',text)

    def test_missing_pts_is_rejected_instead_of_using_fps(self):
        payload=json.dumps({'frames':[{'width':640,'height':400}]}).encode()
        with patch('blackboard_mvp.media.run_tool',return_value=payload), \
                patch('blackboard_mvp.media.tool_path',return_value='/synthetic/ffprobe'):
            with self.assertRaisesRegex(ValueError,'PTS'):
                source_frames(Path('/synthetic.avi'),{'time_base':'1/4','video_stream_index':0},1000)

    def test_best_effort_pts_is_explicitly_distinguished(self):
        payload=json.dumps({'frames':[{'best_effort_timestamp':40},{'best_effort_timestamp':44}]}).encode()
        probe={'time_base':'1/4','video_stream_index':0,'width':640,'height':400}
        with patch('blackboard_mvp.media.run_tool',return_value=payload), \
                patch('blackboard_mvp.media.tool_path',return_value='/synthetic/ffprobe'):
            frames=source_frames(Path('/synthetic.avi'),probe,1000)
        self.assertEqual([f['timestamp_ms'] for f in frames],[0,1000])
        self.assertEqual([f['pts_time_ms'] for f in frames],[10000,11000])
        self.assertEqual(frames[0]['time_status'],'best_effort_pts')
        self.assertEqual(probe['timeline_origin_ms'],10000)

    def test_html_embeds_json_as_data_and_blocks_remote_images(self):
        page = write_preview({'config':{'boards':[{'board_id':'b','label':'</script><img src=https://example.invalid>','roi':[0,0,20,20]}]},
                              'source':{},'frames':[],'events':[],'versions':[],'writing_runs':[]},self.root/'review')
        html=page.read_text()
        self.assertNotIn('</script><img',html)
        self.assertIn('\\u003c/script\\u003e',html)
        self.assertIn("connect-src 'none'",html)
        self.assertNotIn('<script src=',html)
        self.assertIn('localPath',html)
        self.assertIn('textContent',html)


if __name__=='__main__':
    unittest.main()
