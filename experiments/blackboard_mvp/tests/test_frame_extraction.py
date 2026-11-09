"""Long native-index selections and literal paths, using local synthetic media.

No classroom files, credentials, network services or model APIs are used.  The
lossless fixture encodes each source index in its pixels, so these checks catch
omissions, duplicates and reordering rather than only counting PNG files.
"""
from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blackboard_mvp.media import extract_frames, probe_media, source_frames


WIDTH, HEIGHT, FRAME_COUNT, FPS = 64, 48, 1025, 20


def counter_frame(index):
    frame = np.full((HEIGHT, WIDTH, 3),
                    (index & 255, (index >> 8) & 255, (255 - index) & 255),
                    dtype=np.uint8)
    # Asymmetric marks catch flips/crops as well as wrong source indices.
    frame[2:7, 3:10] = (17, 231, 82)
    frame[-6:-2, -12:-3] = (198, 9, 141)
    return frame


def write_counter_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"),
                             FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("The local regression fixture requires lossless FFV1")
    try:
        for index in range(FRAME_COUNT):
            writer.write(counter_frame(index))
    finally:
        writer.release()


class FrameExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="icourse-frame-extraction-")
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "synthetic-counters.avi"
        write_counter_video(cls.video)
        # Both calls invoke the installed FFprobe, not a mock or FPS estimate.
        cls.probe = probe_media(cls.video)
        cls.native = source_frames(cls.video, cls.probe, 50)
        if len(cls.native) != FRAME_COUNT:
            raise AssertionError("Synthetic fixture did not retain every native frame")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fresh_folder(self, label):
        return self.root / (self._testMethodName + "-" + label)

    def assert_selected_pixels_and_times(self, selected, folder):
        self.assertEqual({p.name for p in folder.iterdir()},
                         {f"f{i:04d}.png" for i in range(len(selected))})
        for output_index, sample in enumerate(selected):
            image = cv2.imread(str(folder / f"f{output_index:04d}.png"),
                               cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertTrue(np.array_equal(image, counter_frame(sample["source_index"])),
                            f"Wrong pixels at output {output_index}")
            source = self.native[sample["source_index"]]
            self.assertEqual(sample, source)
            self.assertEqual(round(sample["native_pts"] *
                                   Fraction(sample["native_time_base"]) * 1000),
                             sample["pts_time_ms"])
            self.assertEqual(sample["time_status"], "verified_pts")

    def test_197_sparse_indices_real_ffmpeg_keep_every_pixel_and_native_time(self):
        selected = copy.deepcopy(self.native[:393:2])
        self.assertEqual(len(selected), 197)
        before = copy.deepcopy(selected)
        folder = self.fresh_folder("197")
        receipt = extract_frames(self.video, self.probe, selected, folder)
        self.assertEqual(selected, before, "Extraction must not rewrite PTS metadata")
        self.assertEqual(receipt["selected_frame_count"], 197)
        self.assertEqual(receipt["batch_count"], 1)
        self.assertEqual(receipt["selector"], "balanced_select_batches")
        self.assert_selected_pixels_and_times(selected, folder)

    def test_512_boundary_real_ffmpeg_has_no_missing_or_reordered_frames(self):
        selected = copy.deepcopy(self.native[:512])
        folder = self.fresh_folder("512")
        receipt = extract_frames(self.video, self.probe, selected, folder)
        self.assertEqual(receipt["batch_count"], 1)
        self.assertEqual(receipt["batch_size_limit"], 512)
        self.assert_selected_pixels_and_times(selected, folder)

    def test_513_real_ffmpeg_batches_keep_global_output_numbering(self):
        selected = copy.deepcopy(self.native[::2])
        self.assertEqual(len(selected), 513)
        folder = self.fresh_folder("513")
        receipt = extract_frames(self.video, self.probe, selected, folder)
        self.assertEqual(receipt["batch_count"], 2)
        self.assertEqual(receipt["selected_frame_count"], 513)
        self.assert_selected_pixels_and_times(selected, folder)

    def test_one_final_source_frame_keeps_its_native_timestamp(self):
        selected = [copy.deepcopy(self.native[-1])]
        folder = self.fresh_folder("last")
        receipt = extract_frames(self.video, self.probe, selected, folder)
        self.assertEqual(receipt["selected_frame_count"], 1)
        self.assert_selected_pixels_and_times(selected, folder)

    def test_special_literal_input_and_output_paths_real_ffmpeg(self):
        special = self.fresh_folder("空 格, '引号'; (括号) [方括号] 100% $美元 $(literal)")
        special.mkdir()
        video = special / "录 像, '引号'; (内容) 100% $变量 $(not-a-command).avi"
        video.write_bytes(self.video.read_bytes())
        probe = probe_media(video)
        selected = source_frames(video, probe, 50)[:3]
        folder = special / "输 出, '引号'; (裁剪) 100% $值 $(still-literal)"
        extract_frames(video, probe, selected, folder)
        self.assert_selected_pixels_and_times(selected, folder)

    def test_invalid_selections_fail_before_tools_or_output_creation(self):
        invalid = [
            [], None, {}, "not-a-list", [None], [{}],
            [{"source_index": -1}], [{"source_index": True}],
            [{"source_index": False}], [{"source_index": 1.0}],
            [{"source_index": float("nan")}],
            [{"source_index": "0),movie=untrusted"}],
            [{"source_index": 2 ** 53}],
            [{"source_index": 1}, {"source_index": 1}],
            [{"source_index": 2}, {"source_index": 1}],
            [{"source_index": i} for i in range(10001)],
        ]
        for index, selected in enumerate(invalid):
            with self.subTest(case=index):
                folder = self.fresh_folder(f"invalid-{index}")
                with patch("blackboard_mvp.media.run_tool") as run:
                    with self.assertRaises(ValueError):
                        extract_frames(self.video, self.probe, selected, folder)
                    run.assert_not_called()
                self.assertFalse(folder.exists())

    def test_invalid_stream_indices_fail_before_tools_or_output_creation(self):
        for index, stream in enumerate((-1, True, "0", 0.0, None)):
            with self.subTest(stream=stream):
                probe = {**self.probe, "video_stream_index": stream}
                folder = self.fresh_folder(f"stream-{index}")
                with patch("blackboard_mvp.media.run_tool") as run:
                    with self.assertRaises(ValueError):
                        extract_frames(self.video, probe, self.native[:1], folder)
                    run.assert_not_called()
                self.assertFalse(folder.exists())

    def test_invalid_sampling_interval_and_duration_are_rejected_before_ffprobe(self):
        # In particular, interval=0 used to make target-advancement loop forever.
        for field in ("sample_interval_ms", "max_duration_ms"):
            for value in (0, -1, True, False, 1.5, "1000"):
                with self.subTest(field=field, value=value):
                    arguments = {"sample_interval_ms": 50, "max_duration_ms": 600000}
                    arguments[field] = value
                    with patch("blackboard_mvp.media.run_tool") as run:
                        with self.assertRaises(ValueError):
                            source_frames(self.video, self.probe, **arguments)
                        run.assert_not_called()

    def test_existing_output_is_preserved_without_tool_call(self):
        folder = self.fresh_folder("existing")
        folder.mkdir()
        marker = folder / "keep.txt"
        marker.write_text("existing evidence", encoding="utf-8")
        with patch("blackboard_mvp.media.run_tool") as run:
            with self.assertRaises(FileExistsError):
                extract_frames(self.video, self.probe, self.native[:1], folder)
            run.assert_not_called()
        self.assertEqual(marker.read_text(), "existing evidence")

    def test_missing_selected_output_is_rejected_instead_of_success(self):
        folder = self.fresh_folder("missing")
        with patch("blackboard_mvp.media.run_tool", return_value=b""):
            with self.assertRaises(RuntimeError):
                extract_frames(self.video, self.probe, self.native[:2], folder)
        self.assertTrue(folder.is_dir(), "Preserve this run's failure evidence")

    def test_tool_failure_preserves_partial_frame_evidence(self):
        folder = self.fresh_folder("partial")

        def fail_with_evidence(*args, **kwargs):
            (folder / "f0000.png").write_bytes(b"partial-evidence")
            raise RuntimeError("Synthetic local tool failure")

        with patch("blackboard_mvp.media.run_tool", side_effect=fail_with_evidence):
            with self.assertRaises(RuntimeError):
                extract_frames(self.video, self.probe, self.native[:2], folder)
        self.assertEqual((folder / "f0000.png").read_bytes(), b"partial-evidence")

    def test_unexpected_output_is_not_silently_ignored(self):
        folder = self.fresh_folder("unexpected")

        def extra_output(*args, **kwargs):
            (folder / "f0000.png").write_bytes(b"frame")
            (folder / "unexpected.png").write_bytes(b"unexplained")
            return b""

        with patch("blackboard_mvp.media.run_tool", side_effect=extra_output):
            with self.assertRaises(RuntimeError):
                extract_frames(self.video, self.probe, self.native[:1], folder)
        self.assertTrue((folder / "unexpected.png").is_file())


if __name__ == "__main__":
    unittest.main()
