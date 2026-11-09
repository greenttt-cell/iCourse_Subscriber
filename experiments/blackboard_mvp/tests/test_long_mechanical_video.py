"""197 real toolchain samples preserve source/crop/mask identity through occlusion.

Everything is tiny, artificial, and local. Position checkpoints are explicitly
synthetic geometry truth, not proof of automatic physical-board identification.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blackboard_mvp.runner import run_video


WIDTH, HEIGHT, FRAME_COUNT = 160, 128, 197


def _boards():
    return [
        {"board_id": f"{side}_{layer}", "label": f"{side} {layer}",
         "assembly_id": f"{side}_assembly", "layer": layer,
         "roi": [x, y, 56, 40]}
        for side, x in (("left", 10), ("right", 94))
        for layer, y in (("outer", 12), ("inner", 72))
    ]


def _positions(index):
    if 50 <= index < 80 or 110 <= index < 140:
        shift = 20
    elif 80 <= index < 110:
        shift = 30
    else:
        shift = 0
    result = {}
    for board in _boards():
        roi = list(board["roi"])
        if board["assembly_id"] == "left_assembly":
            roi[1] += shift if board["layer"] == "outer" else -shift
        result[board["board_id"]] = roi
    return result


def _render(index):
    frame = np.full((HEIGHT, WIDTH, 3), (170, 170, 170), np.uint8)
    for x in (2, 86):
        frame[:, x:x+72] = (20, 25, 20)
    # Static landmarks and a changing 4x4 identity marker are outside assemblies.
    for y in range(15, HEIGHT, 20):
        cv2.circle(frame, (80, y), 2, (35, 35, 35), -1)
    frame[4:8, 78:82] = (index, index ^ 0x55, index ^ 0xAA)
    positions = _positions(index)
    for layer in ("inner", "outer"):
        for number, board in enumerate(_boards()):
            if board["layer"] != layer:
                continue
            x, y, w, h = positions[board["board_id"]]
            surface = np.full((h, w, 3), (32 + number, 70 + number, 38), np.uint8)
            cv2.line(surface, (6, 12), (37, 12), (230, 230, 225), 1)
            cv2.line(surface, (12 + number, 7), (12 + number, 30), (230, 230, 225), 1)
            cv2.circle(surface, (40, 25), 3 + number, (230, 230, 225), 1)
            if 80 <= index < 110 and board["board_id"] == "left_inner":
                # An unobservable temporary change, fully behind the outer panel.
                surface[5:20, 20:40] = (245, 245, 245)
            frame[y:y+h, x:x+w] = surface
    return frame


class LongMechanicalVideoTests(unittest.TestCase):
    def test_wrong_decoded_geometry_stops_before_roi_outputs_and_keeps_original(self):
        with tempfile.TemporaryDirectory(prefix="icourse-mismatched-geometry-") as folder:
            root = Path(folder)
            video = root / "one_frame.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 1,
                                     (WIDTH, HEIGHT))
            self.assertTrue(writer.isOpened())
            try:
                writer.write(_render(0))
            finally:
                writer.release()
            config = root / "boards.json"
            config.write_text(json.dumps({
                "sample_interval_ms": 1000,
                "assemblies": [
                    {"assembly_id": "left_assembly", "viewport": [2, 0, 72, HEIGHT]},
                    {"assembly_id": "right_assembly", "viewport": [86, 0, 72, HEIGHT]},
                ],
                "boards": _boards(),
            }), encoding="utf-8")
            output = root / "review"
            # Simulate a decoder/rotation discrepancy, without changing any ROI
            # or the real source frame retained by the FFmpeg extraction step.
            rotated_shape = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
            with patch("blackboard_mvp.runner.cv2.imread", return_value=rotated_shape):
                with self.assertRaisesRegex(RuntimeError, "geometry"):
                    run_video(video, config, output)
            retained = output / "frames" / "f0000.png"
            self.assertTrue(retained.is_file())
            self.assertTrue(np.array_equal(cv2.imread(str(retained)), _render(0)))
            self.assertFalse(list((output / "crops").rglob("*.png")))
            self.assertFalse(list((output / "masks").rglob("*.png")))
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "review.html").exists())

    def test_all_197_samples_keep_pts_pixels_crops_and_masks_through_occlusion(self):
        with tempfile.TemporaryDirectory(prefix="icourse-long-mechanical-") as folder:
            root = Path(folder)
            video = root / "197 合成板 % frames.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 1,
                                     (WIDTH, HEIGHT))
            self.assertTrue(writer.isOpened(), "A trusted lossless local FFV1 encoder is required")
            try:
                for index in range(FRAME_COUNT):
                    writer.write(_render(index))
            finally:
                writer.release()
            configuration = {
                "sample_interval_ms": 1000,
                "geometry_source": "synthetic_ground_truth_manual_geometry",
                "assemblies": [
                    {"assembly_id": "left_assembly", "viewport": [2, 0, 72, HEIGHT]},
                    {"assembly_id": "right_assembly", "viewport": [86, 0, 72, HEIGHT]},
                ],
                "boards": _boards(),
                "tracking_mode": "oracle_geometry",
                "position_keyframes": [
                    {"timestamp_ms": index * 1000, "positions": _positions(index)}
                    for index in range(FRAME_COUNT)
                ],
            }
            config = root / "boards.json"
            config.write_text(json.dumps(configuration), encoding="utf-8")
            output = root / "review"
            manifest = json.loads(run_video(video, config, output).read_text())
            frames = manifest["frames"]
            self.assertEqual(len(frames), FRAME_COUNT)
            self.assertEqual(manifest["source"]["decoded_frame_count"], FRAME_COUNT)
            self.assertEqual(len(list((output / "frames").glob("*.png"))), FRAME_COUNT)
            self.assertEqual(len(list((output / "crops").glob("*/*.png"))), 4 * FRAME_COUNT)
            self.assertEqual(len(list((output / "masks").glob("*/*.png"))), 4 * FRAME_COUNT)
            self.assertEqual(len(manifest["position_history"]), 4 * FRAME_COUNT)
            self.assertEqual(manifest["frame_extraction"]["selector"], "balanced_select_batches")
            self.assertEqual(manifest["frame_extraction"]["selected_frame_count"], FRAME_COUNT)
            self.assertEqual(manifest["frame_extraction"]["batch_count"], 1)
            for index, entry in enumerate(frames):
                with self.subTest(index=index):
                    self.assertEqual(entry["source_index"], index)
                    self.assertEqual(entry["timestamp_ms"], index * 1000)
                    self.assertEqual(entry["time_status"], "verified_pts")
                    exact_ms = entry["native_pts"] * Fraction(entry["native_time_base"]) * 1000
                    self.assertEqual(exact_ms, entry["pts_time_ms"])
                    self.assertEqual(entry["pts_time_ms"], entry["timestamp_ms"])
                    image = cv2.imread(str(output / entry["image_path"]), cv2.IMREAD_COLOR)
                    self.assertTrue(np.array_equal(image, _render(index)), "Source frame content/order changed")
                    self.assertEqual(image[5, 79].tolist(), [index, index ^ 0x55, index ^ 0xAA])
                    for board in entry["boards"]:
                        x, y, w, h = board["crop_roi"]
                        crop = cv2.imread(str(output / board["crop_path"]), cv2.IMREAD_COLOR)
                        mask = cv2.imread(str(output / board["mask_path"]), cv2.IMREAD_GRAYSCALE)
                        self.assertTrue(np.array_equal(crop, image[y:y+h, x:x+w]))
                        expected_mask = np.zeros((h, w), np.uint8)
                        for sx, sy, sw, sh in board["visible_regions"]:
                            expected_mask[sy-y:sy-y+sh, sx-x:sx-x+sw] = 255
                        self.assertTrue(np.array_equal(mask, expected_mask))
                        self.assertEqual(board["screen_roi"], _positions(index)[board["board_id"]])
            self.assertFalse(any(event["kind"] == "board_content_change_event"
                                 for event in manifest["events"]),
                             "Mechanical motion or hidden changes became apparent erasure/writing")
            initial_inner = next(b for b in frames[0]["boards"] if b["board_id"] == "left_inner")
            hidden_inner = next(b for b in frames[90]["boards"] if b["board_id"] == "left_inner")
            restored_inner = next(b for b in frames[-1]["boards"] if b["board_id"] == "left_inner")
            self.assertEqual(hidden_inner["visibility"], "occluded")
            self.assertEqual(hidden_inner["comparison_status"], "paused_occluded")
            self.assertEqual(hidden_inner["visible_regions"], [])
            self.assertEqual(restored_inner["visibility"], "visible")
            self.assertEqual(restored_inner["board_id"], initial_inner["board_id"])
            self.assertEqual(restored_inner["version_id"], initial_inner["version_id"])
            self.assertEqual(len(manifest["versions"]), 4)
            self.assertEqual([h["timestamp_ms"] for h in manifest["position_history"]
                              if h["board_id"] == "left_inner"],
                             list(range(0, FRAME_COUNT * 1000, 1000)))


if __name__ == "__main__":
    unittest.main()
