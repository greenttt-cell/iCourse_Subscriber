"""Offline semantic tests: no credentials, network, course media, or AI calls."""
from __future__ import annotations

import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from blackboard_mvp.core import BoardSpec, DetectorConfig, MultiBoardTracker
from blackboard_mvp.media import tool_path
from blackboard_mvp.synthetic import board_config, multiboard_samples, write_demo, write_video


def run_video(video_path: Path, config_path: Path, output_dir: Path) -> Path:
    # Lazy import also lets the tracker tests run without loading the CLI tools.
    from blackboard_mvp.runner import run_video as run
    return run(video_path, config_path, output_dir)


def specs(config: list[dict] | None = None) -> list[BoardSpec]:
    return [BoardSpec(board_id=b["board_id"], label=b["label"], roi=tuple(b["roi"]))
            for b in (config if config is not None else board_config())]


def change_events(tracker: MultiBoardTracker) -> list[dict]:
    return [e for e in tracker.events if e["kind"] == "visual_change_candidate"]


class MultiBoardTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = multiboard_samples()
        self.tracker = MultiBoardTracker(specs(), DetectorConfig())

    def observe(self, index: int, tracker: MultiBoardTracker | None = None,
                frame: np.ndarray | None = None, timestamp_ms: int | None = None) -> dict:
        sample = self.samples[index]
        tracker = tracker or self.tracker
        frame_id = f"f{index:04d}" if timestamp_ms is None else f"repeat_at_{timestamp_ms}"
        return tracker.observe(
            frame=sample.frame if frame is None else frame,
            frame_id=frame_id,
            timestamp_ms=sample.timestamp_ms if timestamp_ms is None else timestamp_ms,
            image_path=f"frames/{frame_id}.png",
            crop_paths={b["board_id"]: f"crops/{b['board_id']}/{frame_id}.png" for b in board_config()},
            time_status="verified_pts",
        )

    def test_initial_existing_writing_is_initial_observation(self) -> None:
        self.observe(0)
        self.assertEqual(len(self.tracker.events), 4)
        self.assertEqual({e["board_id"] for e in self.tracker.events}, {b["board_id"] for b in board_config()})
        for event in self.tracker.events:
            self.assertEqual(event["kind"], "initial_observation")
            self.assertEqual(event["interval_ms"], [0, 0])
            self.assertIsNone(event["before_frame_id"])
            self.assertIsNone(event["previous_version_id"])
            self.assertIn("already_present_at_clip_start", event["selected_reasons"])
        self.assertEqual(len(self.tracker.versions), 4)
        self.assertTrue(all(v["content_status"] == "initial_existing" for v in self.tracker.versions))

    def test_four_boards_follow_observed_non_spatial_time_order(self) -> None:
        for index in range(5):
            self.observe(index)
        events = change_events(self.tracker)
        self.assertEqual([e["board_id"] for e in events], ["board_c", "board_a", "board_d", "board_b"])
        self.assertEqual([e["observed_at_ms"] for e in events], [1000, 2000, 3000, 4000])
        self.assertEqual([e["interval_ms"] for e in events], [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000]])
        for event in events:
            self.assertEqual(event["order_status"], "observation_only")
            self.assertTrue(event["change_regions"])
            self.assertTrue(event["selected_reasons"])
            self.assertTrue(event["before_crop_path"].startswith(f"crops/{event['board_id']}/"))
            self.assertTrue(event["after_crop_path"].startswith(f"crops/{event['board_id']}/"))

    def test_return_to_first_changed_board_preserves_small_minus_and_versions(self) -> None:
        for index in range(5):
            self.observe(index)
        old_versions = copy.deepcopy(self.tracker.versions)
        self.observe(5)
        event = change_events(self.tracker)[-1]
        self.assertEqual(event["board_id"], "board_c")
        self.assertEqual(event["observed_at_ms"], 5000)
        self.assertEqual(event["interval_ms"], [4000, 5000])
        self.assertEqual(event["before_frame_id"], "f0004")
        self.assertEqual(event["after_frame_id"], "f0005")
        self.assertIsNotNone(event["previous_version_id"])
        self.assertNotEqual(event["previous_version_id"], event["version_id"])
        self.assertEqual(self.tracker.versions[:len(old_versions)], old_versions)
        board_versions = [v for v in self.tracker.versions if v["board_id"] == "board_c"]
        self.assertEqual(len(board_versions), 3)
        self.assertEqual(event["previous_version_id"], board_versions[-2]["version_id"])
        roi = next(b["roi"] for b in board_config() if b["board_id"] == "board_c")
        for region in event["change_regions"]:
            x, y, width, height = region["bbox"]
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, roi[2])
            self.assertLessEqual(y + height, roi[3])

    def test_unchanged_visible_old_content_is_not_new_event(self) -> None:
        self.observe(0)
        self.observe(0, timestamp_ms=1000)
        self.assertEqual(len(self.tracker.events), 4)
        self.assertEqual(len(self.tracker.versions), 4)

    def test_accumulated_tiny_changes_keep_wider_interval_and_reference_evidence(self) -> None:
        self.observe(0)
        frame = self.samples[0].frame.copy()
        board = next(b for b in board_config() if b["board_id"] == "board_c")
        x, y, _, _ = board["roi"]
        for index in range(1, 4):
            # Each step adds only three connected pixels, below the threshold
            # of eight; the reference accumulates nine without claiming when
            # the meaningful symbol first appeared within the wider interval.
            offset = 10 + (index - 1) * 6
            cv2.line(frame, (x + offset, y + 130), (x + offset + 2, y + 130), (240, 240, 230), 1)
            self.observe(index, frame=frame.copy())
            if index < 3:
                self.assertEqual(change_events(self.tracker), [])
        events = change_events(self.tracker)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["board_id"], "board_c")
        self.assertEqual(event["interval_ms"], [0, 3000])
        self.assertEqual(event["observed_at_ms"], 3000)
        self.assertEqual(event["before_frame_id"], "f0000")
        self.assertEqual(event["previous_sample_frame_id"], "f0002")
        self.assertEqual(event["after_frame_id"], "f0003")
        self.assertIn("accumulated_change_since_reference", event["selected_reasons"])
        self.assertIn("change_accumulated_over_multiple_samples", event["uncertainty"])

    def test_occlusion_is_only_a_visual_candidate_and_not_writing(self) -> None:
        for index in range(7):
            self.observe(index)
        events = [e for e in change_events(self.tracker) if e["observed_at_ms"] == 6000]
        self.assertEqual([e["board_id"] for e in events], ["board_a"])
        event = events[0]
        self.assertIn("possible_interference", event["uncertainty"])
        self.assertNotIn("writing_detected", event["selected_reasons"])
        self.assertEqual(event["kind"], "visual_change_candidate")
        # The unchanged board's version must not grow because another board was obscured.
        versions = [v for v in self.tracker.versions if v["board_id"] == "board_b"]
        self.assertEqual(len(versions), 2)

    def test_two_boards_in_one_sample_interval_share_unordered_group(self) -> None:
        for index in range(8):
            self.observe(index)
        events = [e for e in change_events(self.tracker) if e["observed_at_ms"] == 7000]
        self.assertEqual({e["board_id"] for e in events}, {"board_b", "board_d"})
        self.assertEqual(len(events), 2)
        group = events[0]["simultaneous_group"]
        self.assertIsNotNone(group)
        self.assertEqual(events[1]["simultaneous_group"], group)
        for event in events:
            self.assertEqual(event["interval_ms"], [6000, 7000])
            self.assertEqual(event["order_status"], "undetermined_within_interval")
            self.assertTrue(event["uncertainty"])

    def test_one_or_five_boards_do_not_imply_a_fixed_board_count(self) -> None:
        for count in (1, 5):
            boards = [BoardSpec(f"independent_{i}", f"位置 {i}", (i * 48 + 1, 1, 40, 40)) for i in range(count)]
            tracker = MultiBoardTracker(boards, DetectorConfig())
            frame = np.full((60, count * 48 + 1, 3), 70, dtype=np.uint8)
            tracker.observe(frame, "initial", 0, "frames/initial.png",
                            {b.board_id: f"crops/{b.board_id}/initial.png" for b in boards})
            self.assertEqual(len(tracker.events), count)
            self.assertEqual({e["board_id"] for e in tracker.events}, {b.board_id for b in boards})

    def test_resolution_change_requires_reconfirmation_and_stops_versions(self) -> None:
        self.observe(0)
        count = len(self.tracker.versions)
        frame = cv2.resize(self.samples[1].frame, (800, 500))
        record = self.observe(1, frame=frame)
        self.assertEqual(len(self.tracker.versions), count)
        self.assertTrue(all(b["roi_status"] == "needs_reconfirmation" for b in record["boards"]))
        self.assertEqual(len(change_events(self.tracker)), 0)
        self.assertTrue(any(e["kind"] == "roi_needs_reconfirmation" for e in self.tracker.events))
        # Returning to the old resolution cannot silently re-establish identity.
        self.observe(2)
        self.assertEqual(len(self.tracker.versions), count)

    def test_single_board_motion_requires_reconfirmation_without_relabeling_others(self) -> None:
        frame = self.samples[4].frame.copy()
        self.observe(0, frame=frame)
        moved = frame.copy()
        board = next(b for b in board_config() if b["board_id"] == "board_a")
        x, y, width, height = board["roi"]
        moved[y:y+height, x:x+width] = cv2.warpAffine(
            frame[y:y+height, x:x+width], np.float32([[1,0,16],[0,1,0]]),
            (width,height), borderValue=(42,74,35))
        record = self.observe(1, frame=moved)
        states = {b["board_id"]: b["roi_status"] for b in record["boards"]}
        self.assertEqual(states["board_a"], "needs_reconfirmation")
        self.assertTrue(all(status == "confirmed" for name,status in states.items() if name != "board_a"))
        self.assertEqual(len(self.tracker.versions), 4)

    def test_large_camera_shift_requires_reconfirmation(self) -> None:
        self.observe(0)
        before = len(self.tracker.versions)
        transform = np.float32([[1, 0, 45], [0, 1, 20]])
        shifted = cv2.warpAffine(self.samples[0].frame, transform, (640, 400), borderValue=(0, 0, 0))
        record = self.observe(1, frame=shifted)
        self.assertTrue(all(b["roi_status"] == "needs_reconfirmation" for b in record["boards"]))
        self.assertEqual(len(self.tracker.versions), before)


class SyntheticVideoIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="icourse-multiboard-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_demo(self) -> dict[str, str]:
        return write_demo(self.root / "fixture")

    def test_video_preserves_frame_and_board_evidence_with_correct_timeline(self) -> None:
        demo = self.make_demo()
        output = self.root / "output"
        manifest_path = run_video(Path(demo["video"]), Path(demo["config"]), output)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        frames = manifest["frames"]
        self.assertEqual(len(frames), 8)
        self.assertEqual([f["timestamp_ms"] for f in frames], list(range(0, 8000, 1000)))
        self.assertEqual(manifest["source"]["sampled_frame_count"], 8)
        self.assertEqual(len(list((output / "frames").glob("*.png"))), 8)
        for frame in frames:
            self.assertTrue((output / frame["image_path"]).is_file())
            self.assertEqual(len(frame["boards"]), 4)
            for board in frame["boards"]:
                self.assertTrue((output / board["crop_path"]).is_file())
        changes = [e for e in manifest["events"] if e["kind"] == "visual_change_candidate"]
        self.assertEqual([e["board_id"] for e in changes[:5]], ["board_c", "board_a", "board_d", "board_b", "board_c"])
        last = [e for e in changes if e["observed_at_ms"] == 7000]
        self.assertEqual({e["board_id"] for e in last}, {"board_b", "board_d"})
        self.assertEqual({tuple(e["interval_ms"]) for e in last}, {(6000, 7000)})
        self.assertEqual(len({e["simultaneous_group"] for e in last}), 1)
        self.assertNotIn(None, {e["simultaneous_group"] for e in last})
        self.assertTrue((output / "review.html").is_file())
        self.assertTrue(manifest["versions"])
        self.assertTrue(manifest["writing_runs"])
        for event in changes:
            self.assertTrue((output / event["before_crop_path"]).is_file())
            self.assertTrue((output / event["after_crop_path"]).is_file())

    def test_real_ffprobe_native_pts_matches_manifest_sampling(self) -> None:
        """Exercise the installed binaries and actual media, with no tool mocks."""
        demo = self.make_demo()
        probe = tool_path("ffprobe")
        completed = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-show_frames",
             "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time:stream=time_base",
             "-of", "json", demo["video"]],
            capture_output=True, text=True, check=True, timeout=30,
        )
        native = json.loads(completed.stdout)
        timestamps = [round(float(frame["best_effort_timestamp_time"]) * 1000)
                      for frame in native["frames"]]
        self.assertEqual(timestamps, list(range(0, 8000, 250)))
        manifest = json.loads(run_video(Path(demo["video"]), Path(demo["config"]),
                                        self.root / "native_output").read_text(encoding="utf-8"))
        for frame in manifest["frames"]:
            self.assertIn(frame["timestamp_ms"], timestamps)
            self.assertEqual(frame["time_status"], "verified_pts")
            self.assertEqual(frame["pts_time_ms"], frame["timestamp_ms"])
            self.assertIsInstance(frame["native_pts"], int)
            self.assertEqual(round(frame["native_pts"] * Fraction(frame["native_time_base"]) * 1000),
                             frame["pts_time_ms"])
        for event in manifest["events"]:
            self.assertEqual(event["source_interval_ms"], event["interval_ms"])
            self.assertIsInstance(event["source_frame"]["native_pts"], int)

    def test_all_unchanged_sampled_frames_are_retained(self) -> None:
        frame = multiboard_samples()[0]
        video = self.root / "unchanged.avi"
        write_video(video, [frame, frame, frame])
        config = self.root / "boards.json"
        config.write_text(json.dumps({"sample_interval_ms": 1000, "boards": board_config()}), encoding="utf-8")
        output = self.root / "unchanged_output"
        manifest = json.loads(run_video(video, config, output).read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["frames"]), 3)
        self.assertEqual(len(manifest["events"]), 4)
        self.assertEqual(len(list((output / "frames").glob("*.png"))), 3)

    def test_output_existing_directory_is_never_overwritten(self) -> None:
        demo = self.make_demo()
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("do not overwrite", encoding="utf-8")
        with self.assertRaises((FileExistsError, ValueError)):
            run_video(Path(demo["video"]), Path(demo["config"]), output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "do not overwrite")

    def test_url_input_is_rejected_without_network_request(self) -> None:
        demo = self.make_demo()
        with self.assertRaises((ValueError, FileNotFoundError)):
            run_video(Path("https://example.invalid/never-download-course.mp4"), Path(demo["config"]), self.root / "no_url")

    def test_duplicate_board_ids_are_rejected(self) -> None:
        demo = self.make_demo()
        boards = board_config()
        boards[1]["board_id"] = boards[0]["board_id"]
        config = self.root / "duplicates.json"
        config.write_text(json.dumps({"sample_interval_ms": 1000, "boards": boards}), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_video(Path(demo["video"]), config, self.root / "duplicate_output")

    def test_out_of_bounds_roi_is_rejected(self) -> None:
        demo = self.make_demo()
        boards = board_config()
        boards[0]["roi"] = [630, 390, 100, 100]
        config = self.root / "bounds.json"
        config.write_text(json.dumps({"sample_interval_ms": 1000, "boards": boards}), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_video(Path(demo["video"]), config, self.root / "bounds_output")


if __name__ == "__main__":
    unittest.main()
