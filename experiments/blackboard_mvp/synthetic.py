"""Generate a tiny, entirely artificial four-board video for offline verification.

This module does not read course materials or contact any service. Spatial labels
are descriptive only; the scripted changes deliberately use a different order.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 400


def board_config() -> list[dict[str, Any]]:
    """Return an intentionally non-spatial configuration order."""
    return [
        {"board_id": "board_d", "label": "右下", "roi": [336, 224, 288, 156]},
        {"board_id": "board_a", "label": "左上", "roi": [16, 32, 288, 156]},
        {"board_id": "board_c", "label": "左下", "roi": [16, 224, 288, 156]},
        {"board_id": "board_b", "label": "右上", "roi": [336, 32, 288, 156]},
    ]


@dataclass(frozen=True)
class SyntheticSample:
    timestamp_ms: int
    frame: np.ndarray
    changed_boards: tuple[str, ...]
    explanation: str
    occluded_board: str | None = None


def _render(marks: dict[str, list[tuple[str, int, int]]], occluded: str | None) -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (185, 185, 185), dtype=np.uint8)
    # Stable non-board landmarks make camera registration testable. They do not
    # represent automatically discovered boards or teacher identity.
    cv2.line(frame, (0, 10), (639, 10), (20, 20, 20), 2)
    for x in range(8, FRAME_WIDTH, 37):
        cv2.circle(frame, (x, 205), 3, (35, 35, 35), -1)
    for board in board_config():
        x, y, width, height = board["roi"]
        cv2.rectangle(frame, (x - 3, y - 3), (x + width + 2, y + height + 2), (25, 25, 25), -1)
        frame[y:y + height, x:x + width] = (42, 74, 35)
        # Every board already has some writing at the beginning of the clip.
        cv2.putText(frame, board["board_id"][-1].upper() + "0", (x + 12, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 230), 1, cv2.LINE_8)
        for text, dx, dy in marks.get(board["board_id"], []):
            if text == "minus":
                cv2.line(frame, (x + dx, y + dy), (x + dx + 9, y + dy), (240, 240, 230), 1)
            else:
                cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (240, 240, 230), 1, cv2.LINE_8)
        if board["board_id"] == occluded:
            # A large moving person-like patch is intentionally only a possible
            # interference; the test must never label it proven new writing.
            cv2.rectangle(frame, (x + 75, y + 3), (x + 185, y + height - 4), (180, 180, 220), -1)
    return frame


def multiboard_samples() -> list[SyntheticSample]:
    """Persistent versions: C -> A -> D -> B -> C edit -> A occlusion -> B+D."""
    marks: dict[str, list[tuple[str, int, int]]] = {b["board_id"]: [] for b in board_config()}
    samples = [SyntheticSample(0, _render(marks, None), (), "起始时已有板书")]
    for index, board_id in enumerate(("board_c", "board_a", "board_d", "board_b"), start=1):
        marks[board_id].append(("x = 1", 26, 65))
        samples.append(SyntheticSample(index * 1000, _render(marks, None), (board_id,), "不同板按非空间顺序变化"))
    marks["board_c"].append(("minus", 61, 61))
    samples.append(SyntheticSample(5000, _render(marks, None), ("board_c",), "返回首块变化板增加小负号"))
    samples.append(SyntheticSample(6000, _render(marks, "board_a"), ("board_a",), "一块板被遮挡，其他板保持原状", "board_a"))
    marks["board_b"].append(("t > 0", 26, 100))
    marks["board_d"].append(("y = 2", 26, 100))
    samples.append(SyntheticSample(7000, _render(marks, "board_a"), ("board_b", "board_d"),
                                   "同采样区间内两板变化，顺序未知", "board_a"))
    return samples


def write_video(path: Path, samples: list[SyntheticSample] | None = None, fps: int = 4) -> str:
    """Write an artificial video, preferring a lossless codec for tiny marks."""
    from .media import outside_repository
    path = outside_repository(Path(path))
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    samples = multiboard_samples() if samples is None else samples
    if fps <= 0 or not samples:
        raise ValueError("fps must be positive and samples must be nonempty")
    height, width = samples[0].frame.shape[:2]
    # FFV1 is lossless and normally available in the local OpenCV/FFmpeg build.
    # MJPG is a portable fallback; tests still verify changes by board and time.
    for codec in ("FFV1", "MJPG"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            if path.exists():
                path.unlink()
            continue
        try:
            for sample in samples:
                for _ in range(fps):
                    writer.write(sample.frame)
        finally:
            writer.release()
        if path.exists() and path.stat().st_size > 0:
            return codec
    raise RuntimeError("No suitable local video encoder is available")


def write_demo(output_dir: Path) -> dict[str, str]:
    """Create only a fresh directory with synthetic video/config/ground truth."""
    from .media import outside_repository
    output_dir = outside_repository(Path(output_dir))
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    samples = multiboard_samples()
    video_path = output_dir / "synthetic_four_boards.avi"
    codec = write_video(video_path, samples)
    config_path = output_dir / "boards.json"
    config_path.write_text(json.dumps({"sample_interval_ms": 1000, "boards": board_config()},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    truth_path = output_dir / "ground_truth.json"
    truth_path.write_text(json.dumps({
        "synthetic_only": True, "codec": codec,
        "warning": "视觉变化顺序不代表讲解或证明顺序；同区间双板变化未规定先后。",
        "observations": [{"timestamp_ms": s.timestamp_ms, "changed_boards": list(s.changed_boards),
                          "explanation": s.explanation, "occluded_board": s.occluded_board} for s in samples],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"video": str(video_path), "config": str(config_path), "ground_truth": str(truth_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New, nonexistent directory")
    args = parser.parse_args()
    print(json.dumps(write_demo(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
