"""Artificial paired sliding blackboards; no classroom files or services.

The exported positions are labelled manual geometry ground truth. They test the
occlusion/state machine, and must not be mistaken for automatic tracking output.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .synthetic import write_video


WIDTH, HEIGHT = 800, 480
BOARD_WIDTH, BOARD_HEIGHT = 300, 180


def assembly_config():
    return [
        {"assembly_id": "left_assembly", "viewport": [20, 35, 340, 400]},
        {"assembly_id": "right_assembly", "viewport": [420, 35, 340, 400]},
    ]


def mechanical_board_config():
    boards = []
    # Configuration order deliberately carries no temporal ordering meaning.
    for side, x in (("right", 440), ("left", 40)):
        for layer, y in (("inner", 240), ("outer", 50)):
            boards.append({"board_id": f"{side}_{layer}",
                           "label": ("左" if side == "left" else "右") + ("外板" if layer == "outer" else "内板"),
                           "assembly_id": f"{side}_assembly", "layer": layer,
                           "roi": [x, y, BOARD_WIDTH, BOARD_HEIGHT]})
    return boards


def positions(left=0, right=0):
    result = {}
    for b in mechanical_board_config():
        x, y, w, h = b["roi"]
        shift = left if b["assembly_id"] == "left_assembly" else right
        y += shift if b["layer"] == "outer" else -shift
        result[b["board_id"]] = [x, y, w, h]
    return result


def _board_surface(board_id, minus=False, hidden_edit=False):
    # Reproducible distinctive board-local chalk marks survive translation.
    seed = {b["board_id"]: i + 1 for i, b in enumerate(mechanical_board_config())}[board_id]
    rng = np.random.default_rng(seed)
    result = np.full((BOARD_HEIGHT, BOARD_WIDTH, 3), (40, 73, 34), np.uint8)
    for _ in range(72):
        x = int(rng.integers(8, BOARD_WIDTH - 9))
        y = int(rng.integers(8, BOARD_HEIGHT - 9))
        color = tuple(int(v) for v in rng.integers(120, 230, size=3))
        cv2.line(result, (x, y), (x + int(rng.integers(2, 7)), y + int(rng.integers(-3, 4))), color, 1)
    cv2.rectangle(result, (1, 1), (BOARD_WIDTH - 2, BOARD_HEIGHT - 2), (80, 100, 74), 2)
    cv2.putText(result, board_id, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 235, 225), 1)
    cv2.putText(result, "x(t) = A x(0)", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, .62, (235, 235, 220), 1)
    cv2.putText(result, "rank W = n", (18, 125), cv2.FONT_HERSHEY_SIMPLEX, .62, (235, 235, 220), 1)
    if minus:
        cv2.line(result, (88, 92), (99, 92), (240, 240, 230), 1)
    if hidden_edit:
        cv2.putText(result, "hidden", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 230), 1)
    return result


def render_mechanical(board_positions, minus_boards=(), hidden_edit_boards=(), obscured_boards=()):
    frame = np.full((HEIGHT, WIDTH, 3), (170, 170, 170), np.uint8)
    for a in assembly_config():
        x, y, w, h = a["viewport"]
        cv2.rectangle(frame, (x - 4, y - 4), (x + w + 3, y + h + 3), (45, 45, 45), -1)
        frame[y:y+h, x:x+w] = (25, 30, 25)
    # Static camera landmarks, outside both assemblies.
    for y in range(15, HEIGHT, 29):
        cv2.circle(frame, (390, y), 3, (230, 230, 230), -1)
    specs = mechanical_board_config()
    viewports = {a["assembly_id"]: a["viewport"] for a in assembly_config()}
    for layer in ("inner", "outer"):
        for b in specs:
            if b["layer"] != layer:
                continue
            x, y, w, h = board_positions[b["board_id"]]
            ax, ay, aw, ah = viewports[b["assembly_id"]]
            x0, y0, x1, y1 = max(0, x, ax), max(0, y, ay), min(WIDTH, x+w, ax+aw), min(HEIGHT, y+h, ay+ah)
            if x1 <= x0 or y1 <= y0:
                continue
            surface = _board_surface(b["board_id"], b["board_id"] in minus_boards, b["board_id"] in hidden_edit_boards)
            frame[y0:y1, x0:x1] = surface[y0-y:y1-y, x0-x:x1-x]
            if b["board_id"] in obscured_boards:
                # Deliberately erase match evidence with an unrelated occluder.
                frame[y0:y1, x0:x1] = (155, 145, 190)
    return frame


@dataclass(frozen=True)
class MechanicalSample:
    timestamp_ms: int
    frame: np.ndarray
    positions: dict
    scenario: str
    expected: dict


def mechanical_samples():
    scripts = [
        (0, 0, (), (), (), "起始已有内容", {"initial": True}),
        (40, 0, (), (), (), "A 外板下移、内板反向上移", {"motion": ["left_outer", "left_inner"]}),
        (20, 0, (), (), (), "B 外板上推、内板反向下移", {"motion": ["left_outer", "left_inner"]}),
        (85, 0, (), (), (), "C 内板大部分被遮挡", {"visibility": {"left_inner": "partially_occluded"}}),
        (95, 0, (), ("left_inner",), (), "D 内板完全遮挡，隐藏区改变不可观察", {"visibility": {"left_inner": "occluded"}}),
        (60, 0, (), (), (), "E 内板重新出现，保留原身份", {"visibility": {"left_inner": "partially_occluded"}}),
        (0, 0, (), (), (), "F 恢复完整可见，历史版本不变", {"visibility": {"left_inner": "visible"}}),
        (35, 0, ("left_outer",), (), (), "G 移动过程中外板增加负号", {"ambiguous_motion_writing": "left_outer"}),
        (35, 45, ("left_outer",), (), (), "H 右组件独立运动，左组件不动", {"motion": ["right_outer", "right_inner"]}),
        (35, 45, ("left_outer",), (), ("right_outer",), "I 无可靠纹理时身份待复核", {"identity_uncertain": "right_outer"}),
        (85, 45, ("left_outer",), (), (), "J 重叠后仅比较共同可见板面", {"visibility": {"left_inner": "partially_occluded"}}),
        (85, 45, ("left_outer",), (), (), "持续可见的旧内容不新增", {"unchanged": True}),
    ]
    samples = []
    for i, (left, right, minus, hidden, obscured, scenario, expected) in enumerate(scripts):
        p = positions(left, right)
        samples.append(MechanicalSample(i * 1000, render_mechanical(p, minus, hidden, obscured), p, scenario, expected))
    return samples


def write_mechanical_demo(output_dir):
    from .media import outside_repository
    output_dir = outside_repository(Path(output_dir))
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    samples = mechanical_samples()
    video = output_dir / "synthetic_sliding_boards.avi"
    codec = write_video(video, samples, fps=4)
    config = {
        "sample_interval_ms": 1000,
        "geometry_source": "synthetic_ground_truth_manual_geometry",
        "assemblies": assembly_config(), "boards": mechanical_board_config(),
        "tracking_mode":"oracle_geometry", "position_keyframes": [
            {"timestamp_ms": s.timestamp_ms, "positions": s.positions,
             "provenance": "synthetic_ground_truth_manual_geometry"}
            for s in samples if s.timestamp_ms != 9000
        ],
    }
    # Frame I deliberately has no human geometry confirmation, to exercise
    # fail-closed identity handling rather than blessing an obscured image.
    config_path = output_dir / "boards.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    truth_path = output_dir / "ground_truth.json"
    truth_path.write_text(json.dumps({
        "synthetic_only": True, "codec": codec,
        "geometry_source": "synthetic_ground_truth_manual_geometry",
        "warning": "遮挡/身份状态机由已知合成几何验证；这不是自动跟踪精度证明。",
        "observations": [{"timestamp_ms": s.timestamp_ms, "scenario": s.scenario,
                          "positions": s.positions, "expected": s.expected} for s in samples],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"video": str(video), "config": str(config_path), "ground_truth": str(truth_path)}
