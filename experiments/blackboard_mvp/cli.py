"""CLI for the offline prototype. No production modules or services imported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .media import fresh_output, write_probe_preview
from .runner import run_video
from .replay import replay_manifest
from .synthetic import write_demo
from .mechanical_synthetic import write_mechanical_demo


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地黑板视频证据原型；无学校账号、语音分析、OCR或AI API")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="FFprobe探测、首帧预览和人工框选ROI辅助")
    probe.add_argument("--video", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True, help="仓库外且尚不存在的目录")
    analyze = commands.add_parser("analyze", help="独立逐板变化候选、源PTS和离线review.html")
    analyze.add_argument("--video", type=Path, required=True)
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="S0–S2自动几何回放；复用已有采样，不调用视频/模型/数据库")
    replay.add_argument("--input-manifest", type=Path, required=True)
    replay.add_argument("--config", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="从零生成合成视频并运行完整真实工具链")
    demo.add_argument("--output", type=Path, required=True, help="仓库外且尚不存在的目录")
    mechanical = commands.add_parser("mechanical-demo", help="双组件滑动/遮挡合成视频；明确区分人工几何与自动跟踪")
    mechanical.add_argument("--output", type=Path, required=True, help="仓库外且尚不存在的目录")
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            result = {"source_preview": str(write_probe_preview(args.video, args.output))}
        elif args.command == "replay":
            manifest = replay_manifest(args.input_manifest, args.output, args.config)
            result = {"manifest":str(manifest),"review":str(manifest.parent/"review.html")}
        elif args.command == "analyze":
            manifest = run_video(args.video, args.config, args.output)
            result = {"manifest": str(manifest), "review": str(manifest.parent / "review.html")}
        else:
            root = fresh_output(args.output)
            fixture = (write_mechanical_demo if args.command == "mechanical-demo" else write_demo)(root / "input")
            manifest = run_video(Path(fixture["video"]), Path(fixture["config"]), root / "review")
            result = {**fixture, "manifest": str(manifest), "review": str(manifest.parent / "review.html")}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"原型停止（没有覆盖现有目录）：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
