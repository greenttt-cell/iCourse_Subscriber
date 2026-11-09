"""Real FFprobe/FFmpeg, restricted to local files; never estimates frame PTS."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def local_video(value: Path | str) -> Path:
    text = str(value)
    if "://" in text or text.startswith(("http:", "https:", "rtsp:", "file:")):
        raise ValueError("Only a local video file path is accepted; URLs are forbidden")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Video input must be a regular local file")
    outside_repository(path)
    return path


def outside_repository(output: Path):
    target = output.expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            if target == parent or parent in target.parents:
                raise ValueError("Video and runtime output must be outside the Git repository")
            break
    return target


def fresh_output(value: Path) -> Path:
    path = outside_repository(value)
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    return path


def tool_path(name: str) -> str:
    configured = os.environ.get(name.upper() + "_BINARY")
    candidates = [configured] if configured else []
    candidates += [shutil.which(name), str(Path.home() / ".local/share/icourse-blackboard-tools/ffmpeg-9.0.2/bin" / name)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError(f"{name} is required. Set {name.upper()}_BINARY or install a trusted local toolchain")


def run_tool(arguments: list[str], timeout=120) -> bytes:
    # Spool tool output outside the repository, avoiding unbounded RAM capture.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run(arguments, stdout=stdout, stderr=stderr,
                                timeout=timeout, stdin=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise RuntimeError(f"{Path(arguments[0]).name} failed (exit {result.returncode}); input/output were not replaced")
        stdout.seek(0)
        data = stdout.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Tool metadata exceeds prototype limit")
        return data


def probe_media(video: Path, max_duration_ms=600_000) -> dict:
    video = local_video(video)
    binary = tool_path("ffprobe")
    data = json.loads(run_tool([binary, "-v", "error", "-protocol_whitelist", "file,pipe",
                               "-show_streams", "-show_format", "-of", "json", str(video)]))
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError("Input has no non-cover-art video stream")
    selected = videos[0]
    width, height = int(selected["width"]), int(selected["height"])
    duration = data.get("format", {}).get("duration") or selected.get("duration")
    duration_ms = round(float(duration) * 1000) if duration is not None else None
    if duration_ms is None:
        raise ValueError("Source duration is missing; normalize/shorten a local copy before analysis")
    if duration_ms <= 0 or duration_ms > max_duration_ms:
        raise ValueError("Prototype accepts only a bounded short clip; configure max_duration_ms explicitly if required")
    if min(width, height) <= 0 or width * height > 40_000_000:
        raise ValueError("Unsupported video dimensions")
    time_base = selected.get("time_base")
    if not time_base or Fraction(time_base) <= 0:
        raise ValueError("Source video has no usable time_base")
    return {
        "kind": "local_video", "input_name": video.name,
        "size_bytes": video.stat().st_size,
        "sha256": _file_hash(video), "width": width, "height": height,
        "duration_ms": duration_ms, "codec_name": selected.get("codec_name"),
        "video_stream_index": int(selected["index"]), "time_base": time_base,
        "stream_start_time": selected.get("start_time"),
        "avg_frame_rate": selected.get("avg_frame_rate"),
        "roi_coordinate_space": "coded pixels; FFmpeg autorotation disabled",
        "streams": [{key: s.get(key) for key in ("index", "codec_type", "codec_name", "width", "height", "time_base", "start_time", "duration")}
                    for s in streams],
        "metadata_source": "ffprobe", "source_kind_confirmation": "manual_required",
        "tools": {name: run_tool([tool_path(name), "-version"]).decode("utf-8").splitlines()[0]
                  for name in ("ffmpeg", "ffprobe")},
    }


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frames(video: Path, probe: dict, sample_interval_ms: int, max_duration_ms=600_000):
    if type(sample_interval_ms) is not int or sample_interval_ms <= 0:
        raise ValueError("Sampling interval must be positive integer milliseconds")
    if type(max_duration_ms) is not int or max_duration_ms <= 0:
        raise ValueError("Duration limit must be positive integer milliseconds")
    data = json.loads(run_tool([
        tool_path("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", str(probe["video_stream_index"]), "-show_frames",
        "-show_entries", "frame=pts,best_effort_timestamp,pts_time,best_effort_timestamp_time,width,height",
        "-of", "json", str(video),
    ]))
    time_base = Fraction(probe["time_base"])
    selected, previous, origin, target = [], None, None, Fraction(0)
    for index, frame in enumerate(data.get("frames", [])):
        native = frame.get("pts")
        origin_kind = "pts"
        if native is None:
            native = frame.get("best_effort_timestamp")
            origin_kind = "best_effort_timestamp"
        if native is None:
            raise ValueError("Source frame lacks PTS; FPS-based timestamp guessing is disabled")
        native = int(native)
        source_ms = native * time_base * 1000
        if previous is not None and source_ms <= previous:
            raise ValueError("Nonmonotonic/duplicate source PTS requires manual review")
        previous = source_ms
        if origin is None:
            origin = source_ms
        relative = source_ms - origin
        if relative > max_duration_ms:
            raise ValueError("Decoded timeline exceeds prototype duration limit")
        if relative >= target:
            selected.append({
                "source_index": index, "native_pts": native,
                "native_time_base": probe["time_base"], "pts_origin": origin_kind,
                "pts_time_ms": round(source_ms), "timestamp_ms": round(relative),
                "time_status": "verified_pts" if origin_kind == "pts" else "best_effort_pts",
                "width": frame.get("width", probe["width"]),
                "height": frame.get("height", probe["height"]),
            })
            while target <= relative:
                target += sample_interval_ms
        if len(selected) > 10000:
            raise ValueError("Too many sampled frames; stopped instead of discarding samples")
    if not selected:
        raise ValueError("No timestamped source frames")
    probe["timeline_origin_ms"] = round(origin)
    probe["decoded_frame_count"] = len(data["frames"])
    return selected


# Keep each argv filter bounded and its expression-tree depth logarithmic.
_SELECT_BATCH_SIZE = 512
_MAX_SAMPLED_FRAMES = 10000
_MAX_EXACT_FRAME_INDEX = 2**53 - 1


def _selection_indices(samples):
    if not isinstance(samples, (list, tuple)) or not samples or len(samples) > _MAX_SAMPLED_FRAMES:
        raise ValueError("Provide 1..10000 sampled frames; none are silently dropped")
    indices = []
    previous = -1
    for item in samples:
        index = item.get("source_index") if isinstance(item, dict) else None
        if type(index) is not int or index < 0 or index > _MAX_EXACT_FRAME_INDEX:
            raise ValueError("source_index must be a nonnegative exactly representable integer")
        if index <= previous:
            raise ValueError("source_index must increase strictly without duplicates")
        indices.append(index)
        previous = index
    return indices


def _balanced_select_expression(indices):
    """Avoid the bounded expression-tree depth in libavutil/eval.c."""
    def term(first, stop):
        if stop - first == 1:
            return f"eq(n\\,{indices[first]})"
        middle = (first + stop) // 2
        return "(" + term(first, middle) + "+" + term(middle, stop) + ")"
    return "select=" + term(0, len(indices))


def extract_frames(video: Path, probe: dict, samples: list[dict], folder: Path):
    """Extract every requested native source index; no resampling or FPS guessing.

    One batch suffices for the 197-frame regression. Very large selections use
    bounded batches and decode the same source again, retaining global filenames.
    Failure preserves all files already produced and never retries or overwrites.
    """
    indices = _selection_indices(samples)
    stream_index = probe.get("video_stream_index")
    if type(stream_index) is not int or stream_index < 0:
        raise ValueError("video_stream_index must be a nonnegative integer")
    video = local_video(video)
    folder = outside_repository(Path(folder))
    folder.mkdir(mode=0o700, exist_ok=False)
    # image2 parses '%' in the ENTIRE filename pattern, including parent paths.
    output_pattern = str(folder).replace("%", "%%") + "/f%04d.png"
    batches = []
    for start in range(0, len(indices), _SELECT_BATCH_SIZE):
        selected = indices[start:start + _SELECT_BATCH_SIZE]
        expression = _balanced_select_expression(selected)
        run_tool([
            tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-protocol_whitelist", "file,pipe", "-noautorotate", "-i", str(video),
            "-map", "0:" + str(stream_index), "-an", "-sn", "-dn",
            "-vf", expression, "-fps_mode", "vfr", "-threads", "1",
            "-start_number", str(start), output_pattern,
        ])
        # A missing source index must stop this batch, not shift later metadata.
        expected = {f"f{i:04d}.png" for i in range(start + len(selected))}
        if {p.name for p in folder.iterdir()} != expected:
            raise RuntimeError("Extracted frame count/names do not match selected native PTS")
        batches.append({"output_start_number": start, "selected_frame_count": len(selected),
                        "first_source_index": selected[0], "last_source_index": selected[-1]})
    return {"selector": "balanced_select_batches", "batch_size_limit": _SELECT_BATCH_SIZE,
            "batch_count": len(batches), "selected_frame_count": len(indices), "batches": batches}


def write_probe_preview(video: Path, output: Path) -> Path:
    """First actual frame plus a manual rectangle helper; never detects boards."""
    video = local_video(video)
    probe = probe_media(video)
    frames = source_frames(video, probe, 1000)
    output = fresh_output(output)
    extract_frames(video, probe, frames[:1], output / "frames")
    (output / "probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = html.escape(json.dumps(probe, ensure_ascii=False, indent=2))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'"><title>本地视频探测与手动ROI</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f5f8;color:#192536}canvas{max-width:100%;border:1px solid #888;cursor:crosshair}pre,textarea{background:white;padding:14px;max-width:100%;box-sizing:border-box}textarea{width:100%;height:260px}button{padding:10px;margin:8px}</style>
<h1>本地视频探测与手动 ROI</h1><p>原视频的首个真实帧。拖动框选任意数量矩形；没有自动黑板检测。位置标签不代表教学顺序。保存前请人工修改 board_id 和位置标签。</p>
<canvas id="canvas"></canvas><p id="coordinate"></p><button id="undo">撤销最后一个区域</button><textarea id="config" spellcheck="false"></textarea>
<details><summary>FFprobe 媒体元信息</summary><pre>__METADATA__</pre></details>
<script>
'use strict';const c=document.getElementById('canvas'),ctx=c.getContext('2d'),im=new Image(),boards=[];let start=null;
function pos(e){const r=c.getBoundingClientRect();return [Math.round((e.clientX-r.left)*c.width/r.width),Math.round((e.clientY-r.top)*c.height/r.height)];}
function draw(){ctx.drawImage(im,0,0);ctx.lineWidth=2;ctx.font='18px system-ui';for(const b of boards){ctx.strokeStyle='#ffcc00';ctx.strokeRect(...b.roi);ctx.fillStyle='#ffcc00';ctx.fillText(b.board_id,b.roi[0]+4,b.roi[1]+22);}document.getElementById('config').value=JSON.stringify({sample_interval_ms:1000,boards},null,2);}
im.onload=()=>{c.width=im.naturalWidth;c.height=im.naturalHeight;draw();};im.src='frames/f0000.png';
c.onmousedown=e=>{start=pos(e);};c.onmousemove=e=>{document.getElementById('coordinate').textContent='源像素坐标: '+pos(e).join(', ');};
c.onmouseup=e=>{if(!start)return;const p=pos(e),x=Math.min(start[0],p[0]),y=Math.min(start[1],p[1]),w=Math.abs(p[0]-start[0]),h=Math.abs(p[1]-start[1]);start=null;if(w>0&&h>0){const id='board_'+(boards.length+1);boards.push({board_id:id,label:'位置 '+(boards.length+1),roi:[x,y,w,h]});draw();}};
document.getElementById('undo').onclick=()=>{boards.pop();draw();};
</script></html>""".replace("__METADATA__", metadata)
    path = output / "source-preview.html"
    path.write_text(page, encoding="utf-8")
    return path
