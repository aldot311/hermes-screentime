"""Hand-run check of the proof reel on a run the Screen-Time web app stored (local only; never commit the outputs).

Normalizes the clip the way the app did (15 fps, longest edge 960, libx264, no audio), loads the app's stored report
(camelCase JSON) into the screentime_core dataclasses, renders the reel with the original clip as the audio source,
and saves a few frames as PNG to look at. With --check-against it also compares the new normalized copy with the one
the app stored (PSNR, plus a one-frame shift as the negative control), to show the stored frame indices still apply.

    .venv/bin/python dev/proof_try.py --out-dir /private/tmp/claude-501/proof-try
    .venv/bin/python dev/proof_try.py --frames 40 160 300 --check-against \
        ~/Projects/aetrai/screen-time/data/out/lacroix-can/lacroix/lc-12-aldo/lc-12-aldo.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screentime_core.models import FPS, Box, Detection, Mention, Segment, VideoInfo, VideoMetrics  # noqa: E402
from screentime_core.proof import render_proof  # noqa: E402

APP_DATA = Path.home() / "Projects/aetrai/screen-time/data"
DEFAULT_VIDEO = APP_DATA / "videos/lacroix/lc-12-aldo.mp4"
DEFAULT_REPORT = APP_DATA / "out/lacroix-can/lacroix/lc-12-aldo/report.json"
LONG_EDGE = "2*trunc(min(960,max(iw,ih))/2)"
NORMALIZE_VF = f"fps={FPS},scale=w='if(gte(iw,ih),{LONG_EDGE},-2)':h='if(gte(iw,ih),-2,{LONG_EDGE})',setsar=1,format=yuv420p"


def tool(name: str) -> str:
    path = shutil.which(name) or f"/opt/homebrew/bin/{name}"
    if not Path(path).exists():
        sys.exit(f"{name} not found")
    return path


def normalize(src: Path, dst: Path) -> None:
    subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-vf", NORMALIZE_VF,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-map_metadata", "-1", "-movflags", "+faststart", str(dst)], check=True)


def probe(path: Path) -> dict:
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,codec_name,width,height,nb_read_frames,r_frame_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def load_run(report_path: Path, fps: float) -> tuple[VideoInfo, list[Detection], dict[str, str], VideoMetrics]:
    """The app's report.json (camelCase) as the Python dataclasses render_proof takes."""
    r = json.loads(report_path.read_text())
    inf = r["info"]
    info = VideoInfo(width=int(inf["width"]), height=int(inf["height"]), frame_count=int(inf["frameCount"]),
                     duration_sec=float(inf["durationSec"]), fps=fps, has_audio=True)
    dets = []
    for d in r["detections"]:
        centroid = d.get("centroid") or {}
        dets.append(Detection(frame=int(d["frame"]), object_id=int(d["objectId"]),
                              box=Box(*(float(d["box"][k]) for k in ("x1", "y1", "x2", "y2"))),
                              area_fraction=float(d.get("areaFraction", 0.0)),
                              centroid_x=float(centroid.get("x", 0.5)), centroid_y=float(centroid.get("y", 0.5)),
                              confidence=d.get("confidence")))
    verdicts = {str(t["objectId"]): t["verdict"] for t in r["tracks"]}
    m = r["metrics"]
    segments = [Segment(float(s["startSec"]), float(s["endSec"])) for s in m.get("segments", [])]
    longest = max(segments, key=lambda s: s.end_sec - s.start_sec, default=None)
    matched = [d for d in dets if verdicts.get(str(d.object_id)) == "match"]
    peak = max(matched, key=lambda d: d.area_fraction, default=None)
    metrics = VideoMetrics(
        seconds_on_screen=float(m["secondsOnScreen"]), share_of_video=float(m.get("shareOfVideo", 0.0)),
        first_seen_sec=m.get("firstSeenSec"), longest_run_sec=float(m.get("longestRunSec", 0.0)),
        longest_run_start_sec=longest.start_sec if longest else None,
        avg_area_fraction=float(m.get("avgAreaFraction", 0.0)),
        peak_area_fraction=peak.area_fraction if peak else 0.0, peak_area_sec=peak.frame / fps if peak else None,
        segments=segments, visible_frames=[int(f) for f in m.get("visibleFrames", [])],
        mentions=[Mention(float(x["startSec"]), float(x["endSec"]), x["text"]) for x in m.get("mentions", [])],
    )
    return info, dets, verdicts, metrics


def psnr(a: Path, b: Path, shift: int = 0) -> float:
    """Average PSNR of a against b; ``shift`` drops that many leading frames of a (the misaligned control)."""
    graph = f"[0:v]trim=start_frame={shift},setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]psnr"
    err = subprocess.run([tool("ffmpeg"), "-hide_banner", "-i", str(a), "-i", str(b), "-lavfi", graph,
                          "-f", "null", "-"], capture_output=True, text=True, check=False).stderr
    m = re.search(r"average:([0-9.]+|inf)", err)
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="original clip (also the audio source)")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="the app's stored report.json")
    ap.add_argument("--out-dir", type=Path, default=Path(tempfile.gettempdir()) / "proof-try")
    ap.add_argument("--label", default="LaCroix Strawberry Peach can")
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--frames", type=int, nargs="*", default=[40, 160, 300])
    ap.add_argument("--check-against", type=Path, help="the normalized copy the app stored, to confirm alignment")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    norm = args.out_dir / f"{stem}-15fps.mp4"
    if not norm.exists():
        normalize(args.video, norm)
    info, dets, verdicts, metrics = load_run(args.report, float(FPS))
    got = probe(norm)
    stream = got["streams"][0]
    print(f"normalized: {norm} {stream['width']}x{stream['height']} frames={stream['nb_read_frames']} "
          f"(report: {info.width}x{info.height} frames={info.frame_count})")
    if (int(stream["width"]), int(stream["height"]), int(stream["nb_read_frames"])) != (
            info.width, info.height, info.frame_count):
        print("WARNING: the normalized copy does not match the report's video; boxes may not line up")
    if args.check_against:
        same, shifted = psnr(norm, args.check_against), psnr(norm, args.check_against, shift=1)
        print(f"PSNR vs the app's copy: aligned {same:.2f} dB, shifted one frame {shifted:.2f} dB (control)")

    out = args.out_dir / f"{stem}-proof.mp4"
    t0 = time.monotonic()
    render_proof(str(norm), dets, verdicts, metrics, info, str(out), product_label=args.label,
                 max_height=args.max_height, audio_source=str(args.video), workdir=str(args.out_dir / "work"))
    took = time.monotonic() - t0
    res = probe(out)
    streams = {s["codec_type"]: s for s in res["streams"]}
    v = streams["video"]
    print(f"proof: {out}")
    print(f"  {out.stat().st_size} bytes ({out.stat().st_size / 1e6:.2f} MB), duration {float(res['format']['duration']):.3f} s, "
          f"{v['codec_name']} {v['width']}x{v['height']} {v['r_frame_rate']} fps, frames {v['nb_read_frames']} "
          f"(input {stream['nb_read_frames']}), audio {'yes' if 'audio' in streams else 'no'}")
    print(f"  rendered in {took:.2f} s")
    for n in args.frames:
        png = args.out_dir / f"{stem}-proof-f{n}.png"
        subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(out), "-vf", f"select=eq(n\\,{n})",
                        "-fps_mode", "passthrough", "-frames:v", "1", str(png)], check=True)
        print(f"  frame {n}: {png}")


if __name__ == "__main__":
    main()
