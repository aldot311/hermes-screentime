"""/screentime: help, demo, selftest, status, last. Runs in the gateway process; returns the text to show."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import hermes_bridge as hb
from .screentime_core.errors import ScreentimeError

log = logging.getLogger(__name__)

HELP = """Screen-Time: measured seconds for video.
Ask in plain words, with a video attached:
  "How long is the LaCroix Strawberry Peach can on screen? Terms: at least 3 s in total, shown in the first 5 s."
  "Compare my draft with this reference reel."
Commands:
  /screentime demo      run Measure on a bundled clip (recorded API run, no key, no cost) and send the proof reel
  /screentime selftest  check ffmpeg, the cut detector on a generated clip, and the bundled demo
  /screentime status    key, product check, spend today
  /screentime last      your latest result
Measure uses Meta's SAM 3.1 when a Meta Model API key is set (Settings > Plugins > screentime), otherwise your own
Hermes vision model and speech-to-text. Compare needs no model at all."""


def _media_line(path: str, platform: str) -> str:
    return path if platform in ("cli", "tui") else f"MEDIA:{path}"


def run_demo(owner: str = "local", *, render: bool = True):
    """The full Measure pipeline on the bundled clip with the recorded SAM 3.1 run and recorded product checks."""
    from .screentime_core.pipeline import Engines, MeasureRequest, ReplayData, run_measure
    from .screentime_core.store import new_run_id
    from .screentime_core.verify import ReplayVerifier
    from .tools import terms_from_args

    rp = ReplayData.load(hb.DEMO_DIR)
    run_id = new_run_id("demo")
    req = MeasureRequest(
        video_path=rp.video_path,
        product=rp.product,
        sam_prompt=rp.sam_prompt,
        owner=owner,
        photo_path=rp.photo_path,
        brand_names=rp.brand_names,
        terms=terms_from_args([
            {"text": "On screen at least 3 seconds in total", "kind": "min_seconds_total", "seconds": 3},
            {"text": "Says LaCroix out loud", "kind": "mentioned", "min_times": 1},
            {"text": "Discloses the ad: says it is an ad or sponsored", "kind": "disclosed"},
        ]),
    )
    engines = Engines("replay", ReplayVerifier(rp.verdicts), None, transcribe=False, replay=rp)
    return run_measure(req, engines, workdir=hb.work_dir() / run_id, outdir=hb.out_dir(run_id), budget=None,
                       run_id=run_id, render_proof=render)


def _status() -> str:
    from .screentime_core.budget import Budget

    key = "set" if hb.meta_key() else "not set (Measure runs on your own Hermes models)"
    ff = shutil.which("ffmpeg") or "NOT FOUND (install ffmpeg)"
    try:
        spent = Budget(hb.db(), per_video_usd=hb.setting("max_usd_per_video", 0.5),
                       per_day_usd=hb.setting("max_usd_per_day", 5.0)).spent_today()
    except Exception:
        spent = 0.0
    return (
        f"Meta key: {key}\nffmpeg: {ff}\nTracker: {hb.setting('tracker', 'auto')}\n"
        f"Product check: {hb.setting('verifier', 'auto')}\nSpeech-to-text: {hb.setting('speech', 'auto')}\n"
        f"Spent today: ${spent:.3f} of ${hb.setting('max_usd_per_day', 5.0):.2f} (cap per video "
        f"${hb.setting('max_usd_per_video', 0.5):.2f})"
    )


def _selftest() -> str:
    """No network, no key: ffmpeg, the cut detector on a clip with known cuts, and the bundled demo pipeline."""
    lines = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "FAIL ffmpeg not found on PATH. Install it: brew install ffmpeg / apt install ffmpeg"
    ver = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20, check=False).stdout
    lines.append(f"OK   {ver.splitlines()[0] if ver else 'ffmpeg'}")
    from .screentime_core import teardown

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "cuts.mp4"
        # Three visually different 1.2 s sources: true cuts at 1.2 s and 2.4 s.
        srcs = ["testsrc2=s=320x240:r=30:d=1.2", "color=c=red:s=320x240:r=30:d=1.2", "smptebars=s=320x240:r=30:d=1.2"]
        cmd = [ffmpeg, "-v", "error", "-y"]
        for s in srcs:
            cmd += ["-f", "lavfi", "-i", s]
        cmd += ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-pix_fmt", "yuv420p", str(clip)]
        subprocess.run(cmd, check=True, timeout=60)
        rep = teardown.analyze(str(clip), str(Path(tmp) / "w"))
        ok = len(rep.cuts_sec) == 2 and all(abs(a - b) <= 0.1 for a, b in zip(rep.cuts_sec, [1.2, 2.4], strict=False))
        lines.append(f"{'OK  ' if ok else 'FAIL'} cut detector: found {[round(c, 2) for c in rep.cuts_sec]} (truth [1.2, 2.4])")
    try:
        r = run_demo(render=False)
        counted = [t for t in r.tracks if t.verdict == "match"]
        rejected = [t for t in r.tracks if t.verdict == "lookalike"]
        ok = bool(counted) and bool(rejected) and r.metrics.seconds_on_screen > 0
        lines.append(f"{'OK  ' if ok else 'FAIL'} demo pipeline: {r.metrics.seconds_on_screen:.1f} s counted, "
                     f"{len(rejected)} look-alike rejected (replayed run)")
    except Exception as exc:
        lines.append(f"FAIL demo pipeline: {type(exc).__name__}: {str(exc)[:160]}")
    return "\n".join(lines)


def handle(raw_args: str, *, session_key: str = "", platform: str = "", **_kw: Any) -> str:
    sub = (raw_args or "").strip().split()
    cmd = sub[0].lower() if sub else "help"
    owner = hb.owner_key(session_key)  # None: a gateway chat we cannot identify (results not stored)
    try:
        if cmd in ("help", "?"):
            return HELP
        if cmd == "status":
            return _status()
        if cmd == "selftest":
            return _selftest()
        if cmd == "demo":
            from .screentime_core.format import measure_summary
            from .screentime_core.store import RunStore

            report = run_demo(owner or "")
            text = measure_summary(report)
            if owner:
                RunStore(hb.db()).save(report, owner=owner, summary="demo: " + text.splitlines()[0])
            if report.proof_path:
                text += "\n" + _media_line(report.proof_path, platform)
            return text
        if cmd == "last":
            from .screentime_core.store import RunStore

            if not owner:
                return "This chat could not be identified, so earlier results are not shown."
            recent = RunStore(hb.db()).recent(owner=owner, limit=1)
            if not recent:
                return "No Screen-Time runs in this chat yet. Try /screentime demo."
            r = recent[0]
            return f"{r['run_id']} ({r['mode']}): {r['summary']}"
        return HELP
    except ScreentimeError as exc:
        return f"{exc.message}" + (f"\n{exc.fix}" if exc.fix else "")
    except Exception as exc:
        log.exception("/screentime %s failed", cmd)
        return f"/screentime {cmd} failed: {type(exc).__name__}: {str(exc)[:200]}"


def as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
