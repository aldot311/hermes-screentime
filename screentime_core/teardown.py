"""Teardown and Compare: a short video measured shot by shot with ffmpeg alone (no model, no key).

``analyze`` runs one ffmpeg pass over the video:

- video: ``fps`` to the analysis rate, scaled to 360 px high and pinned to limited-range yuv420p (so the scores do not
  depend on which outputs the pass writes), ``scdet`` (scene change score and mafd per frame) printed with
  ``metadata=print`` to a file, and an MJPEG analysis copy ``<run_id>.analysis.mkv`` in the workdir.
- audio: resampled to 48 kHz, ``ebur128=metadata=1`` (EBU R128 loudness in 100 ms blocks), ``silencedetect``, then the
  BS.1770 K-weighting (two ``biquad`` filters with ebur128's own constants) and ``astats`` per 100 ms block, printed with
  ``ametadata=print`` to a file. ebur128 reports its 3 s short-term loudness only once 3 s of audio have passed (-120.7
  before that), so the curve's first 3 s come from the K-weighted blocks over the audio so far; from 3 s on it is
  ebur128's own value (the two agree to 0.01 LU where both exist).

``compare`` lines two reports up metric by metric with neutral differences ("1.5 s later", "3 fewer", "same").
Nothing here scores, ranks or predicts.

Every ffmpeg and ffprobe call that reads a user's file passes ``-protocol_whitelist file,pipe`` and a demuxer whitelist
before ``-i``, and ``analyze`` refuses anything ffprobe does not identify as MP4/MOV, Matroska/WebM, AVI or MPEG-TS.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .errors import ScreentimeError
from .models import CompareReport, CompareRow, Segment, Shot, TeardownReport, VideoInfo

PathLike = str | Path

ANALYSIS_HEIGHT = 360
"""Analysis frames are scaled to this height (width keeps the aspect ratio, rounded to even)."""

BLOCK_SEC = 0.1
"""ebur128 with metadata=1 cuts the audio into 100 ms blocks."""

SHORT_TERM_WINDOW_SEC = 3.0
LRA_MIN_AUDIO_SEC = 4.0
"""Loudness range needs several 3 s short-term windows: ebur128 reports a meaningless 20 LU for a steady 3.0 s tone and
0 LU from 4 s on, so a shorter track gets None."""
LOUDNESS_STEP_SEC = 0.5
LOUDNESS_FLOOR_LUFS = -70.0
"""Curve values below this (the EBU R128 absolute gate) are stored as -70; an integrated value at or under it means
the track is silent."""

PIXEL_CHANGE_SKIP_FRAMES = 2
"""Frames skipped at every detected cut (the cut frame and the one after) when averaging pixel change."""

CUTS_WINDOW_SEC = 10.0
EARLY_WINDOW_SEC = 3.0
START_TOLERANCE_SEC = 0.05
"""A silence counts as starting at 0 when it starts within this of the audio start."""

MAX_DURATION_SEC = 600.0
"""analyze() refuses longer videos with ``too_long`` unless the caller raises the limit."""

ALLOWED_DEMUXERS = ("mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "avi", "mpegts")
"""Name parts of the demuxers analyze() accepts: "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "avi", "mpegts"."""

SAFE_INPUT_ARGS = ("-protocol_whitelist", "file,pipe", "-format_whitelist", ",".join(ALLOWED_DEMUXERS))
"""Input options placed before every ``-i`` that reads a user's file: local files only, real video containers only
(no HLS/DASH playlists, concat lists or network URLs)."""

REQUIRED_FILTERS = (
    "fps", "scale", "scdet", "metadata", "select",
    "aresample", "ebur128", "silencedetect", "biquad", "astats", "ametadata",
)

FFMPEG_FIX = "Install ffmpeg 6 or newer: brew install ffmpeg / apt install ffmpeg"
INPUT_FIX = "Use an MP4, MOV, MKV, WebM, AVI or MPEG-TS video file."

PROBE_TIMEOUT_SEC = 30.0
PASS_BASE_TIMEOUT_SEC = 120.0
PASS_TIMEOUT_PER_SEC = 3.0
FRAMES_TIMEOUT_SEC = 120.0

# ITU-R BS.1770 K-weighting at 48 kHz, the constants of ffmpeg's ebur128 (libavfilter/f_ebur128.c).
_K_SHELF = (
    "biquad=a0=1:a1=-1.69065929318241:a2=0.73248077421585"
    ":b0=1.53512485958697:b1=-2.69169618940638:b2=1.19839281085285"
)
_K_HIGHPASS = "biquad=a0=1:a1=-1.99004745483398:a2=0.99007225036621:b0=1:b1=-2:b2=1"

# --------------------------------------------------------------------------------------------------------------------
# ffmpeg plumbing (private: screentime_core.media is owned by another module and is not imported here)


def _which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ScreentimeError("ffmpeg_missing", f"{name} was not found on PATH.", fix=FFMPEG_FIX)
    return path


def _run(argv: list[str], timeout: float, *, what: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    """Runs one ffmpeg/ffprobe argv (never a shell). A command that outlives `timeout` is killed: ``timeout``."""
    try:
        return subprocess.run(
            argv, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, check=False, cwd=cwd,
            text=True, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise ScreentimeError(
            "timeout", f"{what} took longer than {timeout:.0f} s and was stopped.", fix="Try a shorter clip.",
            retryable=True,
        ) from None
    except FileNotFoundError:
        raise ScreentimeError("ffmpeg_missing", f"{Path(argv[0]).name} could not be started.", fix=FFMPEG_FIX) from None


def _tail(stderr: str, src: Optional[Path] = None, lines: int = 2) -> str:
    text = (stderr or "").strip()
    if src is not None:
        text = text.replace(_input_url(src), src.name).replace(str(src), src.name)
    return " | ".join(text.splitlines()[-lines:]) if text else "no error output"


def _input_url(src: Path) -> str:
    """An explicit file: URL, so no path is ever read as another protocol or as an option."""
    return "file:" + str(src)


@dataclass(frozen=True)
class _Caps:
    version: str
    filters: frozenset[str]
    encoders: frozenset[str]
    fps_mode: bool  # -fps_mode (ffmpeg 5.1+); older builds take -vsync


@functools.lru_cache(maxsize=8)
def _caps(ffmpeg: str) -> _Caps:
    """What this ffmpeg build offers, read once per binary from -version, -filters, -encoders and -h full."""
    version_out = _run([ffmpeg, "-hide_banner", "-version"], PROBE_TIMEOUT_SEC, what="ffmpeg -version").stdout
    m = re.search(r"ffmpeg version (\S+)", version_out)
    filters = set()
    for line in _run([ffmpeg, "-hide_banner", "-filters"], PROBE_TIMEOUT_SEC, what="ffmpeg -filters").stdout.splitlines():
        fm = re.match(r"^\s*[A-Z.|]{2,3}\s+(\S+)\s+\S*->\S*", line)
        if fm:
            filters.add(fm.group(1))
    encoders = set()
    listing = False
    for raw in _run([ffmpeg, "-hide_banner", "-encoders"], PROBE_TIMEOUT_SEC, what="ffmpeg -encoders").stdout.splitlines():
        line = raw.strip()
        if line.startswith("------"):
            listing = True
            continue
        parts = line.split(None, 2)
        if listing and len(parts) >= 2 and len(parts[0]) == 6:
            encoders.add(parts[1])
    help_full = _run([ffmpeg, "-hide_banner", "-h", "full"], PROBE_TIMEOUT_SEC, what="ffmpeg -h full").stdout
    return _Caps(
        version=m.group(1) if m else "unknown",
        filters=frozenset(filters),
        encoders=frozenset(encoders),
        fps_mode=re.search(r"^-fps_mode\b", help_full, re.M) is not None,
    )


def _float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ratio(value: Any) -> Optional[float]:
    if not isinstance(value, str) or ":" not in value:
        return None
    num, _, den = value.partition(":")
    n, d = _float(num), _float(den)
    return n / d if n and d else None


def _rotation(stream: Mapping[str, Any]) -> int:
    for side in stream.get("side_data_list") or []:
        if isinstance(side, Mapping) and "rotation" in side:
            r = _float(side.get("rotation"))
            if r is not None:
                return int(round(r)) % 360
    r = _float((stream.get("tags") or {}).get("rotate"))
    return int(round(r)) % 360 if r is not None else 0


def _display_size(stream: Mapping[str, Any]) -> tuple[int, int]:
    """Width and height as a player shows the video: sample aspect ratio applied, 90/270 degree rotation swapped."""
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    sar = _ratio(stream.get("sample_aspect_ratio"))
    if sar and abs(sar - 1.0) > 1e-6:
        w = int(round(w * sar))
    if _rotation(stream) % 180 == 90:
        w, h = h, w
    return w, h


def check_container(format_name: str) -> None:
    """Raises ``bad_input`` unless ffprobe's format_name is MP4/MOV, Matroska/WebM, AVI or MPEG-TS."""
    parts = [p for p in (format_name or "").split(",") if p]
    if not parts or any(p not in ALLOWED_DEMUXERS for p in parts):
        raise ScreentimeError(
            "bad_input",
            f"This file is not a video container Teardown reads (ffprobe says {format_name or 'unknown'!s}).",
            fix=INPUT_FIX,
        )


def _probe(ffprobe: str, src: Path) -> dict[str, Any]:
    argv = [ffprobe, "-v", "error", *SAFE_INPUT_ARGS, "-show_format", "-show_streams", "-of", "json",
            "-i", _input_url(src)]
    res = _run(argv, PROBE_TIMEOUT_SEC, what="ffprobe")
    if res.returncode != 0:
        raise ScreentimeError(
            "bad_input", f"{src.name} could not be read as a video ({_tail(res.stderr, src)}).", fix=INPUT_FIX
        )
    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        raise ScreentimeError("bad_input", f"ffprobe gave unreadable output for {src.name}.", fix=INPUT_FIX) from None
    check_container(str((data.get("format") or {}).get("format_name") or ""))
    return data


def _streams(data: Mapping[str, Any], src: Path) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise ScreentimeError("bad_input", f"{src.name} has no video stream.", fix=INPUT_FIX)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return video, audio


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)[:80] or "run"


def analysis_copy_path(report: TeardownReport, workdir: PathLike) -> Path:
    """Where analyze() wrote the report's MJPEG analysis copy when it was given `workdir`."""
    return Path(workdir) / f"{_safe_name(report.run_id)}.analysis.mkv"


# --------------------------------------------------------------------------------------------------------------------
# Parsing ffmpeg output

_FRAME_RE = re.compile(r"^frame:\s*(\d+)\s+pts:\s*(\S+)\s+pts_time:\s*(\S+)")
_SIL_START_RE = re.compile(r"silencedetect[^\]]*\]\s*silence_start:\s*(\S+)")
_SIL_END_RE = re.compile(r"silencedetect[^\]]*\]\s*silence_end:\s*(\S+)\s*\|\s*silence_duration:\s*(\S+)")
_INTEGRATED_RE = re.compile(r"Integrated loudness:\s*I:\s*(\S+)\s*LUFS")
_LRA_RE = re.compile(r"Loudness range:\s*LRA:\s*(\S+)\s*LU\b")


def parse_metadata_print(text: str) -> list[dict[str, Any]]:
    """Frames from ffmpeg's metadata/ametadata ``mode=print`` output: a "frame:N pts:P pts_time:T" line, then one
    key=value line per metadata entry. Each frame becomes {"n": N, "t": T or None, key: value, ...}."""
    frames: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _FRAME_RE.match(line)
        if m:
            cur = {"n": int(m.group(1)), "t": _float(m.group(3))}
            frames.append(cur)
        elif cur is not None and "=" in line:
            key, _, value = line.partition("=")
            cur[key.strip()] = value.strip()
    return frames


def merge_cuts(raw_cuts: Sequence[float], duration: float, min_shot_sec: float) -> tuple[list[float], int]:
    """Keeps a cut only when it is at least `min_shot_sec` after the previous kept cut (or the start), and drops a
    last cut that would leave a final shot shorter than that. Returns (kept cuts, number merged)."""
    eps = 1e-6
    kept: list[float] = []
    last = 0.0
    merged = 0
    for t in sorted(raw_cuts):
        if t > eps and t - last >= min_shot_sec - eps:
            kept.append(t)
            last = t
        else:
            merged += 1
    while kept and duration - kept[-1] < min_shot_sec - eps:
        kept.pop()
        merged += 1
    return kept, merged


def _silences(stderr: str, audio_end: Optional[float]) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    start: Optional[float] = None
    for line in stderr.splitlines():
        m = _SIL_START_RE.search(line)
        if m:
            start = _float(m.group(1))
            continue
        m = _SIL_END_RE.search(line)
        if m:
            end, dur = _float(m.group(1)), _float(m.group(2))
            if end is None:
                continue
            begin = start if start is not None else (end - dur if dur is not None else None)
            if begin is not None:
                found.append((max(0.0, begin), end))
            start = None
    if start is not None and audio_end is not None and audio_end > start:
        found.append((max(0.0, start), audio_end))
    return found


def _block_energy(block: Mapping[str, Any]) -> float:
    """Sum over channels of the K-weighted mean square of one 100 ms block (astats RMS_level is in dBFS)."""
    total = 0.0
    for key, value in block.items():
        if isinstance(key, str) and key.startswith("lavfi.astats.") and key.endswith(".RMS_level"):
            level = _float(value)
            if level is not None:
                total += 10.0 ** (level / 10.0)
    return total


def _lufs(energy: float) -> Optional[float]:
    return -0.691 + 10.0 * math.log10(energy) if energy > 0 else None


def block_loudness(blocks: Sequence[Mapping[str, Any]]) -> list[tuple[float, Optional[float], Optional[float]]]:
    """Per 100 ms block: (end time, short-term loudness from the K-weighted blocks over the last 3 s or the audio so
    far, ebur128's own short-term value once its 3 s window is full, else None)."""
    window = int(round(SHORT_TERM_WINDOW_SEC / BLOCK_SEC))
    cum = [0.0]
    for b in blocks:
        cum.append(cum[-1] + _block_energy(b))
    out: list[tuple[float, Optional[float], Optional[float]]] = []
    for k, b in enumerate(blocks):
        t0 = b.get("t")
        t_end = (t0 if isinstance(t0, float) else k * BLOCK_SEC) + BLOCK_SEC
        lo = max(0, k + 1 - window)
        own = _lufs((cum[k + 1] - cum[lo]) / (k + 1 - lo))
        ff = _float(b.get("lavfi.r128.S")) if k + 1 >= window else None
        out.append((t_end, own, ff))
    return out


def loudness_curve(blocks: Sequence[Mapping[str, Any]], step: float = LOUDNESS_STEP_SEC) -> list[tuple[float, float]]:
    """(time, short-term LUFS) every `step` seconds, floored at -70 LUFS. Uses ebur128's value where its window is
    full and the K-weighted blocks before that."""
    points: dict[float, float] = {}
    for t_end, own, ff in block_loudness(blocks):
        grid = round(t_end / step) * step
        if grid <= 0 or abs(t_end - grid) > BLOCK_SEC / 2 + 1e-9:
            continue
        value = ff if ff is not None else own
        value = LOUDNESS_FLOOR_LUFS if value is None else max(value, LOUDNESS_FLOOR_LUFS)
        points.setdefault(round(grid, 3), round(value, 1))
    return sorted(points.items())


# --------------------------------------------------------------------------------------------------------------------
# Analysis


@dataclass
class _Pass:
    frames: list[dict[str, Any]]
    blocks: Optional[list[dict[str, Any]]]  # None = audio not measured
    stderr: str
    copy_written: bool


def _pass_argv(
    ffmpeg: str, caps: _Caps, src: Path, tag: str, v_index: int, a_index: Optional[int], copy_path: Optional[Path],
    *, fps: float, threshold: float, silence_db: float, silence_d: float,
) -> list[str]:
    graph = [
        (
            f"[0:{v_index}]fps={fps:g},scale=-2:{ANALYSIS_HEIGHT}:out_range=tv,format=yuv420p,"
            f"scdet=threshold={threshold:g},metadata=mode=print:file={tag}.scd.txt[v]"
        )
    ]
    if a_index is not None:
        graph.append(
            f"[0:{a_index}]aresample=48000,ebur128=metadata=1,silencedetect=noise={silence_db:g}dB:d={silence_d:g},"
            f"{_K_SHELF},{_K_HIGHPASS},astats=metadata=1:reset=1:measure_perchannel=RMS_level:measure_overall=none,"
            f"ametadata=mode=print:file={tag}.audio.txt[a]"
        )
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-y",
            *SAFE_INPUT_ARGS, "-i", _input_url(src), "-filter_complex", ";".join(graph)]
    if copy_path is not None:
        argv += ["-map", "[v]", "-c:v", "mjpeg", "-q:v", "5", "-an", "-sn", "-dn", str(copy_path)]
        if a_index is not None:
            argv += ["-map", "[a]", "-f", "null", "-"]
    else:
        argv += ["-map", "[v]"] + (["-map", "[a]"] if a_index is not None else []) + ["-f", "null", "-"]
    return argv


def _measure(
    ffmpeg: str, caps: _Caps, src: Path, work: Path, tag: str, v_index: int, a_index: Optional[int], *,
    fps: float, threshold: float, silence_db: float, silence_d: float, timeout: float,
) -> _Pass:
    """The single ffmpeg pass. If it fails it is retried without the analysis copy, then without audio."""
    scd_file, audio_file = work / f"{tag}.scd.txt", work / f"{tag}.audio.txt"
    copy_path = work / f"{tag}.analysis.mkv"
    with_copy = "mjpeg" in caps.encoders
    plans: list[tuple[Optional[int], bool]] = [(a_index, with_copy)]
    if with_copy:
        plans.append((a_index, False))
    if a_index is not None:
        plans.append((None, False))
    last = ""
    for audio_idx, copy in plans:
        for f in (scd_file, audio_file, copy_path):
            f.unlink(missing_ok=True)
        argv = _pass_argv(
            ffmpeg, caps, src, tag, v_index, audio_idx, copy_path if copy else None,
            fps=fps, threshold=threshold, silence_db=silence_db, silence_d=silence_d,
        )
        res = _run(argv, timeout, what="The ffmpeg analysis pass", cwd=work)
        if res.returncode == 0 and scd_file.exists():
            frames = parse_metadata_print(scd_file.read_text(encoding="utf-8", errors="replace"))
            if frames:
                blocks = None
                if audio_idx is not None and audio_file.exists():
                    blocks = parse_metadata_print(audio_file.read_text(encoding="utf-8", errors="replace"))
                return _Pass(frames=frames, blocks=blocks, stderr=res.stderr, copy_written=copy and copy_path.exists())
        last = _tail(res.stderr, src)
    raise ScreentimeError("bad_input", f"ffmpeg could not decode the video in {src.name} ({last}).", fix=INPUT_FIX)


def _check_params(**params: Any) -> dict[str, float]:
    """Each threshold as a float, or ``bad_input`` naming every one out of range (the tool layer passes user input)."""
    ranges = {
        "analysis_fps": (1.0, 60.0, "between 1 and 60"),
        "scdet_threshold": (0.0, 100.0, "between 0 and 100"),
        "min_shot_sec": (0.0, 10.0, "between 0 and 10"),
        "silence_db": (-120.0, -0.001, "below 0 dB (for example -40)"),
        "silence_min_sec": (0.001, 60.0, "above 0 and at most 60"),
        "first_sound_min_sec": (0.001, 60.0, "above 0 and at most 60"),
    }
    values: dict[str, float] = {}
    problems = []
    for name, (lo, hi, text) in ranges.items():
        v = _float(params[name]) if not isinstance(params[name], bool) else None
        if v is None or not lo <= v <= hi:
            problems.append(f"{name} must be {text}")
        else:
            values[name] = v
    if problems:
        raise ScreentimeError("bad_input", "; ".join(problems) + ".")
    return values


def analyze(
    video_path: PathLike,
    workdir: PathLike,
    *,
    analysis_fps: float = 15,
    scdet_threshold: float = 10.0,
    min_shot_sec: float = 0.2,
    silence_db: float = -40.0,
    silence_min_sec: float = 0.5,
    first_sound_min_sec: float = 0.2,
    product_first_seen_sec: Optional[float] = None,
    brand_first_said_sec: Optional[float] = None,
    run_id: Optional[str] = None,
    max_duration_sec: Optional[float] = MAX_DURATION_SEC,
) -> TeardownReport:
    """Measures one video shot by shot: cuts, shots, pixel change, loudness, first sound and silences.

    Cuts are frames where scdet's score reaches `scdet_threshold`; a cut less than `min_shot_sec` after the previous
    one (a flash frame) is merged into it. Pixel change is the mean scdet mafd (mean absolute difference to the
    previous frame, 0-100) over a shot's frames, skipping the first two frames at every detected cut; it measures how
    much the picture changes, not camera motion. With no audio stream every loudness field is None, never 0.

    `product_first_seen_sec` and `brand_first_said_sec` come from a Measure run of the same file when one exists.
    Files written to `workdir`: ``<run_id>.analysis.mkv`` (MJPEG, analysis_fps, 360 px high), ``<run_id>.scd.txt``,
    ``<run_id>.audio.txt`` and ``<run_id>.ffmpeg.log``.
    """
    checked = _check_params(
        analysis_fps=analysis_fps, scdet_threshold=scdet_threshold, min_shot_sec=min_shot_sec, silence_db=silence_db,
        silence_min_sec=silence_min_sec, first_sound_min_sec=first_sound_min_sec,
    )
    fps = checked["analysis_fps"]
    scdet_threshold, min_shot_sec, silence_db = (
        checked["scdet_threshold"], checked["min_shot_sec"], checked["silence_db"]
    )
    silence_min_sec, first_sound_min_sec = checked["silence_min_sec"], checked["first_sound_min_sec"]
    given = Path(video_path).expanduser()
    if not given.is_file():
        raise ScreentimeError("bad_input", f"{given.name or given} is not a file that can be read.", fix=INPUT_FIX)
    src = given.resolve()
    work = Path(workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    ffmpeg, ffprobe = _which("ffmpeg"), _which("ffprobe")
    caps = _caps(ffmpeg)
    missing = [f for f in REQUIRED_FILTERS if f not in caps.filters]
    if missing:
        raise ScreentimeError(
            "ffmpeg_missing", f"This ffmpeg ({caps.version}) lacks the filters {', '.join(missing)}.", fix=FFMPEG_FIX
        )

    data = _probe(ffprobe, src)
    video, audio = _streams(data, src)
    fmt = data.get("format") or {}
    width, height = _display_size(video)
    probed = _float(video.get("duration")) or _float(fmt.get("duration"))
    if max_duration_sec and probed and probed > max_duration_sec:
        raise ScreentimeError(
            "too_long", f"{src.name} is {probed:.0f} s long; Teardown takes up to {max_duration_sec:.0f} s.",
            fix="Trim the video, or raise the limit.",
        )
    run_id = run_id or _new_run_id("td")
    tag = _safe_name(run_id)
    timeout = PASS_BASE_TIMEOUT_SEC + PASS_TIMEOUT_PER_SEC * (probed or MAX_DURATION_SEC)
    measured = _measure(
        ffmpeg, caps, src, work, tag, int(video.get("index", 0)),
        int(audio["index"]) if audio is not None and audio.get("index") is not None else None,
        fps=fps, threshold=scdet_threshold, silence_db=silence_db,
        silence_d=min(silence_min_sec, first_sound_min_sec), timeout=timeout,
    )
    (work / f"{tag}.ffmpeg.log").write_text(measured.stderr, encoding="utf-8")
    notes: list[str] = []

    # Shots ------------------------------------------------------------------------------------------------------
    frames = measured.frames
    times = [f["t"] if f.get("t") is not None else i / fps for i, f in enumerate(frames)]
    duration = round(probed if probed else len(frames) / fps, 3)
    # scdet sets lavfi.scd.time where score >= threshold. The printed score is rounded to 3 decimals, so the fallback
    # (for a build that does not print the time) asks for strictly more, never counting a score rounded up to it.
    raw_idx = [
        i for i, f in enumerate(frames)
        if i > 0 and ("lavfi.scd.time" in f or (_float(f.get("lavfi.scd.score")) or 0.0) > scdet_threshold)
    ]
    kept, merged = merge_cuts([times[i] for i in raw_idx], duration, min_shot_sec)
    cuts = [round(c, 3) for c in kept]
    if merged:
        notes.append(
            f"{merged} detected cut{'s' if merged != 1 else ''} less than {min_shot_sec:g} s from the previous cut "
            f"(or the start or end) {'were' if merged != 1 else 'was'} merged, as flash frames."
        )
    skip = {0}
    for i in raw_idx:
        skip.update(range(i, i + PIXEL_CHANGE_SKIP_FRAMES))
    per_shot: list[list[float]] = [[] for _ in range(len(cuts) + 1)]
    for i, f in enumerate(frames):
        mafd = _float(f.get("lavfi.scd.mafd"))
        if i in skip or mafd is None:
            continue
        per_shot[bisect.bisect_right(kept, times[i] + 1e-6)].append(mafd)
    bounds = [0.0, *cuts, duration]
    shots = [
        Shot(
            index=k,
            start_sec=round(bounds[k], 3),
            end_sec=round(bounds[k + 1], 3),
            duration_sec=round(bounds[k + 1] - bounds[k], 3),
            pixel_change=round(statistics.fmean(per_shot[k]), 3) if per_shot[k] else None,
        )
        for k in range(len(bounds) - 1)
    ]
    windows = [0] * max(1, math.ceil(duration / CUTS_WINDOW_SEC - 1e-9))
    for c in cuts:
        windows[min(len(windows) - 1, int((c + 1e-6) // CUTS_WINDOW_SEC))] += 1
    longest = max(shots, key=lambda s: (s.duration_sec, -s.start_sec))

    # Audio ------------------------------------------------------------------------------------------------------
    integrated: Optional[float] = None
    lra: Optional[float] = None
    curve: list[tuple[float, float]] = []
    first_sound: Optional[float] = None
    silences: list[Segment] = []
    if audio is None:
        notes.append("No audio stream: loudness, first sound and silences are not measured.")
    elif measured.blocks is None:
        notes.append("The audio track could not be decoded: loudness, first sound and silences are not measured.")
    else:
        a_start = max(0.0, _float(audio.get("start_time")) or 0.0)
        a_dur = _float(audio.get("duration"))
        last_block = measured.blocks[-1].get("t") if measured.blocks else None
        a_end = a_start + a_dur if a_dur else (last_block + BLOCK_SEC if isinstance(last_block, float) else duration)
        found = _silences(measured.stderr, a_end)
        i_match = _INTEGRATED_RE.findall(measured.stderr)
        lra_match = _LRA_RE.findall(measured.stderr)
        integrated = _float(i_match[-1]) if i_match else None
        lra = _float(lra_match[-1]) if lra_match else None
        lead = next((s for s in found if s[0] <= a_start + START_TOLERANCE_SEC), None)
        if integrated is not None and integrated <= LOUDNESS_FLOOR_LUFS + 0.05:
            notes.append(
                f"The audio track is silent (under {LOUDNESS_FLOOR_LUFS:g} LUFS): loudness and first sound are "
                "not measured."
            )
            integrated, lra = None, None
        else:
            if integrated is None:
                notes.append("ebur128 printed no summary: integrated loudness and loudness range are not measured.")
            curve = loudness_curve(measured.blocks)
            if a_end - a_start < LRA_MIN_AUDIO_SEC:
                lra = None
            if lead is not None and lead[1] >= a_end - START_TOLERANCE_SEC:
                notes.append(f"No sound above {silence_db:g} dB in the audio track.")
            elif lead is not None and lead[1] >= first_sound_min_sec:
                first_sound = round(lead[1], 3)
            elif a_start >= first_sound_min_sec:
                first_sound = round(a_start, 3)
            else:
                first_sound = 0.0
        silences = [Segment(round(s, 3), round(e, 3)) for s, e in found if e - s >= silence_min_sec - 1e-6]

    thresholds: dict[str, Any] = {
        "analysis_fps": int(fps) if fps.is_integer() else fps,
        "analysis_height_px": ANALYSIS_HEIGHT,
        "scdet_threshold": scdet_threshold,
        "min_shot_sec": min_shot_sec,
        "pixel_change_skip_frames": PIXEL_CHANGE_SKIP_FRAMES,
        "silence_db": silence_db,
        "silence_min_sec": silence_min_sec,
        "first_sound_min_sec": first_sound_min_sec,
        "loudness_step_sec": LOUDNESS_STEP_SEC,
        "short_term_window_sec": SHORT_TERM_WINDOW_SEC,
        "lra_min_audio_sec": LRA_MIN_AUDIO_SEC,
        "loudness_floor_lufs": LOUDNESS_FLOOR_LUFS,
        "cuts_window_sec": CUTS_WINDOW_SEC,
        "early_window_sec": EARLY_WINDOW_SEC,
        "ffmpeg_version": caps.version,
    }
    return TeardownReport(
        run_id=run_id,
        mode="teardown",
        video_path=str(src),
        video_sha256=_sha256(src),
        info=VideoInfo(
            width=width, height=height, frame_count=len(frames), duration_sec=duration, fps=fps,
            has_audio=audio is not None,
        ),
        cuts_sec=cuts,
        shots=shots,
        first_cut_sec=cuts[0] if cuts else None,
        cuts_first_3s=sum(1 for c in cuts if c < EARLY_WINDOW_SEC - 1e-6),
        cuts_per_10s=windows,
        median_shot_sec=round(statistics.median(s.duration_sec for s in shots), 3),
        longest_shot_sec=longest.duration_sec,
        longest_shot_start_sec=longest.start_sec,
        integrated_lufs=round(integrated, 1) if integrated is not None else None,
        loudness_range_lu=round(lra, 1) if lra is not None else None,
        loudness_curve=curve,
        first_sound_sec=first_sound,
        silences=silences,
        thresholds=thresholds,
        product_first_seen_sec=_float(product_first_seen_sec),
        brand_first_said_sec=_float(brand_first_said_sec),
        notes=notes,
    )


# --------------------------------------------------------------------------------------------------------------------
# Shot frames for the card


def extract_shot_frames(
    report: TeardownReport, workdir: PathLike, *, height: int = ANALYSIS_HEIGHT
) -> list[Optional[Path]]:
    """One JPEG per shot, the frame in the middle of the shot on the analysis frame grid, written to `workdir`.

    Reads the analysis copy when `workdir` holds it, else the original video. Returns paths in shot order, None where
    no frame could be extracted (for example when the video has moved since the analysis).
    """
    work = Path(workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not report.shots:
        return []
    fps = float(report.thresholds.get("analysis_fps") or report.info.fps or 15)
    last = report.info.frame_count - 1 if report.info.frame_count else None
    mids = []
    for s in report.shots:
        n = int(((s.start_sec + s.end_sec) / 2.0) * fps + 1e-6)
        lo, hi = int(math.ceil(s.start_sec * fps - 1e-6)), int(math.ceil(s.end_sec * fps - 1e-6)) - 1
        n = max(lo, min(n, hi)) if hi >= lo else lo
        mids.append(max(0, min(n, last)) if last is not None else max(0, n))
    wanted = sorted(set(mids))
    copy = analysis_copy_path(report, work)
    src = copy if copy.is_file() else Path(report.video_path)
    if not src.is_file():
        return [None] * len(report.shots)
    try:
        ffmpeg = _which("ffmpeg")
        caps = _caps(ffmpeg)
    except ScreentimeError:
        return [None] * len(report.shots)
    stem = f"{_safe_name(report.run_id)}.shot"
    for old in work.glob(f"{stem}_*.jpg"):
        old.unlink(missing_ok=True)
    select = "select='" + "+".join(f"eq(n,{n})" for n in wanted) + "'"
    chain = select if src == copy else f"fps={fps:g},scale=-2:{int(height)}:out_range=tv,format=yuv420p,{select}"
    pattern = str(work).replace("%", "%%") + f"/{stem}_%04d.jpg"
    sync = ["-fps_mode", "passthrough"] if caps.fps_mode else ["-vsync", "0"]
    encoder = ["-c:v", "mjpeg", "-q:v", "3"] if "mjpeg" in caps.encoders else []
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *SAFE_INPUT_ARGS, "-i",
            _input_url(src.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", chain, *sync,
            "-frames:v", str(len(wanted)), *encoder, pattern]
    try:
        res = _run(argv, FRAMES_TIMEOUT_SEC, what="ffmpeg frame extraction")
    except ScreentimeError:
        return [None] * len(report.shots)
    if res.returncode != 0:
        return [None] * len(report.shots)
    by_frame = {}
    for i, n in enumerate(wanted):
        p = work / f"{stem}_{i + 1:04d}.jpg"
        if p.is_file():
            by_frame[n] = p
    return [by_frame.get(n) for n in mids]


# --------------------------------------------------------------------------------------------------------------------
# Compare

_UNITS = {
    "Duration": " s",
    "First cut": " s",
    "Median shot": " s",
    "First sound": " s",
    "Product first seen": " s",
    "Brand first said": " s",
    "Integrated loudness": " LUFS",
}


def format_value(metric: str, value: Any) -> str:
    """A compare row value as display text: 0.1 s for times, 1 decimal LUFS, counts as integers."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.1f}{_UNITS.get(metric, '')}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value("", v) for v in value) if value else "none"
    return str(value)


def _r1(x: Optional[float]) -> Optional[float]:
    """Rounded to 0.1, with -0.0 turned into 0.0."""
    return None if x is None else round(float(x), 1) + 0.0


def _one_sided(y: Any, t: Any) -> Optional[str]:
    if y is None and t is None:
        return "same"
    if y is None:
        return "none in yours"
    if t is None:
        return "none in reference"
    return None


def _diff_amount(y: Optional[float], t: Optional[float], unit: str, more: str, less: str) -> str:
    side = _one_sided(y, t)
    if side:
        return side
    d = round(y - t, 1)  # type: ignore[operator]
    if abs(d) < 0.05:
        return "same"
    return f"{abs(d):.1f}{unit} {more if d > 0 else less}"


def _diff_count(y: int, t: int) -> str:
    d = y - t
    return "same" if d == 0 else f"{abs(d)} {'more' if d > 0 else 'fewer'}"


def _diff_windows(y: Sequence[int], t: Sequence[int]) -> str:
    if list(y) == list(t):
        return "same"
    n = min(len(y), len(t))
    deltas = [a - b for a, b in zip(y[:n], t[:n], strict=True)]
    span = f" (first {n * CUTS_WINDOW_SEC:g} s)" if len(y) != len(t) else ""
    if all(d == 0 for d in deltas):
        return "same" + span
    if n == 1:  # one window: plain words, as every other row reads
        return _diff_count(y[0], t[0]) + span
    words = []
    for i, d in enumerate(deltas):
        if d:
            lo, hi = i * CUTS_WINDOW_SEC, (i + 1) * CUTS_WINDOW_SEC
            words.append(f"{abs(d)} {'more' if d > 0 else 'fewer'} in {lo:g}-{hi:g} s")
    return ", ".join(words) + span


def mean_pixel_change(report: TeardownReport) -> Optional[float]:
    """Shot pixel change averaged over the video, weighted by shot length."""
    pairs = [(s.pixel_change, s.duration_sec) for s in report.shots if s.pixel_change is not None and s.duration_sec > 0]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total > 0 else None


def longest_shot_text(r: TeardownReport) -> Optional[str]:
    """"13.0 s at 0.0 s": the longest shot and where it starts."""
    if r.longest_shot_sec is None:
        return None
    start = f" at {r.longest_shot_start_sec:.1f} s" if r.longest_shot_start_sec is not None else ""
    return f"{r.longest_shot_sec:.1f} s{start}"


def _diff_longest(y: TeardownReport, t: TeardownReport) -> str:
    side = _one_sided(y.longest_shot_sec, t.longest_shot_sec)
    if side:
        return side
    length = _diff_amount(_r1(y.longest_shot_sec), _r1(t.longest_shot_sec), " s", "longer", "shorter")
    start = _diff_amount(_r1(y.longest_shot_start_sec), _r1(t.longest_shot_start_sec), " s", "later", "earlier")
    parts = [length] if length != "same" else []
    if start != "same":
        parts.append(f"starts {start}")
    return ", ".join(parts) or "same"


def compare(yours: TeardownReport, theirs: TeardownReport, run_id: Optional[str] = None) -> CompareReport:
    """Yours against a reference, metric by metric. Values are rounded the way they are shown (0.1 s, 1 decimal LUFS)
    and every difference reads from your side: "1.5 s later", "3 fewer", "same". No score, no verdict."""
    y, t = yours, theirs

    def amount(metric: str, a: Optional[float], b: Optional[float], unit: str, more: str, less: str) -> CompareRow:
        a1, b1 = _r1(a), _r1(b)
        return CompareRow(metric=metric, yours=a1, theirs=b1, diff=_diff_amount(a1, b1, unit, more, less))

    rows = [
        amount("Duration", y.info.duration_sec, t.info.duration_sec, " s", "longer", "shorter"),
        amount("First cut", y.first_cut_sec, t.first_cut_sec, " s", "later", "earlier"),
        CompareRow("Cuts in first 3 s", y.cuts_first_3s, t.cuts_first_3s, _diff_count(y.cuts_first_3s, t.cuts_first_3s)),
        CompareRow("Shots", len(y.shots), len(t.shots), _diff_count(len(y.shots), len(t.shots))),
        amount("Median shot", y.median_shot_sec, t.median_shot_sec, " s", "longer", "shorter"),
        CompareRow("Longest shot", longest_shot_text(y), longest_shot_text(t), _diff_longest(y, t)),
        CompareRow("Cuts per 10 s", list(y.cuts_per_10s), list(t.cuts_per_10s), _diff_windows(y.cuts_per_10s, t.cuts_per_10s)),
        amount("Pixel change (mean)", mean_pixel_change(y), mean_pixel_change(t), "", "higher", "lower"),
        amount("Integrated loudness", y.integrated_lufs, t.integrated_lufs, " LU", "louder", "quieter"),
        amount("First sound", y.first_sound_sec, t.first_sound_sec, " s", "later", "earlier"),
    ]
    if y.product_first_seen_sec is not None or t.product_first_seen_sec is not None:
        rows.append(amount("Product first seen", y.product_first_seen_sec, t.product_first_seen_sec, " s", "later", "earlier"))
    if y.brand_first_said_sec is not None or t.brand_first_said_sec is not None:
        rows.append(amount("Brand first said", y.brand_first_said_sec, t.brand_first_said_sec, " s", "later", "earlier"))
    return CompareReport(run_id=run_id or _new_run_id("cmp"), mode="compare", yours=yours, theirs=theirs, rows=rows)
