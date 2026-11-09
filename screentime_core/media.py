"""Video and audio helpers around the ffmpeg and ffprobe executables.

Every call passes an argv list to ``subprocess.run`` (never a shell) with a time limit, and raises
``ScreentimeError`` for expected failures. Port of the Screen-Time app's src/lib/media/index.ts, with two changes:
ffprobe JSON replaces parsing ``ffmpeg -i`` text, and the normalized video carries no audio (audio is read from the
source with ``extract_audio_wav``).

A normalized video is constant frame rate with timestamps regenerated as frame / fps, so a frame index divided by the
frame rate is an exact time, and ``frame_jpeg`` returns exactly the frame asked for.

Untrusted input files: every ffprobe and ffmpeg read of a user's file allows only the file and pipe protocols, and
probing accepts only real video containers (ALLOWED_FORMATS) through ffmpeg's own format whitelist, which refuses a
playlist (HLS, concat, ...) before it opens anything the playlist names. Later reads force the demuxer the probe found
(``input_args``), so a file swapped after probing is not re-detected as something else.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from .errors import ScreentimeError
from .models import FPS, Box, VideoInfo

PathLike = Union[str, "os.PathLike[str]"]

FFMPEG_FIX = "Install ffmpeg: brew install ffmpeg / apt install ffmpeg"

MAX_EDGE = 960
"""Longest edge of a normalized video in the app (src/lib/media/index.ts MAX_EDGE)."""

WAV_SAMPLE_RATE = 16000
"""Muse Voice Transcribe takes mono 16-bit PCM WAV at 16 or 24 kHz."""

ENCODER_CHAIN = ("libx264", "h264_videotoolbox", "libopenh264", "mpeg4")
"""H.264 encoders in order of preference, then MPEG-4 part 2. SAM 3.1 accepts H.264, H.265 and MPEG-4 in MP4."""

PROBE_TIMEOUT_SEC = 30.0
EXTRACT_TIMEOUT_SEC = 30.0
NORMALIZE_BASE_TIMEOUT_SEC = 120.0
"""normalize() allows this plus 2 s per second of video unless a timeout is given."""

JPEG_QSCALE = 3  # ffmpeg mjpeg -q:v, as in the app (2 is best, 31 worst)

ALLOWED_FORMATS = ("mov", "mp4", "matroska", "webm", "avi", "mpegts")
"""Demuxers a user's file may open with: the mov family (mov,mp4,m4a,3gp,3g2,mj2), matroska/webm, avi and mpegts."""

INPUT_PROTOCOLS = "file,pipe"
_FORMAT_WHITELIST = ",".join(ALLOWED_FORMATS)
_PROBED_AS = re.compile(r"Format (\S+) probed with size")


# --------------------------------------------------------------------------------------------------------------------
# Executables


def _which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ScreentimeError("ffmpeg_missing", f"{name} was not found on PATH.", fix=FFMPEG_FIX)
    return path


def ffmpeg_bin() -> str:
    """Path of the ffmpeg executable on PATH."""
    return _which("ffmpeg")


def ffprobe_bin() -> str:
    """Path of the ffprobe executable on PATH."""
    return _which("ffprobe")


def _run(argv: list[str], timeout: float, *, what: str) -> subprocess.CompletedProcess[bytes]:
    """Runs one ffmpeg or ffprobe command. A command that outlives `timeout` is killed and raises ``timeout``."""
    try:
        return subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ScreentimeError(
            "timeout", f"{what} took longer than {timeout:.0f} s and was stopped.", retryable=True
        ) from None
    except FileNotFoundError:
        raise ScreentimeError("ffmpeg_missing", f"{Path(argv[0]).name} could not be started.", fix=FFMPEG_FIX) from None


def _tail(stderr: bytes, lines: int = 4) -> str:
    text = stderr.decode("utf-8", "replace").strip()
    return " | ".join(text.splitlines()[-lines:]) if text else "no error output"


@functools.lru_cache(maxsize=8)
def _encoders_for(ffmpeg: str) -> dict[str, str]:
    result = _run([ffmpeg, "-hide_banner", "-encoders"], PROBE_TIMEOUT_SEC, what="ffmpeg -encoders")
    found: dict[str, str] = {}
    listing = False
    for raw in result.stdout.decode("utf-8", "replace").splitlines():
        line = raw.strip()
        if line.startswith("------"):
            listing = True
            continue
        if not listing or not line:
            continue
        parts = line.split(None, 2)
        # Flags column first ("V....D"), then the encoder name, then its description.
        if len(parts) >= 2 and len(parts[0]) == 6:
            found[parts[1]] = parts[2] if len(parts) == 3 else ""
    return found


def encoders() -> dict[str, str]:
    """Every encoder this ffmpeg build has, name -> description (from ``ffmpeg -hide_banner -encoders``)."""
    return dict(_encoders_for(ffmpeg_bin()))


def video_encoders() -> list[str]:
    """The encoders from ENCODER_CHAIN this ffmpeg has, in the order normalize() tries them."""
    have = encoders()
    return [name for name in ENCODER_CHAIN if name in have]


# --------------------------------------------------------------------------------------------------------------------
# Probe


def _ratio(value: Any) -> Optional[float]:
    """'30000/1001' or '16:9' to a float; None when absent, zero or unreadable."""
    if not isinstance(value, str):
        return None
    for sep in ("/", ":"):
        if sep in value:
            num, _, den = value.partition(sep)
            try:
                n, d = float(num), float(den)
            except ValueError:
                return None
            return n / d if n > 0 and d > 0 else None
    try:
        v = float(value)
    except ValueError:
        return None
    return v if v > 0 else None


def _rotation(stream: Mapping[str, Any]) -> int:
    """Display rotation in degrees from the display matrix side data, or the older 'rotate' tag."""
    for item in stream.get("side_data_list") or []:
        if isinstance(item, Mapping) and "rotation" in item:
            try:
                return int(round(float(item["rotation"])))
            except (TypeError, ValueError):
                continue
    tag = (stream.get("tags") or {}).get("rotate")
    try:
        return int(round(float(tag))) if tag is not None else 0
    except (TypeError, ValueError):
        return 0


def _not_a_video(fmt: str) -> ScreentimeError:
    return ScreentimeError("bad_input", f"That file is not a video Screen-Time can read (format {fmt}).")


def _allowed_demuxer(format_name: str) -> Optional[str]:
    """The first allowed demuxer name in ffprobe's format_name ("mov,mp4,m4a,3gp,3g2,mj2" -> "mov"), else None."""
    return next((name for name in format_name.split(",") if name in ALLOWED_FORMATS), None)


def _absolute_file(path: PathLike) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():  # also refuses FIFOs and devices, which would hang or never end
        raise ScreentimeError("bad_input", f"No video file at {p}.")
    return p


def _detected_format(path: str) -> Optional[str]:
    """Which demuxer ffmpeg's probe picked for a file the format whitelist refused (from its debug log)."""
    result = _run(
        [ffprobe_bin(), "-v", "debug", "-protocol_whitelist", INPUT_PROTOCOLS, "-format_whitelist", _FORMAT_WHITELIST,
         "-i", path],
        PROBE_TIMEOUT_SEC,
        what="ffprobe",
    )
    match = _PROBED_AS.search(result.stderr.decode("utf-8", "replace"))
    return match.group(1) if match else None


@functools.lru_cache(maxsize=256)
def _probe_cached(path: str, mtime_ns: int, size: int) -> tuple[str, str]:
    """(ffprobe JSON text, demuxer) for one version of one file; raises for anything that is not a real video."""
    result = _run(
        [ffprobe_bin(), "-v", "error", "-protocol_whitelist", INPUT_PROTOCOLS, "-format_whitelist", _FORMAT_WHITELIST,
         "-print_format", "json", "-show_streams", "-show_format", "-i", path],
        PROBE_TIMEOUT_SEC,
        what="ffprobe",
    )
    name = Path(path).name
    if result.returncode != 0:
        if b"not on whitelist" in result.stderr:
            raise _not_a_video(_detected_format(path) or "unknown")
        raise ScreentimeError("bad_input", f"Could not read {name}: {_tail(result.stderr)}")
    text = result.stdout.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ScreentimeError("bad_input", f"ffprobe gave unreadable output for {name}.") from None
    format_name = str(((data if isinstance(data, dict) else {}).get("format") or {}).get("format_name") or "")
    demuxer = _allowed_demuxer(format_name)
    if demuxer is None:
        raise _not_a_video(format_name or "unknown")
    return text, demuxer


def _probe_json(path: PathLike) -> tuple[dict[str, Any], str]:
    p = _absolute_file(path)
    st = p.stat()
    text, demuxer = _probe_cached(str(p), st.st_mtime_ns, st.st_size)
    return json.loads(text), demuxer


def input_args(path: PathLike) -> list[str]:
    """ffmpeg input arguments for a user's file, after checking it is a real video container:
    ["-protocol_whitelist", "file,pipe", "-f", <demuxer>, "-i", <absolute path>]. Cached per path and mtime."""
    p = _absolute_file(path)
    _, demuxer = _probe_json(p)
    return ["-protocol_whitelist", INPUT_PROTOCOLS, "-f", demuxer, "-i", str(p)]


def _video_stream(data: Mapping[str, Any], path: Path) -> dict[str, Any]:
    for stream in data.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        if (stream.get("disposition") or {}).get("attached_pic") == 1:
            continue  # cover art, not the video
        return stream
    raise ScreentimeError("bad_input", f"{path.name} has no video stream.")


def _count_packets(path: Path, demuxer: str) -> Optional[int]:
    result = _run(
        [ffprobe_bin(), "-v", "error", "-protocol_whitelist", INPUT_PROTOCOLS, "-f", demuxer, "-count_packets",
         "-select_streams", "V:0", "-show_entries", "stream=nb_read_packets", "-print_format", "json", "-i", str(path)],
        NORMALIZE_BASE_TIMEOUT_SEC,
        what="ffprobe -count_packets",
    )
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout.decode("utf-8", "replace")).get("streams") or []
        return int(streams[0]["nb_read_packets"])
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None


def _display_size(stream: Mapping[str, Any]) -> tuple[int, int]:
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    sar = _ratio(stream.get("sample_aspect_ratio"))
    if sar and abs(sar - 1.0) > 1e-6:
        width = max(1, int(round(width * sar)))
    if abs(_rotation(stream)) % 180 == 90:
        width, height = height, width
    return width, height


def probe(path: PathLike) -> VideoInfo:
    """Size as displayed (rotation and pixel aspect applied), frame count, duration, frame rate and audio presence.

    Raises bad_input for anything that is not a video in an allowed container (see ALLOWED_FORMATS)."""
    p = _absolute_file(path)
    data, demuxer = _probe_json(p)
    stream = _video_stream(data, p)
    width, height = _display_size(stream)
    if width <= 0 or height <= 0:
        raise ScreentimeError("bad_input", f"{p.name} reports no frame size.")
    fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate")) or 0.0
    duration = _ratio(stream.get("duration")) or _ratio((data.get("format") or {}).get("duration")) or 0.0
    try:
        frames = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0:
        frames = _count_packets(p, demuxer) or (int(round(duration * fps)) if duration and fps else 0)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams") or [])
    return VideoInfo(
        width=width,
        height=height,
        frame_count=frames,
        duration_sec=float(duration),
        fps=float(fps),
        has_audio=has_audio,
    )


# --------------------------------------------------------------------------------------------------------------------
# Normalize


def _even_by_aspect(side: int, num: int, den: int) -> int:
    """ffmpeg scale's -2: side * num / den rounded to the nearest whole number of pixel pairs (av_rescale)."""
    return 2 * ((side * num + den) // (2 * den))


def target_size(
    width: int, height: int, *, max_edge: Optional[int] = MAX_EDGE, max_height: Optional[int] = None
) -> tuple[int, int]:
    """Output size for a displayed width x height: never upscaled, both sides even.

    max_edge reproduces the app's scaleFilter (the longest edge becomes 2*trunc(min(max_edge, longest)/2) and the other
    side follows the aspect ratio, rounded as ffmpeg's -2 rounds). max_height then caps the height the same way.
    """
    if width <= 0 or height <= 0:
        raise ScreentimeError("bad_input", f"Invalid frame size {width}x{height}.")
    if max_edge is not None and max_edge < 2:
        raise ValueError("max_edge must be at least 2")
    if max_height is not None and max_height < 2:
        raise ValueError("max_height must be at least 2")
    if max_edge is not None:
        if width >= height:
            out_w = 2 * (min(max_edge, width) // 2)
            out_h = _even_by_aspect(out_w, height, width)
        else:
            out_h = 2 * (min(max_edge, height) // 2)
            out_w = _even_by_aspect(out_h, width, height)
    else:
        out_w, out_h = 2 * (width // 2), 2 * (height // 2)
    if max_height is not None and out_h > max_height:
        out_h = 2 * (max_height // 2)
        out_w = _even_by_aspect(out_h, width, height)
    return max(2, out_w), max(2, out_h)


def _encoder_args(name: str, width: int, height: int, fps: float) -> list[str]:
    # About 0.2 bits per pixel per frame for the encoders that take a bitrate rather than a quality.
    bitrate = f"{max(500, int(width * height * fps * 0.2 / 1000))}k"
    if name == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    if name == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-allow_sw", "1"]
    if name == "libopenh264":
        return ["-c:v", "libopenh264", "-b:v", bitrate]
    if name == "mpeg4":
        return ["-c:v", "mpeg4", "-q:v", "3"]
    raise ValueError(f"unsupported encoder {name!r}")


def _fps_text(fps: float) -> str:
    return str(int(fps)) if float(fps).is_integer() else repr(float(fps))


def _trim_args(start_sec: Optional[float], duration_sec: Optional[float]) -> tuple[list[str], list[str]]:
    """(-ss before -i for an accurate input seek, -t after -i)."""
    before: list[str] = []
    after: list[str] = []
    if start_sec is not None:
        if start_sec < 0:
            raise ScreentimeError("bad_input", "start_sec cannot be negative.")
        if start_sec > 0:
            before = ["-ss", f"{start_sec:.6f}"]
    if duration_sec is not None:
        if duration_sec <= 0:
            raise ScreentimeError("bad_input", "duration_sec must be positive.")
        after = ["-t", f"{duration_sec:.6f}"]
    return before, after


def normalize(
    src: PathLike,
    dst: PathLike,
    fps: float = FPS,
    max_height: Optional[int] = None,
    max_edge: Optional[int] = MAX_EDGE,
    *,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    encoder: Optional[str] = None,
    timeout: Optional[float] = None,
) -> VideoInfo:
    """Re-encode `src` to an MP4 SAM 3.1 accepts and frame indices map exactly to time.

    Exactly `fps` frames per second (fps filter, timestamps regenerated as frame / fps), autorotated, square pixels,
    even width and height (scaled per ``target_size``), yuv420p, no audio, no metadata, faststart. Encoders are tried in
    ENCODER_CHAIN order (or only `encoder` when given). start_sec/duration_sec cut a window first (accurate seek).
    """
    if not fps or fps <= 0:
        raise ValueError("fps must be positive")
    source, out = Path(src), Path(dst)
    info = probe(source)
    if info.frame_count <= 0:
        raise ScreentimeError("bad_input", f"{source.name} has no video frames.")
    out_w, out_h = target_size(info.width, info.height, max_edge=max_edge, max_height=max_height)
    seek, limit = _trim_args(start_sec, duration_sec)
    span = info.duration_sec if duration_sec is None else min(duration_sec, info.duration_sec or duration_sec)
    limit_sec = timeout if timeout is not None else NORMALIZE_BASE_TIMEOUT_SEC + 2.0 * max(0.0, span)

    rate = _fps_text(fps)
    vf = f"fps={rate},setpts=N/({rate}*TB),scale={out_w}:{out_h},setsar=1,format=yuv420p"
    names = [encoder] if encoder else video_encoders()
    if not names:
        raise ScreentimeError(
            "ffmpeg_missing", "This ffmpeg has no H.264 or MPEG-4 encoder.", fix="Install a full ffmpeg build."
        )
    available = encoders()
    source_args = input_args(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial")
    failures: list[str] = []
    for name in names:
        if name not in available:
            failures.append(f"{name}: not in this ffmpeg build")
            continue
        argv = [
            ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "error", "-y",
            *seek, *source_args, *limit,
            "-map", "0:V:0", "-an", "-sn", "-dn",
            "-vf", vf,
            *_encoder_args(name, out_w, out_h, fps),
            "-pix_fmt", "yuv420p",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-movflags", "+faststart",
            "-f", "mp4", str(partial),
        ]
        try:
            result = _run(argv, limit_sec, what=f"ffmpeg ({name})")
        except ScreentimeError:
            partial.unlink(missing_ok=True)
            raise
        if result.returncode == 0 and partial.is_file() and partial.stat().st_size > 0:
            os.replace(partial, out)
            return _normalized_info(out, fps)
        partial.unlink(missing_ok=True)
        failures.append(f"{name}: {_tail(result.stderr, 2)}")
    raise ScreentimeError("bad_input", f"Could not re-encode {source.name}. " + "; ".join(failures))


def _normalized_info(path: Path, fps: float) -> VideoInfo:
    data, demuxer = _probe_json(path)
    stream = _video_stream(data, path)
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    try:
        frames = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0:
        frames = _count_packets(path, demuxer) or 0
    if frames <= 0:
        raise ScreentimeError("bad_input", f"Re-encoding {path.name} produced no frames.")
    return VideoInfo(
        width=width, height=height, frame_count=frames, duration_sec=frames / fps, fps=float(fps), has_audio=False
    )


# --------------------------------------------------------------------------------------------------------------------
# Audio


def extract_audio_wav(
    src: PathLike,
    dst: PathLike,
    rate: int = WAV_SAMPLE_RATE,
    *,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    timeout: Optional[float] = None,
) -> bool:
    """Mono 16-bit PCM WAV of the first audio stream at `rate` Hz. False (and no file) when there is no audio."""
    source, out = _absolute_file(src), Path(dst)
    data, _ = _probe_json(source)
    if not any(s.get("codec_type") == "audio" for s in data.get("streams") or []):
        return False
    seek, limit = _trim_args(start_sec, duration_sec)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial")
    argv = [
        ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "error", "-y",
        *seek, *input_args(source), *limit,
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(int(rate)), "-c:a", "pcm_s16le",
        "-map_metadata", "-1", "-f", "wav", str(partial),
    ]
    result = _run(argv, timeout if timeout is not None else NORMALIZE_BASE_TIMEOUT_SEC, what="ffmpeg (audio)")
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise ScreentimeError("bad_input", f"Could not read the audio of {source.name}: {_tail(result.stderr)}")
    try:
        with wave.open(str(partial), "rb") as w:
            samples = w.getnframes()
    except (wave.Error, EOFError):
        samples = 0
    if samples <= 0:
        partial.unlink(missing_ok=True)
        return False
    os.replace(partial, out)
    return True


def wav_duration_sec(path: PathLike) -> float:
    """Length of a PCM WAV file in seconds."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError) as err:
        raise ScreentimeError("bad_input", f"Not a readable WAV file: {Path(path).name} ({err})") from None


# --------------------------------------------------------------------------------------------------------------------
# Frames and crops


def seek_seconds(frame: int, fps: float = FPS) -> float:
    """Half a frame early: an accurate seek returns the first frame with pts >= t, so this lands on `frame` even when
    the decimal time rounds slightly past its exact timestamp (as in the app)."""
    return max(0.0, (frame - 0.5) / fps)


def frame_jpeg(
    video: PathLike,
    frame: int,
    fps: float = FPS,
    max_side: Optional[int] = None,
    *,
    timeout: float = EXTRACT_TIMEOUT_SEC,
) -> bytes:
    """JPEG of frame `frame` (0-based) of a normalized video, optionally scaled so its longest side <= max_side."""
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ScreentimeError("bad_input", f"Invalid frame index {frame!r}.")
    path = _absolute_file(video)
    seek = ["-ss", f"{seek_seconds(frame, fps):.6f}"] if frame > 0 else []
    filters = []
    if max_side is not None:
        if max_side < 1:
            raise ValueError("max_side must be positive")
        m = int(max_side)
        filters = ["-vf", f"scale=w='if(gte(iw,ih),min(iw,{m}),-1)':h='if(gte(iw,ih),-1,min(ih,{m}))'"]
    argv = [
        ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "error",
        *seek, *input_args(path),
        "-map", "0:V:0", "-an", "-sn", "-dn", *filters,
        "-frames:v", "1", "-q:v", str(JPEG_QSCALE), "-c:v", "mjpeg", "-f", "image2pipe", "pipe:1",
    ]
    result = _run(argv, timeout, what="ffmpeg (frame)")
    data = result.stdout
    if not data:
        # A seek past the last frame writes nothing (ffmpeg 7 exits 0, ffmpeg 9 exits with an error); say which.
        frames = probe(path).frame_count
        if frames and frame >= frames:
            raise ScreentimeError("bad_input", f"Frame {frame} is past the end of {path.name} ({frames} frames).")
        raise ScreentimeError("bad_input", f"Could not read frame {frame} of {path.name}: {_tail(result.stderr)}")
    if result.returncode != 0 or not data.startswith(b"\xff\xd8"):
        raise ScreentimeError("bad_input", f"Could not read frame {frame} of {path.name}: {_tail(result.stderr)}")
    return data


def _box_values(box: Box | Mapping[str, float] | Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(box, Box):
        return box.x1, box.y1, box.x2, box.y2
    if isinstance(box, Mapping):
        return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def _js_round(value: float) -> int:
    """JavaScript Math.round (halves go up), which the app's paddedCrop used."""
    return math.floor(value + 0.5)


def padded_crop(
    box: Box | Mapping[str, float] | Sequence[float], pad: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Inclusive box grown by `pad` (a fraction of the box size per side), clamped to the image: (x1, y1, x2, y2),
    inclusive. Same arithmetic as the app's paddedCrop."""
    x1, y1, x2, y2 = _box_values(box)
    pad_x = _js_round((x2 - x1 + 1) * pad)
    pad_y = _js_round((y2 - y1 + 1) * pad)
    left = max(0, math.floor(x1) - pad_x)
    top = max(0, math.floor(y1) - pad_y)
    right = min(width - 1, math.ceil(x2) + pad_x)
    bottom = min(height - 1, math.ceil(y2) + pad_y)
    if right < left or bottom < top:
        raise ScreentimeError("bad_input", "The crop box lies outside the frame.")
    return left, top, right, bottom


def crop_jpeg(
    jpeg_bytes: bytes,
    box: Box | Mapping[str, float] | Sequence[float],
    pad: float = 0.15,
    *,
    quality: int = 90,
) -> bytes:
    """JPEG of `box` (inclusive x2/y2, in this image's pixels) grown by `pad` per side and clamped to the image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            img.load()
            left, top, right, bottom = padded_crop(box, pad, img.width, img.height)
            region = img.crop((left, top, right + 1, bottom + 1)).convert("RGB")
    except (UnidentifiedImageError, OSError) as err:
        raise ScreentimeError("bad_input", f"Not a readable image: {err}") from None
    out = io.BytesIO()
    region.save(out, format="JPEG", quality=quality)
    return out.getvalue()


# --------------------------------------------------------------------------------------------------------------------
# Files


def sha256_file(path: PathLike) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
