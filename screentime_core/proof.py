"""The proof reel: the measured video with the evidence drawn on every frame.

Frame N of the reel is frame N of the normalized video, with the tracker's boxes for that frame drawn on it: solid lime
on the product where the frame was counted, dashed red on a rejected look-alike, dashed amber where the check was
unclear. A title band above the video names the product and its measured total. A band below it holds a strip of the
whole video (green where frames were counted, red where a look-alike was on screen), a playhead, and a counter that only
moves on counted frames. The bands sit outside the picture, so nothing but the boxes and their tags covers the video.

How: ffmpeg decodes the video to raw RGB at the output size through a pipe, Pillow draws each frame, and a second ffmpeg
encodes H.264 (yuv420p, +faststart) from a pipe, muxing the original audio when one is given. Argv lists only.
"""

from __future__ import annotations

import bisect
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from .errors import ScreentimeError
from .models import Detection, VideoInfo, VideoMetrics

__all__ = ["render_proof"]

RGB = tuple[int, int, int]

LIME: RGB = (163, 230, 53)  # counted (#a3e635)
LIME_INK: RGB = (26, 46, 5)
RED: RGB = (239, 68, 68)  # look-alike, not counted (#ef4444)
AMBER: RGB = (245, 158, 11)  # unclear, not counted (#f59e0b)
AMBER_INK: RGB = (69, 26, 3)
INK: RGB = (20, 19, 15)  # bands and outlines (#14130f)
TRACK: RGB = (62, 61, 56)  # the strip where nothing was counted
WHITE: RGB = (255, 255, 255)
MUTED: RGB = (196, 194, 187)

AUDIO_KBPS = 96
MAX_TAGS = 6  # per frame
ENCODERS = ("libx264", "h264_videotoolbox", "libopenh264", "mpeg4")  # preference order
_FFMPEG_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")  # GUI-launched agents often lack these on PATH
_MIN_HEIGHT = 240
_PHONE = (390.0, 844.0)  # a phone screen in points, portrait
_REF_SCALE = 1.15  # points per pixel of a 9:16 reel shown full screen on that phone: the scale base sizes are set for
# RGB to tagged BT.709 yuv420p. setparams labels the frames too: ffmpeg 7+ takes colour tags from frames, not options.
_TO_YUV = ("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
           "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
_COLOR_TAGS = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv")
# Before every -i: local files and our own pipes only, so no input can pull in a network URL or a nested protocol.
_LOCAL_ONLY = ("-protocol_whitelist", "file,pipe")


# --------------------------------------------------------------------------------------------------------------------
# Public entry point


def render_proof(
    video_path: str,
    detections: Sequence[Detection],
    verdict_by_object: Mapping[str, str],
    metrics: VideoMetrics,
    info: VideoInfo,
    out_mp4: str,
    *,
    product_label: str,
    max_height: int = 720,
    max_mb: float = 19.0,
    workdir: Optional[str] = None,
    audio_source: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> str:
    """Render the proof reel of ``video_path`` (the normalized copy Measure analysed) to ``out_mp4``; return its path.

    ``detections`` are every tracked box (all verdicts) in that video's pixel space (x2, y2 inclusive);
    ``verdict_by_object`` maps str(object_id) to "match", "lookalike" or "unclear" (anything else, or a missing id, is
    drawn as unclear); ``metrics.visible_frames`` are the counted frames. The reel keeps every input frame at the input
    rate, is at most ``max_height`` pixels tall (never upscaled) and aims to stay under ``max_mb`` (10^6 bytes): the
    bitrate is capped from the duration, and one smaller re-encode follows if the first is over. ``audio_source`` (the
    original upload; the normalized copy is silent) gives the reel its sound, trimmed or padded to the video.
    ``timeout_sec`` bounds each encoding pass (default: from the duration). Temporary files go in ``workdir`` (or the
    system temp dir) and are removed. Raises ScreentimeError: ffmpeg_missing, bad_input, timeout or incomplete.
    """
    ffmpeg = _ffmpeg_bin()
    if not video_path or not os.path.isfile(video_path):
        raise ScreentimeError("bad_input", "The video for the proof reel was not found.", fix="Run Measure again.")
    if not info or int(info.width or 0) <= 0 or int(info.height or 0) <= 0:
        raise ScreentimeError("bad_input", "The proof reel needs the video's width and height.")
    fps = float(info.fps or 0)
    if not math.isfinite(fps) or fps <= 0:
        raise ScreentimeError("bad_input", "The proof reel needs the video's frame rate.")
    audio = None
    if audio_source:
        if not os.path.isfile(audio_source):
            raise ScreentimeError("bad_input", "The audio source for the proof reel was not found.")
        if _has_audio(ffmpeg, audio_source):
            audio = audio_source

    try:
        label = _clean_label(product_label)
        overlay = _Overlay(info, fps, detections, verdict_by_object or {}, metrics, label, max_height)
    except ScreentimeError:
        raise
    except Exception as exc:  # a drawing bug must cost the reel, not the Measure run that asked for it
        raise ScreentimeError("incomplete", f"The proof reel could not be laid out ({type(exc).__name__}).") from exc
    encoder = _pick_encoder(ffmpeg)

    duration = float(info.duration_sec or 0)
    if not math.isfinite(duration) or duration <= 0:
        duration = overlay.slots / fps
    max_bytes = max(1, int(max_mb * 1_000_000))
    audio_kbps = AUDIO_KBPS if audio else 0
    cap_kbps = max(64, int(max_bytes * 8 * 0.92 / max(duration, 1.0) / 1000) - audio_kbps)
    layout = overlay.layout
    quality_kbps = max(300, int(layout.width * layout.height * fps * 0.15 / 1000))
    kbps = cap_kbps if encoder == "libx264" else min(cap_kbps, quality_kbps)
    timeout = float(timeout_sec) if timeout_sec else max(120.0, 30.0 + 5.0 * duration)

    # Written next to the destination and renamed at the end, so a half-written reel never sits at out_mp4.
    out_mp4 = os.path.abspath(out_mp4)
    out_dir = os.path.dirname(out_mp4)
    partial = os.path.join(out_dir, f".{os.path.basename(out_mp4)}.{uuid.uuid4().hex[:8]}.partial.mp4")
    tmp = None
    try:
        os.makedirs(out_dir, exist_ok=True)
        if workdir:
            os.makedirs(workdir, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="screentime-proof-", dir=workdir or None)
        job = _Job(ffmpeg, video_path, overlay, encoder, fps, audio, partial, tmp, timeout)
        job.run(crf=23, kbps=kbps)
        size = os.path.getsize(partial)
        if size > max_bytes:
            # Over the cap: one smaller pass, from the bitrate the first pass actually used.
            used_kbps = max(1, int(size * 8 / 1000 / max(duration, 0.1)) - audio_kbps)
            kbps = max(48, int(min(kbps, used_kbps) * max_bytes / size * 0.85))
            job.run(crf=28, kbps=kbps)
        os.replace(partial, out_mp4)
    except ScreentimeError:
        raise
    except Exception as exc:  # disk full, no permission, a drawing bug: the reel is lost, the run is not
        why = exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__
        raise ScreentimeError("incomplete", f"The proof reel could not be finished ({why}).", retryable=True) from exc
    finally:
        try:
            os.remove(partial)
        except OSError:
            pass
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


# --------------------------------------------------------------------------------------------------------------------
# ffmpeg helpers (private: media.py belongs to another module owner)


def _ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for folder in _FFMPEG_DIRS:
        path = os.path.join(folder, "ffmpeg")
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise ScreentimeError("ffmpeg_missing", "ffmpeg is not installed, so the proof reel cannot be made.",
                          fix="Install ffmpeg (brew install ffmpeg, or apt install ffmpeg) and try again.")


def _run_text(args: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return -1, "", ""
    return p.returncode, p.stdout, p.stderr


def _has_audio(ffmpeg: str, path: str) -> bool:
    """True when ffmpeg lists an audio stream in ``path`` (``ffmpeg -i`` exits 1 by design; only the listing matters)."""
    _, _, err = _run_text([ffmpeg, "-hide_banner", "-nostdin", *_LOCAL_ONLY, "-i", path])
    return re.search(r"Stream #\d+:\d+\S*: Audio:", err) is not None


@lru_cache(maxsize=4)
def _passthrough(ffmpeg: str) -> tuple[str, ...]:
    """Keep every frame as it comes: -fps_mode since ffmpeg 5.1 (ffmpeg 8+ dropped -vsync), -vsync before."""
    _, out, _ = _run_text([ffmpeg, "-hide_banner", "-version"])
    m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
    if m and (int(m.group(1)), int(m.group(2))) < (5, 1):
        return ("-vsync", "passthrough")
    return ("-fps_mode", "passthrough")


def _codec_args(name: str, *, crf: int, kbps: int, gop: int) -> list[str]:
    if name == "libx264":  # constant quality, with a bitrate ceiling that keeps the file under the size cap
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-maxrate", f"{kbps}k",
                "-bufsize", f"{2 * kbps}k", "-g", str(gop)]
    if name == "h264_videotoolbox":
        return ["-c:v", name, "-b:v", f"{kbps}k", "-g", str(gop), "-allow_sw", "1"]
    if name == "libopenh264":
        return ["-c:v", name, "-b:v", f"{kbps}k", "-g", str(gop), "-allow_skip_frames", "0"]
    return ["-c:v", "mpeg4", "-b:v", f"{kbps}k", "-g", str(gop)]


def _listed_encoders(ffmpeg: str) -> set[str]:
    _, out, _ = _run_text([ffmpeg, "-hide_banner", "-encoders"])
    return set(re.findall(r"^\s*V[A-Z.]{5}\s+(\S+)", out, flags=re.MULTILINE))


def _encoder_works(ffmpeg: str, name: str) -> bool:
    """A listed encoder can still fail (no hardware, missing library): encode three frames to find out."""
    w = h = 128
    args = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{w}x{h}", "-framerate", "15", *_LOCAL_ONLY, "-i", "pipe:0", "-vf", _TO_YUV,
            *_codec_args(name, crf=23, kbps=500, gop=30), *_COLOR_TAGS, "-f", "null", "-"]
    try:
        p = subprocess.run(args, input=bytes(w * h * 3 * 3), capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0


@lru_cache(maxsize=4)
def _pick_encoder(ffmpeg: str) -> str:
    listed = _listed_encoders(ffmpeg)
    for name in ENCODERS:
        if name in listed and _encoder_works(ffmpeg, name):
            return name
    raise ScreentimeError("ffmpeg_missing", "This ffmpeg cannot encode H.264 or MPEG-4 video, so the proof reel "
                          "cannot be made.", fix="Install an ffmpeg build with libx264 and try again.")


def _log_tail(path: str, limit: int = 240) -> str:
    try:
        with open(path, "rb") as fh:
            text = fh.read()[-4000:].decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][-limit:] if lines else ""


def _kill(expired: threading.Event, *procs: Optional[subprocess.Popen]) -> None:
    expired.set()
    for p in procs:
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


class _Job:
    """One encoding pass: decoder pipe -> Pillow -> encoder pipe."""

    def __init__(self, ffmpeg: str, video: str, overlay: _Overlay, encoder: str, fps: float,
                 audio: Optional[str], out: str, tmp: str, timeout: float):
        self.ffmpeg, self.video, self.overlay, self.encoder = ffmpeg, video, overlay, encoder
        self.fps, self.audio, self.out, self.tmp, self.timeout = fps, audio, out, tmp, timeout
        rate = Fraction(fps).limit_denominator(1001)
        self.rate = f"{rate.numerator}/{rate.denominator}"
        self.gop = max(1, int(round(fps * 2)))  # a keyframe every 2 s keeps scrubbing in chat players quick

    def _args(self, crf: int, kbps: int) -> tuple[list[str], list[str]]:
        L = self.overlay.layout
        keep = list(_passthrough(self.ffmpeg))
        dec = [self.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", *_LOCAL_ONLY, "-i", self.video,
               "-map", "0:v:0", "-vf", f"scale={L.width}:{L.video_h}:flags=lanczos,format=rgb24", *keep,
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        enc = [self.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{L.width}x{L.height}",
               "-framerate", self.rate, *_LOCAL_ONLY, "-i", "pipe:0"]
        if self.audio:
            enc += [*_LOCAL_ONLY, "-i", self.audio]
        enc += ["-map", "0:v:0"]
        if self.audio:
            enc += ["-map", "1:a:0"]
        enc += ["-vf", _TO_YUV, *_codec_args(self.encoder, crf=crf, kbps=kbps, gop=self.gop), *_COLOR_TAGS, *keep]
        if self.audio:
            # Padded with silence and cut at the last video frame, so the audio never changes the frame count.
            enc += ["-af", "apad", "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-shortest"]
        enc += ["-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", self.out]
        return dec, enc

    def run(self, *, crf: int, kbps: int) -> int:
        dec_args, enc_args = self._args(crf, kbps)
        L = self.overlay.layout
        frame_bytes = L.width * L.video_h * 3
        dec_log_path = os.path.join(self.tmp, "decode.log")
        enc_log_path = os.path.join(self.tmp, "encode.log")
        expired = threading.Event()
        dec = enc = None
        timer = None
        frames = 0
        broken = False
        with open(dec_log_path, "wb") as dec_log, open(enc_log_path, "wb") as enc_log:
            try:
                try:
                    dec = subprocess.Popen(dec_args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=dec_log)
                    enc = subprocess.Popen(enc_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_log)
                except OSError as exc:
                    raise ScreentimeError("ffmpeg_missing", f"ffmpeg could not be started ({exc.strerror or exc}).",
                                          fix="Check the ffmpeg install and try again.") from exc
                timer = threading.Timer(self.timeout, _kill, args=(expired, dec, enc))
                timer.daemon = True
                timer.start()
                assert dec.stdout is not None and enc.stdin is not None
                try:
                    while True:
                        raw = dec.stdout.read(frame_bytes)
                        if len(raw) < frame_bytes:
                            break
                        enc.stdin.write(self.overlay.compose(raw, frames))
                        frames += 1
                except (BrokenPipeError, ConnectionResetError):
                    broken = True
                try:
                    enc.stdin.close()
                except OSError:
                    broken = True
                if broken:
                    _kill(threading.Event(), dec)  # nobody reads its output any more
                enc.wait()
                dec.wait()
            finally:
                if timer is not None:
                    timer.cancel()
                for p in (dec, enc):
                    if p is not None:
                        if p.poll() is None:
                            p.kill()
                            p.wait()
                if dec is not None and dec.stdout is not None:
                    dec.stdout.close()
        ok = not broken and frames > 0 and dec.returncode == 0 and enc.returncode == 0
        if ok:
            return frames
        if expired.is_set():
            raise ScreentimeError("timeout", f"The proof reel took longer than {self.timeout:.0f} s and was stopped.",
                                  fix="Try again; a shorter video renders faster.", retryable=True)
        if frames == 0:
            why = _log_tail(dec_log_path) or _log_tail(enc_log_path)
            raise ScreentimeError("bad_input", "ffmpeg could not read the video for the proof reel"
                                  + (f": {why}" if why else "."), fix="Run Measure again on the original file.")
        if dec.returncode != 0:
            why = _log_tail(dec_log_path)
            raise ScreentimeError("incomplete", f"The video stopped decoding after {frames} frames"
                                  + (f": {why}" if why else "."), retryable=True)
        why = _log_tail(enc_log_path)
        raise ScreentimeError("incomplete", f"ffmpeg ({self.encoder}) stopped while writing the proof reel"
                              + (f": {why}" if why else "."), retryable=True)


# --------------------------------------------------------------------------------------------------------------------
# Text and layout


def _tenths(x: float) -> str:
    """Seconds to one decimal, rounded down, so a reel never shows more time than was measured."""
    return f"{math.floor(max(0.0, x) * 10 + 1e-6) / 10:.1f}"


_PUNCT = {0x2014: "-", 0x2013: "-", 0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'}


def _clean_label(text: Any) -> str:
    """One printable line the default font can draw (it has Latin glyphs, no emoji)."""
    text = " ".join(str(text or "").split()).translate(_PUNCT)
    text = "".join(ch for ch in text if ch.isprintable() and (ord(ch) < 0x2000 or ch == "\u2026"))
    return text.strip()[:120]


class _Font:
    """The default font (Aileron, scalable since Pillow 10.1) at one size, positioned by cap height."""

    def __init__(self, px: int):
        try:
            self.font: Any = ImageFont.load_default(size=px)
        except TypeError:  # Pillow without a scalable default font
            self.font = ImageFont.load_default()
        box = self.font.getbbox("H")
        self.cap_top = int(box[1])
        self.cap_h = max(1, int(box[3] - box[1]))
        self._widths: dict[str, int] = {}

    def width(self, text: str) -> int:
        w = self._widths.get(text)
        if w is None:
            w = self._widths[text] = int(math.ceil(self.font.getlength(text)))
        return w

    def draw(self, d: ImageDraw.ImageDraw, x: int, cap_y: int, text: str, fill: RGB) -> int:
        """Draw ``text`` with its cap height starting at ``cap_y``; return the x after it."""
        d.text((x, cap_y - self.cap_top), text, font=self.font, fill=fill)
        return x + self.width(text)

    def fit(self, text: str, max_w: int) -> str:
        """``text``, cut with an ellipsis until it fits ``max_w``."""
        if self.width(text) <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.width(text[:mid].rstrip() + "\u2026") <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + "\u2026" if lo else "\u2026"


Run = tuple[str, RGB]


def _runs_width(runs: Sequence[Run], font: _Font) -> int:
    return sum(font.width(t) for t, _ in runs)


def _draw_runs(d: ImageDraw.ImageDraw, x: int, cap_y: int, runs: Sequence[Run], font: _Font) -> None:
    for text, color in runs:
        x = font.draw(d, x, cap_y, text, color)


class _Sizes:
    """Every size in the reel, from one unit ``s`` (1.0 = a 9:16 reel about 340 px wide)."""

    def __init__(self, s: float):
        def px(base: float, lo: int = 1) -> int:
            return max(lo, int(round(base * s)))

        self.s = s
        self.title_px, self.tag_px, self.count_px = px(17, 10), px(13, 9), px(15, 9)
        self.pad = px(12, 6)
        self.title_pad, self.title_line, self.title_gap = px(9, 5), px(17 * 1.3, 13), px(2, 1)
        self.band_pad_top, self.count_line, self.count_gap = px(8, 4), px(15 * 1.25, 12), px(7, 4)
        self.bar_h, self.lane_gap, self.lane_h, self.band_pad_bottom = px(10, 6), px(2, 1), px(4, 3), px(9, 5)
        self.playhead_w, self.playhead_ext = px(3, 2), px(3, 2)
        self.box_w = min(6, max(4, int(round(4 * s))))
        self.dash_w = min(5, max(3, int(round(3 * s))))
        self.tag_pad_x, self.tag_pad_y, self.tag_gap, self.tag_margin = px(6, 4), px(5, 3), px(4, 3), px(4, 2)

    def title_height(self, lines: int) -> int:
        return 2 * self.title_pad + lines * self.title_line + (lines - 1) * self.title_gap

    def band_height(self) -> int:
        return (self.band_pad_top + self.count_line + self.count_gap + self.bar_h + self.lane_gap + self.lane_h
                + self.band_pad_bottom)


def _unit(w: float, h: float) -> float:
    """The size unit for a ``w`` x ``h`` reel: text should look the same size on a phone whatever the shape."""
    pw, ph = _PHONE
    scale = min(pw / w, ph / h) if w <= h else min(ph / w, pw / h)
    return min(2.4, max(0.8, _REF_SCALE / max(scale, 1e-6)))


def _title_lines(label: str, total: str, font: _Font, max_w: int) -> list[list[Run]]:
    """"<label>: 7.9 s on screen" on one line when it fits, else the label (cut to fit) over the total."""
    number: list[Run] = [(f"{total} s", LIME), (" on screen", WHITE)]
    if label:
        one: list[Run] = [(f"{label}: ", WHITE), *number]
        if _runs_width(one, font) <= max_w:
            return [one]
    if _runs_width(number, font) > max_w:  # a very narrow reel keeps the number
        number = number[:1]
    return [[(font.fit(label, max_w), WHITE)], number] if label else [number]


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


@dataclass
class _Layout:
    sizes: _Sizes
    width: int  # canvas width = video width
    height: int  # canvas height
    video_h: int
    video_y: int  # top of the video = height of the title band
    band_y: int  # top of the strip band
    band_h: int
    title: list[list[Run]]
    title_font: _Font
    tag_font: _Font
    count_font: _Font


def _layout(info: VideoInfo, max_height: int, label: str, total: str) -> _Layout:
    w0, h0 = int(info.width), int(info.height)
    cap = max(_MIN_HEIGHT, int(max_height))
    fonts: dict[int, _Font] = {}

    def font(px: int) -> _Font:
        if px not in fonts:
            fonts[px] = _Font(px)
        return fonts[px]

    # The unit depends on the reel's shape, which depends on the bands, which depend on the unit: settle it in turns.
    vh = float(min(h0, cap))
    vw = w0 * vh / h0
    top = bottom = 0
    sizes = _Sizes(1.0)
    for _ in range(4):
        sizes = _Sizes(_unit(vw, vh + top + bottom))
        lines = _title_lines(label, total, font(sizes.title_px), int(vw) - 2 * sizes.pad)
        top, bottom = sizes.title_height(len(lines)), sizes.band_height()
        vh = float(max(32, min(h0, cap - top - bottom)))
        vw = w0 * vh / h0
    # Whole pixels: an even width (yuv420p), the height from the aspect, and the title wrapped for that width.
    tfont = font(sizes.title_px)
    width = _even(vw)
    for _ in range(4):
        lines = _title_lines(label, total, tfont, width - 2 * sizes.pad)
        top = sizes.title_height(len(lines))
        video_h = max(32, min(h0, cap - top - bottom, int(round(width * h0 / w0))))
        new_width = min(width, _even(w0 * video_h / h0))
        if new_width == width:
            break
        width = new_width
    height = top + video_h + bottom
    if height % 2:
        bottom += 1
        height += 1
    return _Layout(sizes, width, height, video_h, top, top + video_h, bottom, lines, tfont,
                   font(sizes.tag_px), font(sizes.count_px))


# --------------------------------------------------------------------------------------------------------------------
# Drawing


@dataclass(frozen=True)
class _Style:
    color: RGB
    ink: RGB  # text on the tag
    dashed: bool
    counted: bool
    tag: str
    short: str  # tag for a narrow video
    rank: int  # tags are placed in rank order (counted first); boxes are drawn the other way (counted on top)


COUNTED = _Style(LIME, LIME_INK, False, True, "counted", "counted", 0)
LOOKALIKE = _Style(RED, WHITE, True, False, "not counted: look-alike", "look-alike", 1)
UNCLEAR = _Style(AMBER, AMBER_INK, True, False, "not counted: unclear", "unclear", 2)
# A "match" box on a frame the metrics did not count (should not happen): drawn so the reel never says "counted" for
# a frame the strip and the total leave out.
UNCOUNTED = _Style(LIME, LIME_INK, True, False, "not counted", "not counted", 3)

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1, inclusive


def _style_for(verdict: Any, frame_counted: bool) -> _Style:
    if verdict == "match":
        return COUNTED if frame_counted else UNCOUNTED
    if verdict == "lookalike":
        return LOOKALIKE
    return UNCLEAR


def _as_frame(value: Any) -> Optional[int]:
    """A frame index from a number; None unless it is a finite number >= 0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) and f >= 0 else None


def _verdict_of(verdicts: Mapping[Any, Any], det: Any) -> Any:
    oid = getattr(det, "object_id", None)
    v = verdicts.get(str(oid))
    return verdicts.get(oid) if v is None else v


def _runs(frames: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted unique frames to inclusive runs of consecutive frames."""
    out: list[tuple[int, int]] = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        elif not out or f > out[-1][1]:
            out.append((f, f))
    return out


def _hit(a: Rect, b: Rect, margin: int = 0) -> bool:
    return a[0] - margin <= b[2] and b[0] <= a[2] + margin and a[1] - margin <= b[3] and b[1] <= a[3] + margin


def _dash_spans(a: int, z: int, dash: int, gap: int) -> list[tuple[int, int]]:
    """Dashes along [a, z] that start and end on the corners."""
    length = z - a + 1
    if length <= 2 * dash + gap:
        return [(a, z)]
    n = max(2, int(round((length + gap) / (dash + gap))))
    step = (length - dash) / (n - 1)
    spans = []
    for k in range(n):
        start = a + int(round(k * step))
        spans.append((start, min(z, start + dash - 1)))
    return spans


def _solid_box(d: ImageDraw.ImageDraw, r: Rect, color: RGB, width: int) -> None:
    x0, y0, x1, y1 = r
    d.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=INK, width=1)
    d.rectangle(r, outline=color, width=width)


def _dashed_box(d: ImageDraw.ImageDraw, r: Rect, color: RGB, width: int) -> None:
    x0, y0, x1, y1 = r
    dash, gap = max(6, 3 * width + 2), max(4, 2 * width)
    segs: list[Rect] = []
    for a, z in _dash_spans(x0, x1, dash, gap):
        segs.append((a, y0, z, min(y1, y0 + width - 1)))
        segs.append((a, max(y0, y1 - width + 1), z, y1))
    for a, z in _dash_spans(y0, y1, dash, gap):
        segs.append((x0, a, min(x1, x0 + width - 1), z))
        segs.append((max(x0, x1 - width + 1), a, x1, z))
    for sx0, sy0, sx1, sy1 in segs:  # a dark rim under every dash keeps it visible on any background
        d.rectangle((sx0 - 1, sy0 - 1, sx1 + 1, sy1 + 1), fill=INK)
    for seg in segs:
        d.rectangle(seg, fill=color)


class _Overlay:
    """Everything drawn on the reel. ``compose(raw, i)`` turns decoded frame ``i`` into the reel's frame ``i``."""

    def __init__(self, info: VideoInfo, fps: float, detections: Sequence[Detection], verdicts: Mapping[Any, Any],
                 metrics: VideoMetrics, label: str, max_height: int):
        self.fps = fps
        frame_count = int(info.frame_count or 0)
        self.slots = frame_count if frame_count > 0 else max(1, int(round(float(info.duration_sec or 0) * fps)))
        duration = float(info.duration_sec or 0)
        if not math.isfinite(duration) or duration <= 0:
            duration = self.slots / fps
        self.counted = sorted({f for f in map(_as_frame, metrics.visible_frames or []) if f is not None})
        self._counted_set = set(self.counted)
        try:
            seconds = float(metrics.seconds_on_screen)
        except (TypeError, ValueError):
            seconds = math.nan
        if not math.isfinite(seconds) or seconds < 0:
            seconds = len(self.counted) / fps
        self.total = _tenths(duration)
        self.label = label
        L = self.layout = _layout(info, max_height, label, _tenths(seconds))
        self.sx = L.width / float(info.width)
        self.sy = L.video_h / float(info.height)
        self._w0, self._h0 = int(info.width), int(info.height)

        # Boxes by frame, with their style, and the frames each kind of rejected box was on screen.
        self.by_frame: dict[int, list[tuple[Rect, _Style, int]]] = defaultdict(list)
        lookalike: set[int] = set()
        unclear: set[int] = set()
        for det in detections or []:
            frame = _as_frame(getattr(det, "frame", None))
            rect = self._rect(det)
            if frame is None or rect is None:
                continue
            style = _style_for(_verdict_of(verdicts, det), frame in self._counted_set)
            try:
                oid = int(det.object_id)
            except (TypeError, ValueError):
                oid = -1
            self.by_frame[frame].append((rect, style, oid))
            if style is LOOKALIKE:
                lookalike.add(frame)
            elif style is UNCLEAR:
                unclear.add(frame)

        sz = L.sizes
        self.tag_h = L.tag_font.cap_h + 2 * sz.tag_pad_y
        self._tags: dict[tuple[str, _Style], tuple[str, int]] = {}
        self.bar_x, self.bar_w = sz.pad, L.width - 2 * sz.pad
        self.bar_y = sz.band_pad_top + sz.count_line + sz.count_gap  # inside the band
        self.lane_y = self.bar_y + sz.bar_h + sz.lane_gap
        widest = [_runs_width(self._counter(self.slots, True, k), L.count_font) for k in (0, 1)]
        self._form = next((k for k in (0, 1) if widest[k] <= self.bar_w), 2)

        # The canvas keeps the title band; each frame repaints the video and the strip band.
        self.canvas = Image.new("RGB", (L.width, L.height), INK)
        d = ImageDraw.Draw(self.canvas)
        y = sz.title_pad
        for line in L.title:
            _draw_runs(d, sz.pad, y + (sz.title_line - L.title_font.cap_h) // 2, line, L.title_font)
            y += sz.title_line + sz.title_gap
        self.band = Image.new("RGB", (L.width, L.band_h), INK)
        d = ImageDraw.Draw(self.band)
        d.rectangle((self.bar_x, self.bar_y, self.bar_x + self.bar_w - 1, self.bar_y + sz.bar_h - 1), fill=TRACK)
        for a, b in _runs(f for f in self.counted if f < self.slots):
            x0, x1 = self._span(a, b)
            d.rectangle((x0, self.bar_y, x1, self.bar_y + sz.bar_h - 1), fill=LIME)
        for frames, color in ((unclear, AMBER), (lookalike, RED)):
            for a, b in _runs(sorted(f for f in frames if f < self.slots)):
                x0, x1 = self._span(a, b)
                d.rectangle((x0, self.lane_y, x1, self.lane_y + sz.lane_h - 1), fill=color)
        self._legend(d, bool(lookalike), bool(unclear))

    # -- geometry

    def _rect(self, det: Any) -> Optional[Rect]:
        """The detection's box in reel pixels (inclusive), clamped to the frame; None when it misses the frame."""
        try:
            box = det.box
            x1, y1, x2, y2 = float(box.x1), float(box.y1), float(box.x2), float(box.y2)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 < 0 or y2 < 0 or x1 > self._w0 - 1 or y1 > self._h0 - 1:
            return None
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(self._w0 - 1.0, x2), min(self._h0 - 1.0, y2)
        vw, vh = self.layout.width, self.layout.video_h
        rx0 = min(max(int(math.floor(x1 * self.sx)), 0), vw - 1)
        ry0 = min(max(int(math.floor(y1 * self.sy)), 0), vh - 1)
        rx1 = min(max(int(math.ceil((x2 + 1) * self.sx)) - 1, rx0), vw - 1)
        ry1 = min(max(int(math.ceil((y2 + 1) * self.sy)) - 1, ry0), vh - 1)
        return (rx0, ry0, rx1, ry1)

    def _span(self, a: int, b: int) -> tuple[int, int]:
        """Strip pixels covering frames a..b (inclusive), at least 2 px wide."""
        x0 = self.bar_x + int(math.floor(self.bar_w * a / self.slots))
        x1 = self.bar_x + int(math.ceil(self.bar_w * (b + 1) / self.slots)) - 1
        x1 = min(max(x1, x0 + 1), self.bar_x + self.bar_w - 1)
        return min(x0, x1 - 1), x1

    # -- static pieces

    def _counter(self, n: int, on: bool, form: Optional[int] = None) -> list[Run]:
        """"counted 3.4 s of 21.7 s"; on a narrow reel "3.4 s of 21.7 s", then "3.4 s". Lime while counting."""
        form = self._form if form is None else form
        number: Run = (f"{_tenths(n / self.fps)} s", LIME if on else WHITE)
        runs: list[Run] = [("counted ", MUTED), number, (f" of {self.total} s", MUTED)]
        return runs if form == 0 else runs[1:] if form == 1 else [number]

    def _legend(self, d: ImageDraw.ImageDraw, lookalike: bool, unclear: bool) -> None:
        """Names the colours under the strip, at the right of the counter line, when there is room."""
        items = [(c, t) for c, t, on in ((RED, "look-alike", lookalike), (AMBER, "unclear", unclear)) if on]
        if not items:
            return
        sz, f = self.layout.sizes, self.layout.count_font
        chip = max(6, int(round(10 * sz.s)))
        space = max(4, int(round(5 * sz.s)))
        widths = [chip + space + f.width(t) for _, t in items]
        total = sum(widths) + 2 * space * (len(items) - 1)
        if _runs_width(self._counter(self.slots, True), f) + 2 * space + total > self.bar_w:
            return
        x = self.bar_x + self.bar_w - total
        cap_y = sz.band_pad_top + (sz.count_line - f.cap_h) // 2
        for (color, text), w in zip(items, widths, strict=True):
            chip_y = cap_y + (f.cap_h - sz.lane_h) // 2
            d.rectangle((x, chip_y, x + chip - 1, chip_y + sz.lane_h - 1), fill=color)
            f.draw(d, x + chip + space, cap_y, text, MUTED)
            x += w + 2 * space

    def _tag(self, text: str, style: _Style) -> tuple[str, int]:
        """The tag text that fits the video's width, and its pill width."""
        key = (text, style)
        hit = self._tags.get(key)
        if hit is not None:
            return hit
        f, sz = self.layout.tag_font, self.layout.sizes
        room = self.layout.width - 2 * sz.tag_margin - 2 * sz.tag_pad_x
        if f.width(text) > room:
            if text.startswith("counted: "):
                text = "counted: " + f.fit(text[len("counted: "):], room - f.width("counted: "))
            if f.width(text) > room:
                text = f.fit(style.short, room)
        out = (text, f.width(text) + 2 * sz.tag_pad_x)
        self._tags[key] = out
        return out

    def _place(self, boxes: list[Rect], sizes: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """One tag per box: above, below, then inside its top edge, left or right aligned, inside the frame and off the
        other tags (the web player's layoutLabels)."""
        vw, vh = self.layout.width, self.layout.video_h
        m, gap = self.layout.sizes.tag_margin, self.layout.sizes.tag_gap
        placed: list[Rect] = []
        out: list[tuple[int, int]] = []
        for i, (r, (w, h)) in enumerate(zip(boxes, sizes, strict=True)):
            right = max(m, vw - w - m)
            xs = [int(min(max(x, m), right)) for x in (r[0] - 1, r[2] - w + 2)]
            ys = [r[1] - h - gap, r[3] + gap + 1, r[1] + gap]
            best: Optional[Rect] = None
            best_score = 0
            for y in ys:
                if y < m or y + h > vh - m:
                    continue
                for x in xs:
                    cand = (x, y, x + w - 1, y + h - 1)
                    score = 1000 * sum(1 for p in placed if _hit(p, cand, 3))
                    score += 10 * sum(1 for j, o in enumerate(boxes) if j != i and _hit(o, cand))
                    if best is None or score < best_score:
                        best, best_score = cand, score
            if best is None:
                y = int(min(max(r[1] + gap, m), max(m, vh - h - m)))
                best = (xs[0], y, xs[0] + w - 1, y + h - 1)
            for _ in range(8):  # still on another tag: step down under it while there is room
                if not any(_hit(p, best, 2) for p in placed) or best[3] + h + gap > vh - m:
                    break
                best = (best[0], best[1] + h + gap, best[2], best[3] + h + gap)
            placed.append(best)
            out.append((best[0], best[1]))
        return out

    # -- per frame

    def _draw_boxes(self, img: Image.Image, items: list[tuple[Rect, _Style, int]]) -> None:
        d = ImageDraw.Draw(img)
        sz = self.layout.sizes
        for rect, style, _ in sorted(items, key=lambda it: -it[1].rank):
            if style.dashed:
                _dashed_box(d, rect, style.color, sz.dash_w)
            else:
                _solid_box(d, rect, style.color, sz.box_w)
        # Tags: counted first; a crowded frame (a shelf of look-alikes) keeps its boxes but not a tag on each.
        ordered = sorted(items, key=lambda it: (it[1].rank, it[2]))[:MAX_TAGS]
        tags: list[tuple[str, int]] = []
        named = False
        for _, style, _ in ordered:
            text = style.tag
            if style is COUNTED and not named and self.label:
                text, named = f"counted: {self.label}", True
            tags.append(self._tag(text, style))
        h = self.tag_h
        spots = self._place([it[0] for it in ordered], [(w, h) for _, w in tags])
        f = self.layout.tag_font
        for (_, style, _), (text, w), (x, y) in zip(ordered, tags, spots, strict=True):
            d.rounded_rectangle((x, y, x + w - 1, y + h - 1), radius=max(1, min(h, w) // 2), fill=style.color,
                                outline=INK, width=1)
            f.draw(d, x + sz.tag_pad_x, y + (h - f.cap_h) // 2, text, style.ink)

    def compose(self, raw: bytes, i: int) -> bytes:
        L, sz = self.layout, self.layout.sizes
        frame = Image.frombytes("RGB", (L.width, L.video_h), raw)
        items = self.by_frame.get(i)
        if items:
            self._draw_boxes(frame, items)
        canvas = self.canvas
        canvas.paste(frame, (0, L.video_y))
        canvas.paste(self.band, (0, L.band_y))
        d = ImageDraw.Draw(canvas)
        f = L.count_font
        n = bisect.bisect_right(self.counted, i)
        cap_y = L.band_y + sz.band_pad_top + (sz.count_line - f.cap_h) // 2
        _draw_runs(d, self.bar_x, cap_y, self._counter(n, i in self._counted_set), f)
        pw = sz.playhead_w
        x = self.bar_x + int(self.bar_w * (min(i, self.slots - 1) + 0.5) / self.slots) - pw // 2
        x = min(max(x, self.bar_x), self.bar_x + self.bar_w - pw)
        top = L.band_y + self.bar_y - sz.playhead_ext
        bottom = L.band_y + self.lane_y + sz.lane_h - 1 + sz.playhead_ext
        d.rectangle((x - 1, top - 1, x + pw, bottom + 1), fill=INK)
        d.rectangle((x, top, x + pw - 1, bottom), fill=WHITE)
        return canvas.tobytes()
