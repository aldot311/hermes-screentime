"""media.py against real ffmpeg on clips generated here (lavfi), no network."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from screentime_core import media
from screentime_core.errors import ScreentimeError
from screentime_core.models import Box

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH")


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args], check=True, timeout=60)


def ffprobe_json(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, timeout=30,
    ).stdout
    return json.loads(out)


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    # 3 s at 30 fps, odd size 641x361 with square pixels (4:4:4 keeps the odd size), a 440 Hz tone.
    odd = d / "odd.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3,scale=641:361,setsar=1,format=yuv444p",
       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
       "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(odd))
    # The same clip tagged to display rotated by 90 degrees (a phone filmed sideways).
    rotated = d / "rotated.mp4"
    try:
        ff("-display_rotation", "90", "-i", str(odd), "-c", "copy", str(rotated))
    except subprocess.CalledProcessError:
        ff("-i", str(odd), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated))
    # Every frame's brightness is its own index: luma 16 + 2*N at 30 fps, lossless.
    coded = d / "coded.mp4"
    ff("-f", "lavfi", "-i", "color=c=black:size=64x48:rate=30:duration=3,format=gray,geq=lum='16+2*N'",
       "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv444p", str(coded))
    return {"odd": odd, "rotated": rotated, "coded": coded, "dir": d}


@pytest.fixture(scope="session")
def normalized(clips) -> tuple[Path, media.VideoInfo]:
    out = clips["dir"] / "norm" / "rotated15.mp4"
    return out, media.normalize(clips["rotated"], out)


@pytest.fixture(scope="session")
def coded15(clips) -> Path:
    out = clips["dir"] / "norm" / "coded15.mp4"
    media.normalize(clips["coded"], out)
    return out


def luma(jpeg: bytes) -> float:
    return ImageStat.Stat(Image.open(io.BytesIO(jpeg)).convert("L")).mean[0]


# --------------------------------------------------------------------------------------------------------------------
# Executables


def test_bins_found():
    assert Path(media.ffmpeg_bin()).name.startswith("ffmpeg")
    assert Path(media.ffprobe_bin()).name.startswith("ffprobe")


def test_missing_ffmpeg_raises_with_the_install_fix(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(ScreentimeError) as err:
        media.ffmpeg_bin()
    assert err.value.code == "ffmpeg_missing"
    assert "brew install ffmpeg" in err.value.fix and "apt install ffmpeg" in err.value.fix


def test_encoders_lists_an_encoder_from_the_chain():
    found = media.encoders()
    assert isinstance(found, dict) and found
    assert media.video_encoders(), "no H.264 or MPEG-4 encoder in this ffmpeg"
    assert all(name in found for name in media.video_encoders())


def test_no_shell_anywhere():
    source = Path(media.__file__).read_text()
    assert "shell=True" not in source
    assert "os.system" not in source


# --------------------------------------------------------------------------------------------------------------------
# Probe and the untrusted-input guard


def test_probe_reports_the_displayed_size(clips):
    plain = media.probe(clips["odd"])
    assert (plain.width, plain.height) == (641, 361)
    info = media.probe(clips["rotated"])
    assert (info.width, info.height) == (361, 641)  # rotated 90 degrees for display
    assert info.frame_count == 90
    assert info.fps == pytest.approx(30)
    assert info.duration_sec == pytest.approx(3, abs=0.05)
    assert info.has_audio is True


def test_probe_rejects_a_playlist_disguised_as_mp4(tmp_path):
    evil = tmp_path / "evil.mp4"
    evil.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n")
    with pytest.raises(ScreentimeError) as err:
        media.probe(evil)
    assert err.value.code == "bad_input"
    for call in (lambda: media.input_args(evil), lambda: media.frame_jpeg(evil, 0),
                 lambda: media.extract_audio_wav(evil, tmp_path / "a.wav"),
                 lambda: media.normalize(evil, tmp_path / "n.mp4")):
        with pytest.raises(ScreentimeError) as err:
            call()
        assert err.value.code == "bad_input"
    assert not (tmp_path / "n.mp4").exists()


@pytest.mark.parametrize(
    "name,content,fmt",
    [
        ("list.m3u8", "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n", "hls"),
        ("concat.mp4", "ffconcat version 1.0\nfile /etc/hosts\n", "concat"),
    ],
)
def test_probe_names_the_refused_format(tmp_path, name, content, fmt):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(ScreentimeError) as err:
        media.probe(path)
    assert err.value.message == f"That file is not a video Screen-Time can read (format {fmt})."


def test_probe_refuses_an_image(tmp_path):
    png = tmp_path / "still.mp4"
    Image.new("RGB", (32, 32), "red").save(png, format="PNG")
    with pytest.raises(ScreentimeError) as err:
        media.probe(png)
    assert err.value.code == "bad_input"
    assert "not a video" in err.value.message


def test_probe_refuses_missing_files_directories_and_fifos(tmp_path):
    fifo = tmp_path / "pipe.mp4"
    os.mkfifo(fifo)
    for path in (tmp_path / "nope.mp4", tmp_path, fifo):
        with pytest.raises(ScreentimeError) as err:
            media.probe(path)
        assert err.value.code == "bad_input"


def test_input_args_shape_and_a_real_mp4_passes(clips):
    args = media.input_args(clips["odd"])
    assert args == ["-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(clips["odd"].resolve())]


def test_input_args_rechecks_a_file_that_changed(tmp_path, clips):
    path = tmp_path / "swap.mp4"
    shutil.copy(clips["odd"], path)
    assert media.input_args(path)[3] == "mov"
    path.write_text("ffconcat version 1.0\nfile /etc/hosts\n")
    os.utime(path, ns=(1, 1))  # a different mtime and size: the cached verdict must not be reused
    with pytest.raises(ScreentimeError):
        media.input_args(path)


# --------------------------------------------------------------------------------------------------------------------
# Normalize


@pytest.mark.parametrize(
    "size,kwargs,expected",
    [
        ((1080, 1920), {}, (540, 960)),  # the app: longest edge 960
        ((1920, 1080), {}, (960, 540)),
        ((641, 361), {}, (640, 360)),  # even, never upscaled
        ((361, 641), {}, (360, 640)),
        ((200, 100), {}, (200, 100)),
        ((1080, 1920), {"max_edge": None, "max_height": 720}, (406, 720)),
        ((1080, 1920), {"max_edge": 960, "max_height": 720}, (406, 720)),
        ((1080, 1920), {"max_edge": None}, (1080, 1920)),
        ((3, 3), {}, (2, 2)),
    ],
)
def test_target_size(size, kwargs, expected):
    assert media.target_size(*size, **kwargs) == expected


def test_normalize_odd_rotated_clip(normalized):
    out, info = normalized
    assert info.fps == 15
    assert (info.width, info.height) == (360, 640)  # autorotated to portrait, even
    assert info.width % 2 == 0 and info.height % 2 == 0
    assert info.has_audio is False
    assert info.frame_count == 45 == round(info.duration_sec * 15)
    data = ffprobe_json(out)
    streams = data["streams"]
    assert [s["codec_type"] for s in streams] == ["video"]  # no audio
    video = streams[0]
    assert video["r_frame_rate"] == "15/1" and video["avg_frame_rate"] == "15/1"
    assert video["pix_fmt"] == "yuv420p"
    assert video.get("sample_aspect_ratio", "1:1") == "1:1"
    assert not video.get("side_data_list")  # rotation applied, not carried
    assert int(video["nb_frames"]) == 45
    raw = out.read_bytes()
    assert raw.index(b"moov") < raw.index(b"mdat")  # faststart


def test_normalize_regenerates_timestamps(normalized):
    out, _ = normalized
    pts = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0",
         str(out)], check=True, capture_output=True, timeout=30,
    ).stdout.decode().split()
    # ffprobe prints pts_time to 6 decimals; frame i must sit at i / 15 s (a frame is 0.0667 s).
    assert [float(t.split(",")[0]) * 15 for t in pts] == pytest.approx(list(range(45)), abs=1e-3)


def test_normalize_max_height(clips, tmp_path):
    info = media.normalize(clips["odd"], tmp_path / "small.mp4", max_height=180, max_edge=None)
    assert (info.width, info.height) == (320, 180)


def test_normalize_cuts_a_window(clips, tmp_path):
    info = media.normalize(clips["odd"], tmp_path / "cut.mp4", start_sec=1.0, duration_sec=1.0)
    assert info.frame_count == 15


@pytest.mark.parametrize("encoder", media.ENCODER_CHAIN)
def test_each_encoder_in_the_chain(clips, tmp_path, encoder):
    if encoder not in media.encoders():
        pytest.skip(f"{encoder} not in this ffmpeg build")
    out = tmp_path / f"{encoder}.mp4"
    info = media.normalize(clips["odd"], out, encoder=encoder)
    assert info.frame_count == 45 and info.fps == 15
    codec = ffprobe_json(out)["streams"][0]["codec_name"]
    assert codec == ("mpeg4" if encoder == "mpeg4" else "h264")


def test_falls_back_to_the_next_encoder(clips, tmp_path, monkeypatch):
    real = media._encoder_args

    def broken_x264(name, *args):
        if name == "libx264":
            return ["-c:v", "libx264", "-preset", "no-such-preset"]
        return real(name, *args)

    monkeypatch.setattr(media, "_encoder_args", broken_x264)
    monkeypatch.setattr(media, "video_encoders", lambda: ["not-an-encoder", "libx264", "mpeg4"])
    out = tmp_path / "fallback.mp4"
    info = media.normalize(clips["odd"], out)
    assert info.frame_count == 45
    assert ffprobe_json(out)["streams"][0]["codec_name"] == "mpeg4"
    assert not list(tmp_path.glob(".*partial"))


def test_all_encoders_failing_is_bad_input(clips, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "_encoder_args", lambda name, *a: ["-c:v", "libx264", "-preset", "no-such-preset"])
    monkeypatch.setattr(media, "video_encoders", lambda: ["libx264", "mpeg4"])
    with pytest.raises(ScreentimeError) as err:
        media.normalize(clips["odd"], tmp_path / "x.mp4")
    assert err.value.code == "bad_input"
    assert "libx264" in err.value.message and "mpeg4" in err.value.message
    assert not (tmp_path / "x.mp4").exists()


def test_a_stuck_ffmpeg_is_killed_at_its_time_limit(clips, tmp_path):
    with pytest.raises(ScreentimeError) as err:
        media.normalize(clips["odd"], tmp_path / "slow.mp4", timeout=0.001)
    assert err.value.code == "timeout"


# --------------------------------------------------------------------------------------------------------------------
# Frames and crops


def test_frame_jpeg_returns_exactly_the_frame_asked_for(coded15):
    """Source frame N has luma 16 + 2N at 30 fps, so frame k of the 15 fps video must read 16 + 4k."""
    assert media.probe(coded15).frame_count == 45
    for k in range(45):
        assert luma(media.frame_jpeg(coded15, k)) == pytest.approx(16 + 4 * k, abs=1.5), f"frame {k}"


def test_frame_jpeg_matches_ffmpeg_decoding_from_the_start(normalized):
    """Same pixels as ffmpeg's own select=eq(n,k), which counts decoded frames from the first one."""
    out, _ = normalized
    for k in (0, 1, 7, 22, 44):
        ours = Image.open(io.BytesIO(media.frame_jpeg(out, k))).convert("RGB")
        ref = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(out), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",
             "-q:v", "3", "-c:v", "mjpeg", "-f", "image2pipe", "pipe:1"],
            check=True, capture_output=True, timeout=30,
        ).stdout
        assert ours.tobytes() == Image.open(io.BytesIO(ref)).convert("RGB").tobytes(), f"frame {k}"
    # Negative control: neighbouring frames differ, so the comparison can tell frames apart.
    a = Image.open(io.BytesIO(media.frame_jpeg(out, 7))).convert("RGB").tobytes()
    b = Image.open(io.BytesIO(media.frame_jpeg(out, 8))).convert("RGB").tobytes()
    assert a != b


def test_frame_jpeg_past_the_end(coded15):
    with pytest.raises(ScreentimeError) as err:
        media.frame_jpeg(coded15, 45)
    assert err.value.code == "bad_input"
    assert "past the end" in err.value.message


@pytest.mark.parametrize("bad", [-1, 1.0, True, "3"])
def test_frame_jpeg_rejects_bad_indices(coded15, bad):
    with pytest.raises(ScreentimeError):
        media.frame_jpeg(coded15, bad)


def test_frame_jpeg_max_side(normalized):
    out, _ = normalized
    img = Image.open(io.BytesIO(media.frame_jpeg(out, 3, max_side=100)))
    assert max(img.size) == 100
    assert img.size[1] > img.size[0]  # still portrait
    full = Image.open(io.BytesIO(media.frame_jpeg(out, 3, max_side=5000)))
    assert full.size == (360, 640)  # never upscaled


def _jpeg(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("box", [Box(10, 20, 19, 29), {"x1": 10, "y1": 20, "x2": 19, "y2": 29}, (10, 20, 19, 29)])
def test_crop_jpeg_pads_like_the_app(box):
    # 10x10 box, 15% padding: round(1.5) = 2 px per side (JavaScript rounding), so 14x14.
    crop = Image.open(io.BytesIO(media.crop_jpeg(_jpeg(100, 50), box)))
    assert crop.size == (14, 14)
    assert media.padded_crop(box, 0.15, 100, 50) == (8, 18, 21, 31)


def test_crop_jpeg_clamps_to_the_image():
    assert media.padded_crop(Box(0, 0, 99, 49), 0.15, 100, 50) == (0, 0, 99, 49)
    crop = Image.open(io.BytesIO(media.crop_jpeg(_jpeg(100, 50), Box(90, 40, 99, 49))))
    assert crop.size == (12, 12)  # grows left and up only
    with pytest.raises(ScreentimeError) as err:
        media.crop_jpeg(_jpeg(100, 50), Box(150, 80, 160, 90), pad=0)
    assert err.value.code == "bad_input"


def test_crop_jpeg_rejects_non_images():
    with pytest.raises(ScreentimeError) as err:
        media.crop_jpeg(b"not a jpeg", Box(0, 0, 1, 1))
    assert err.value.code == "bad_input"


# --------------------------------------------------------------------------------------------------------------------
# Audio and files


def test_extract_audio_wav(clips, tmp_path):
    out = tmp_path / "a.wav"
    assert media.extract_audio_wav(clips["rotated"], out) is True
    with wave.open(str(out), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() / 16000 == pytest.approx(3.0, abs=0.05)
    assert media.wav_duration_sec(out) == pytest.approx(3.0, abs=0.05)


def test_extract_audio_wav_window(clips, tmp_path):
    out = tmp_path / "w.wav"
    assert media.extract_audio_wav(clips["odd"], out, start_sec=1.0, duration_sec=1.5) is True
    assert media.wav_duration_sec(out) == pytest.approx(1.5, abs=0.03)


def test_no_audio_stream_is_false_and_writes_nothing(normalized, tmp_path):
    out, _ = normalized
    target = tmp_path / "none.wav"
    assert media.extract_audio_wav(out, target) is False
    assert not target.exists()


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"screen-time" * 100_000)
    assert media.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
