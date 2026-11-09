"""Proof reel: frame-exact boxes, the strip, size, audio and failure modes. Inputs are made with ffmpeg lavfi.

The synthetic clip has a blue "product" and a purple "look-alike" painted at the detection boxes, so the tests find
where the video landed in the reel by colour and check the drawn boxes against the picture itself, not against the
module's own layout maths.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from screentime_core import proof
from screentime_core.errors import ScreentimeError
from screentime_core.models import Box, Detection, Segment, VideoInfo, VideoMetrics

FPS = 15
W, H, N = 640, 360, 45  # 3 s
PRODUCT = (100, 80, 219, 239)  # inclusive pixel box of the blue product, tracked on frames 5-20
LOOKALIKE = (400, 100, 499, 279)  # the purple look-alike, tracked on frames 10-30
MATCH_FRAMES = list(range(5, 21))
LOOK_FRAMES = list(range(10, 31))
MARKER = (150, 140, 169, 159)  # white square inside the product, painted on frames 5-20 only: picture/box sync

# Colour classes as (min, max) per R, G, B, loose enough for H.264 at CRF 23.
LIME = ((100, 225), (180, 255), (0, 125))
RED = ((180, 255), (0, 125), (0, 125))
AMBER = ((200, 255), (115, 200), (0, 90))
BLUE = ((0, 95), (40, 145), (165, 255))
PURPLE = ((95, 185), (15, 115), (135, 235))
WHITE = ((235, 255), (235, 255), (235, 255))


# --------------------------------------------------------------------------------------------------------------------
# Helpers


def mask(img: Image.Image, ranges) -> Image.Image:
    """255 where the pixel is in the colour class, else 0."""
    out = None
    for ch, (lo, hi) in zip(img.split(), ranges, strict=True):
        m = ch.point(lambda v, lo=lo, hi=hi: 255 if lo <= v <= hi else 0)
        out = m if out is None else ImageChops.multiply(out, m)
    return out


def bbox(m: Image.Image) -> tuple[int, int, int, int]:
    b = m.getbbox()
    assert b is not None, "colour not found"
    return (b[0], b[1], b[2] - 1, b[3] - 1)


def edge_hits(m: Image.Image, rect, t: int = 6, band: int = 2) -> float:
    """Share of positions along the four edges of ``rect`` where a band ``t`` px into the box (+-band) has the colour.
    A solid box scores about 1, a dashed one about its dash share, no box 0. Corners are left out."""
    x0, y0, x1, y1 = rect
    hits = total = 0
    for y in range(y0 + t + 2, y1 - t - 1):
        for xa, xb in ((x0 - band, x0 + t + band), (x1 - t - band, x1 + band)):
            total += 1
            hits += m.crop((xa, y, xb + 1, y + 1)).getbbox() is not None
    for x in range(x0 + t + 2, x1 - t - 1):
        for ya, yb in ((y0 - band, y0 + t + band), (y1 - t - band, y1 + band)):
            total += 1
            hits += m.crop((x, ya, x + 1, yb + 1)).getbbox() is not None
    return hits / total


def det(frame: int, oid: int, rect) -> Detection:
    x0, y0, x1, y1 = rect
    return Detection(frame=frame, object_id=oid, box=Box(x0, y0, x1, y1), area_fraction=0.08, centroid_x=0.5,
                     centroid_y=0.5)


def metrics(visible=()) -> VideoMetrics:
    visible = list(visible)
    secs = len(visible) / FPS
    first = visible[0] / FPS if visible else None
    return VideoMetrics(seconds_on_screen=secs, share_of_video=secs / (N / FPS), first_seen_sec=first,
                        longest_run_sec=secs, longest_run_start_sec=first, avg_area_fraction=0.08,
                        peak_area_fraction=0.08, peak_area_sec=first,
                        segments=[Segment(visible[0] / FPS, (visible[-1] + 1) / FPS)] if visible else [],
                        visible_frames=visible)


def info() -> VideoInfo:
    return VideoInfo(width=W, height=H, frame_count=N, duration_sec=N / FPS, fps=FPS, has_audio=False)


def scene() -> tuple[list[Detection], dict[str, str], VideoMetrics]:
    dets = [det(f, 1, PRODUCT) for f in MATCH_FRAMES] + [det(f, 2, LOOKALIKE) for f in LOOK_FRAMES]
    return dets, {"1": "match", "2": "lookalike"}, metrics(MATCH_FRAMES)


def ffmpeg_or_skip() -> SimpleNamespace:
    try:
        ff = proof._ffmpeg_bin()
    except ScreentimeError:
        pytest.skip("ffmpeg is not installed")
    fp = Path(ff).with_name("ffprobe")
    ffprobe = str(fp) if fp.exists() else shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe is not installed")
    return SimpleNamespace(ffmpeg=ff, ffprobe=ffprobe)


def run(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


def probe(m, path: Path) -> dict:
    entries = "stream=codec_type,codec_name,width,height,nb_read_frames,r_frame_rate,pix_fmt:format=duration"
    out = subprocess.run([m.ffprobe, "-v", "error", "-count_frames", "-show_entries", entries, "-of", "json",
                          str(path)], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    data["by_type"] = {s["codec_type"]: s for s in data["streams"]}
    return data


def frames(m, path: Path, wanted: list[int]) -> dict[int, Image.Image]:
    """Decoded frames of ``path`` by index (select on the decoder's frame number, nothing dropped or doubled)."""
    v = probe(m, path)["by_type"]["video"]
    w, h = int(v["width"]), int(v["height"])
    order = sorted(wanted)
    expr = "+".join(f"eq(n\\,{i})" for i in order)
    raw = subprocess.run([m.ffmpeg, "-v", "error", "-i", str(path), "-vf", f"select={expr}",
                          *proof._passthrough(m.ffmpeg), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, check=True).stdout
    size = w * h * 3
    assert len(raw) == size * len(order)
    return {i: Image.frombytes("RGB", (w, h), raw[k * size:(k + 1) * size]) for k, i in enumerate(order)}


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    m = ffmpeg_or_skip()
    d = tmp_path_factory.mktemp("proof-media")
    paint = (f"drawbox=x={PRODUCT[0]}:y={PRODUCT[1]}:w={PRODUCT[2] - PRODUCT[0] + 1}:h={PRODUCT[3] - PRODUCT[1] + 1}"
             f":color=0x285adc:t=fill,drawbox=x={LOOKALIKE[0]}:y={LOOKALIKE[1]}:w={LOOKALIKE[2] - LOOKALIKE[0] + 1}"
             f":h={LOOKALIKE[3] - LOOKALIKE[1] + 1}:color=0x8c3cbe:t=fill,drawbox=x={MARKER[0]}:y={MARKER[1]}"
             f":w=20:h=20:color=white:t=fill:enable='between(n,{MATCH_FRAMES[0]},{MATCH_FRAMES[-1]})'")
    m.clip = d / "clip.mp4"  # what normalize() leaves: video only, 15 fps
    run([m.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x464646:s={W}x{H}:r={FPS}:d=3", "-vf", paint,
         "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(m.clip)])
    m.upload = d / "upload.mp4"  # an original upload with sound, 5 s: longer than the clip
    run([m.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
         str(m.upload)])
    m.short_audio = d / "short.m4a"  # audio only, 1 s: shorter than the clip
    run([m.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=660:duration=1", "-c:a", "aac",
         str(m.short_audio)])
    return m


def render(m, out: Path, **kw) -> Path:
    dets, verdicts, met = scene()
    kw.setdefault("product_label", "Test can")
    return Path(proof.render_proof(str(m.clip), dets, verdicts, met, info(), str(out), **kw))


@pytest.fixture(scope="module")
def reels(media, tmp_path_factory):
    d = tmp_path_factory.mktemp("proof-reels")
    return {mh: render(media, d / f"reel-{mh}.mp4", max_height=mh) for mh in (720, 300)}


# --------------------------------------------------------------------------------------------------------------------
# End to end (ffmpeg)


@pytest.mark.parametrize("max_height", [720, 300])
def test_every_frame_kept_and_boxes_on_their_own_frames(media, reels, max_height):
    out = reels[max_height]
    src, got = probe(media, media.clip), probe(media, out)
    v = got["by_type"]["video"]
    assert int(v["nb_read_frames"]) == int(src["by_type"]["video"]["nb_read_frames"]) == N
    assert v["r_frame_rate"] == "15/1" and v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    assert abs(float(got["format"]["duration"]) - float(src["format"]["duration"])) < 1 / FPS
    assert int(v["height"]) <= max_height and int(v["width"]) % 2 == 0 and int(v["height"]) % 2 == 0
    assert "audio" not in got["by_type"]

    want = [0, 4, 5, 8, 9, 10, 20, 21, 25, 30, 31, 40]
    fr = frames(media, out, want)
    # Where the picture landed (frame 40 has no boxes), found by colour: the boxes must sit on these.
    product = bbox(mask(fr[40], BLUE))
    lookalike = bbox(mask(fr[40], PURPLE))
    if max_height == 720:  # not scaled: the picture keeps its size under the title band
        assert product[2] - product[0] == PRODUCT[2] - PRODUCT[0] and product[0] == PRODUCT[0]
    else:
        assert product[2] - product[0] < PRODUCT[2] - PRODUCT[0] - 20
    sx = (product[2] - product[0] + 1) / (PRODUCT[2] - PRODUCT[0] + 1)
    sy = (product[3] - product[1] + 1) / (PRODUCT[3] - PRODUCT[1] + 1)
    marker_at = (product[0] + round((MARKER[0] + 10 - PRODUCT[0]) * sx),
                 product[1] + round((MARKER[1] + 10 - PRODUCT[1]) * sy))
    for i in want:
        lime = edge_hits(mask(fr[i], LIME), product)
        red = edge_hits(mask(fr[i], RED), lookalike)
        # The picture under the boxes is the same frame: the marker shows exactly when the lime box does.
        assert (mask(fr[i], WHITE).getpixel(marker_at) == 255) == (i in MATCH_FRAMES), i
        if i in MATCH_FRAMES:
            assert lime > 0.9, (i, lime)
        else:
            assert lime < 0.02, (i, lime)
        if i in LOOK_FRAMES:
            assert 0.35 < red < 0.95, (i, red)  # dashed
        else:
            assert red < 0.02, (i, red)
        assert edge_hits(mask(fr[i], LIME), lookalike) < 0.02 and edge_hits(mask(fr[i], RED), product) < 0.02


@pytest.mark.parametrize("encoder", proof.ENCODERS)
def test_each_fallback_encoder_keeps_frames_and_boxes(media, tmp_path, monkeypatch, encoder):
    if not proof._encoder_works(media.ffmpeg, encoder):
        pytest.skip(f"{encoder} is not usable in this ffmpeg build")
    monkeypatch.setattr(proof, "_pick_encoder", lambda ffmpeg: encoder)
    out = render(media, tmp_path / f"{encoder}.mp4")
    v = probe(media, out)["by_type"]["video"]
    assert int(v["nb_read_frames"]) == N and v["codec_name"] == ("mpeg4" if encoder == "mpeg4" else "h264")
    fr = frames(media, out, [8, 40])
    product = bbox(mask(fr[40], BLUE))
    assert edge_hits(mask(fr[8], LIME), product) > 0.8 and edge_hits(mask(fr[40], LIME), product) < 0.02


@pytest.fixture
def passes(monkeypatch) -> list[tuple[int, int]]:
    """(crf, kbps) of every encoding pass render_proof runs."""
    seen: list[tuple[int, int]] = []
    real = proof._Job.run

    def spy(self, *, crf, kbps):
        seen.append((crf, kbps))
        return real(self, crf=crf, kbps=kbps)

    monkeypatch.setattr(proof._Job, "run", spy)
    return seen


def test_under_the_size_cap_in_one_pass(media, tmp_path, passes):
    out = render(media, tmp_path / "reel.mp4", max_mb=19.0)
    assert out.stat().st_size < 19.0 * 1_000_000
    assert len(passes) == 1


def test_over_the_size_cap_retries_once_smaller(media, tmp_path, passes):
    out = render(media, tmp_path / "tiny.mp4", max_mb=0.001)  # 1 kB: no 3 s video fits, so the retry must run
    assert len(passes) == 2
    (crf1, kbps1), (crf2, kbps2) = passes
    assert crf2 > crf1 and kbps2 < kbps1
    assert int(probe(media, out)["by_type"]["video"]["nb_read_frames"]) == N


@pytest.mark.parametrize("source", ["upload", "short_audio"])
def test_original_audio_is_muxed_without_changing_frames(media, tmp_path, source):
    out = render(media, tmp_path / "reel.mp4", audio_source=str(getattr(media, source)))
    got = probe(media, out)
    assert got["by_type"]["audio"]["codec_name"] == "aac"
    assert int(got["by_type"]["video"]["nb_read_frames"]) == N
    # Trimmed (5 s source) or padded (1 s source) to the video: the reel lasts as long as the clip.
    assert abs(float(got["format"]["duration"]) - N / FPS) < 0.1


def test_audio_source_without_audio_gives_a_silent_reel(media, tmp_path):
    out = render(media, tmp_path / "reel.mp4", audio_source=str(media.clip))
    got = probe(media, out)
    assert "audio" not in got["by_type"] and int(got["by_type"]["video"]["nb_read_frames"]) == N


def test_temp_files_are_removed(media, tmp_path):
    work, out_dir = tmp_path / "work", tmp_path / "out"
    render(media, out_dir / "proof.mp4", workdir=str(work))
    assert [p.name for p in out_dir.iterdir()] == ["proof.mp4"]
    assert list(work.iterdir()) == []


def test_every_ffmpeg_input_is_local_only(media, tmp_path, monkeypatch):
    proof._pick_encoder.cache_clear()
    proof._passthrough.cache_clear()
    seen: list[list[str]] = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def spy_run(args, *a, **k):
        seen.append(list(args))
        return real_run(args, *a, **k)

    class SpyPopen(real_popen):
        def __init__(self, args, *a, **k):
            seen.append(list(args))
            super().__init__(args, *a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(proof.subprocess, "run", spy_run)
        mp.setattr(proof.subprocess, "Popen", SpyPopen)
        render(media, tmp_path / "reel.mp4", audio_source=str(media.upload))
    inputs = [(args, i) for args in seen for i, a in enumerate(args) if a == "-i"]
    assert {args[i + 1] for args, i in inputs} >= {str(media.clip), str(media.upload), "pipe:0"}
    for args, i in inputs:
        assert args[i - 2:i] == ["-protocol_whitelist", "file,pipe"], args


def test_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(proof.shutil, "which", lambda name: None)
    monkeypatch.setattr(proof, "_FFMPEG_DIRS", ())
    with pytest.raises(ScreentimeError) as e:
        proof.render_proof(str(tmp_path / "clip.mp4"), [], {}, metrics(), info(), str(tmp_path / "o.mp4"),
                           product_label="x")
    assert e.value.code == "ffmpeg_missing"


def test_unwritable_destination_is_a_screentime_error(media, tmp_path):
    (tmp_path / "file").write_text("in the way")
    with pytest.raises(ScreentimeError) as e:
        render(media, tmp_path / "file" / "proof.mp4")
    assert e.value.code == "incomplete"


@pytest.mark.parametrize("video", ["missing.mp4", "http://127.0.0.1:9/clip.mp4", "not-a-video.mp4"])
def test_unreadable_video_is_bad_input(media, tmp_path, video):
    (tmp_path / "not-a-video.mp4").write_text("hello")
    path = video if video.startswith("http") else str(tmp_path / video)
    with pytest.raises(ScreentimeError) as e:
        proof.render_proof(path, [], {}, metrics(), info(), str(tmp_path / "o.mp4"), product_label="x")
    assert e.value.code == "bad_input"
    assert not (tmp_path / "o.mp4").exists() and not list(tmp_path.glob(".o.mp4.*"))


def test_missing_audio_source_is_bad_input(media, tmp_path):
    with pytest.raises(ScreentimeError) as e:
        render(media, tmp_path / "o.mp4", audio_source=str(tmp_path / "gone.mp4"))
    assert e.value.code == "bad_input"


# --------------------------------------------------------------------------------------------------------------------
# Drawing and layout (no ffmpeg)


def overlay(dets, verdicts, visible=(), label="Test can") -> proof._Overlay:
    return proof._Overlay(info(), float(FPS), dets, verdicts, metrics(visible), label, 720)


def composed(ov: proof._Overlay, i: int) -> Image.Image:
    L = ov.layout
    raw = bytes([70]) * (L.width * L.video_h * 3)
    return Image.frombytes("RGB", (L.width, L.height), ov.compose(raw, i))


def on_canvas(ov: proof._Overlay, rect) -> tuple[int, int, int, int]:
    assert (ov.layout.width, ov.layout.video_h) == (W, H)  # 640x360 under 720: no scaling
    y = ov.layout.video_y
    return (rect[0], rect[1] + y, rect[2], rect[3] + y)


def test_unknown_or_missing_verdicts_are_drawn_as_unclear():
    ov = overlay([det(0, 7, PRODUCT), det(0, 8, LOOKALIKE)], {"7": "banana"})  # 8 is not in the map
    img = composed(ov, 0)
    for rect in (PRODUCT, LOOKALIKE):
        r = on_canvas(ov, rect)
        assert edge_hits(mask(img, AMBER), r) > 0.35
        assert edge_hits(mask(img, LIME), r) == 0 and edge_hits(mask(img, RED), r) == 0


def test_boxes_outside_the_frame_are_clamped_or_skipped():
    dets = [det(0, 1, (-50, -40, 60, 50)), det(0, 2, (700, 10, 800, 50)), det(0, 3, (math.nan, 0, 10, 10)),
            det(-3, 4, PRODUCT)]
    ov = overlay(dets, {"1": "match", "2": "match", "3": "match", "4": "match"}, visible=[0])
    assert [oid for _, _, oid in ov.by_frame[0]] == [1]
    img = composed(ov, 0)
    lime = mask(img, LIME)
    y = ov.layout.video_y
    assert ov.by_frame[0][0][0] == (0, 0, 60, 50)
    assert lime.crop((0, y + 12, 3, y + 40)).getbbox() is not None  # left edge, on the frame's border
    assert lime.crop((56, y + 12, 61, y + 40)).getbbox() is not None  # right edge, where the box really ends


def test_a_match_on_an_uncounted_frame_is_not_called_counted():
    ov = overlay([det(3, 1, PRODUCT), det(4, 1, PRODUCT)], {"1": "match"}, visible=[4])
    assert ov.by_frame[3][0][1] is proof.UNCOUNTED and ov.by_frame[4][0][1] is proof.COUNTED
    img = composed(ov, 3)
    assert 0.35 < edge_hits(mask(img, LIME), on_canvas(ov, PRODUCT)) < 0.95  # dashed, not solid


def test_a_crowded_frame_keeps_every_box_but_at_most_six_tags(monkeypatch):
    boxes = [(10 + 62 * k, 120, 50 + 62 * k, 220) for k in range(10)]
    ov = overlay([det(0, k, b) for k, b in enumerate(boxes)], {str(k): "lookalike" for k in range(10)})
    tagged = []
    real = ov._place
    monkeypatch.setattr(ov, "_place", lambda b, s: tagged.append(len(b)) or real(b, s))
    red = mask(composed(ov, 0), RED)
    assert tagged == [proof.MAX_TAGS]
    assert all(edge_hits(red, on_canvas(ov, b), t=4) > 0.3 for b in boxes)


def test_missing_total_falls_back_to_the_counted_frames():
    m = metrics(MATCH_FRAMES)
    m.seconds_on_screen = None
    ov = proof._Overlay(info(), float(FPS), [], {}, m, "Test can", 720)
    assert any(text == "1.0 s" for line in ov.layout.title for text, _ in line)


def test_strip_and_playhead_on_frames_without_boxes():
    ov = overlay([det(f, 1, PRODUCT) for f in MATCH_FRAMES], {"1": "match"}, visible=MATCH_FRAMES)
    L = ov.layout
    bar_y = L.band_y + ov.bar_y + L.sizes.bar_h // 2

    def slot_x(f: int) -> int:
        return ov.bar_x + int(ov.bar_w * (f + 0.5) / N)

    def playhead_x(img: Image.Image) -> int:
        white = mask(img, WHITE).crop((0, bar_y, L.width, bar_y + 1))
        return bbox(white)[0]

    first, last = composed(ov, 0), composed(ov, N - 1)
    for img in (first, last):
        lime = mask(img, LIME)
        assert all(lime.getpixel((slot_x(f), bar_y)) == 255 for f in (5, 12, 20))
        assert all(lime.getpixel((slot_x(f), bar_y)) == 0 for f in (2, 21, 30))
    assert playhead_x(first) < slot_x(1) and playhead_x(last) > slot_x(N - 2)


def test_counter_only_moves_on_counted_frames():
    ov = overlay([det(f, 1, PRODUCT) for f in MATCH_FRAMES], {"1": "match"}, visible=MATCH_FRAMES)
    L = ov.layout
    # The counter text: left of the legend, above the playhead (which reaches a little above the bar).
    line = (0, L.band_y, L.width // 2, L.band_y + ov.bar_y - L.sizes.playhead_ext - 2)

    def counter(i: int) -> bytes:
        return composed(ov, i).crop(line).tobytes()

    assert counter(0) == counter(4)  # nothing counted yet
    assert counter(4) != counter(5)  # first counted frame: lime number
    assert counter(21) == counter(40)  # after the last counted frame the total holds


@pytest.mark.parametrize("max_height", [720, 300])
@pytest.mark.parametrize("w,h", [(540, 960), (960, 540), (960, 960), (641, 359), (64, 64), (1080, 1920)])
def test_layout_fits_the_cap_without_upscaling(w, h, max_height):
    ov = proof._Overlay(VideoInfo(width=w, height=h, frame_count=330, duration_sec=22.0), float(FPS), [], {},
                        metrics(), "LaCroix Strawberry Peach can", max_height)
    L = ov.layout
    assert L.width % 2 == 0 and L.height % 2 == 0 and L.height <= max_height
    assert L.video_h <= h and L.width <= w + 1
    assert abs(L.width / L.video_h - w / h) / (w / h) < 0.03
    assert L.video_y + L.video_h == L.band_y and L.band_y + L.band_h == L.height
    # Text stays inside the reel however narrow it gets.
    assert all(proof._runs_width(line, L.title_font) <= L.width - 2 * L.sizes.pad for line in L.title)
    assert proof._runs_width(ov._counter(ov.slots, True), L.count_font) <= ov.bar_w


def test_title_is_one_line_when_it_fits_and_wraps_when_not():
    f = proof._Font(17)
    assert len(proof._title_lines("Stanley tumbler", "7.9", f, 600)) == 1
    two = proof._title_lines("LaCroix Strawberry Peach can", "21.7", f, 300)
    assert [t for t, _ in two[1]] == ["21.7 s", " on screen"] and two[0][0][0] == "LaCroix Strawberry Peach can"
    assert proof._title_lines("", "0.0", f, 300) == [[("0.0 s", proof.LIME), (" on screen", proof.WHITE)]]


def test_seconds_round_down_to_tenths():
    assert [proof._tenths(x) for x in (7.96, 326 / 15, 16 / 15, 0.1 * 3, 0.0, -1.0)] == [
        "7.9", "21.7", "1.0", "0.3", "0.0", "0.0"]


def test_dashes_start_and_end_on_the_corners():
    spans = proof._dash_spans(10, 209, 14, 8)
    assert spans[0][0] == 10 and spans[-1][1] == 209
    assert all(b - a + 1 == 14 for a, b in spans)
    assert all(nxt[0] - cur[1] > 1 for cur, nxt in zip(spans, spans[1:], strict=False))
    assert proof._dash_spans(0, 20, 14, 8) == [(0, 20)]  # too short for a pattern: solid


def test_label_is_cleaned_to_one_drawable_line():
    assert proof._clean_label("  LaCroix\u2014Peach \n can \U0001f964 ") == "LaCroix-Peach can"
    assert proof._clean_label(None) == ""
