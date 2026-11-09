"""Teardown and Compare cards: size, content, readable text, and graceful handling of a moved video."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from screentime_core import card, teardown
from screentime_core.models import CompareReport, TeardownReport

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe on PATH")

SIZE = "320x240"
FIRST_STRIP = (card.PAD + 6, card.OVERLINE_Y + card.TITLE_STEP + card.AFTER_TITLE + card.LABEL_H + 12)


def _ffmpeg(out: Path, *args: str) -> Path:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args, str(out)], check=True, timeout=120)
    return out


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> tuple[TeardownReport, TeardownReport, Path]:
    d = tmp_path_factory.mktemp("cardclips")
    draft = _ffmpeg(
        d / "draft.mp4",
        "-f", "lavfi", "-t", "1.2", "-i", f"testsrc2=size={SIZE}:rate=30",
        "-f", "lavfi", "-t", "0.9", "-i", f"color=c=red:size={SIZE}:rate=30",
        "-f", "lavfi", "-t", "2.1", "-i", f"smptebars=size={SIZE}:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=0.6", "-f", "lavfi", "-i", "sine=f=440:r=48000:d=3.6",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v];[3:a][4:a]concat=n=2:v=0:a=1[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac", "-b:a", "96k",
    )
    ref = _ffmpeg(
        d / "ref.mp4", "-f", "lavfi", "-t", "5.5", "-i", f"testsrc2=size={SIZE}:rate=30", "-c:v", "mpeg4", "-q:v", "3"
    )
    work = d / "work"
    yours = teardown.analyze(draft, work, product_first_seen_sec=1.4, brand_first_said_sec=0.9)
    theirs = teardown.analyze(ref, work)
    return yours, theirs, work


def _check_png(path: str) -> Image.Image:
    img = Image.open(path)
    assert img.size == (1080, 1350) and img.mode == "RGB"
    colors = img.getcolors(maxcolors=1_000_000)
    assert colors is not None and len(colors) > 500  # thumbnails, lines and text, not a blank canvas
    return img


def test_compare_card_renders(reports: tuple[TeardownReport, TeardownReport, Path], tmp_path: Path) -> None:
    yours, theirs, work = reports
    c: CompareReport = teardown.compare(yours, theirs)
    out = card.render_compare_card(c, tmp_path / "cards" / "compare.png", workdir=work)
    assert c.card_path == out and Path(out).stat().st_size > 20_000
    img = _check_png(out)
    assert img.getpixel(FIRST_STRIP) not in (card.PLACEHOLDER, card.BG)  # a real thumbnail in the first strip


def test_teardown_card_renders_without_a_workdir(
    reports: tuple[TeardownReport, TeardownReport, Path], tmp_path: Path
) -> None:
    yours = copy.deepcopy(reports[0])
    out = card.render_teardown_card(yours, tmp_path / "teardown.png", label="Draft")
    assert yours.card_path == out
    img = _check_png(out)
    # thumbnails came from a temporary folder that must outlive the drawing
    assert img.getpixel(FIRST_STRIP) not in (card.PLACEHOLDER, card.BG)


def test_card_draws_placeholders_when_the_video_has_moved(
    reports: tuple[TeardownReport, TeardownReport, Path], tmp_path: Path
) -> None:
    gone = copy.deepcopy(reports[1])
    gone.video_path = str(tmp_path / "moved-away.mp4")
    assert teardown.extract_shot_frames(gone, tmp_path) == [None] * len(gone.shots)
    out = card.render_compare_card(teardown.compare(gone, gone), tmp_path / "gone.png")
    img = _check_png(out)
    assert img.getpixel(FIRST_STRIP) == card.PLACEHOLDER


def test_no_text_under_22_px(
    reports: tuple[TeardownReport, TeardownReport, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []
    real = card._font

    def spy(size: int) -> object:
        sizes.append(size)
        return real(size)

    monkeypatch.setattr(card, "_font", spy)
    yours, theirs, work = reports
    card.render_compare_card(teardown.compare(yours, theirs), tmp_path / "c.png", yours_label="Take A",
                             theirs_label="Take B", workdir=work)
    card.render_teardown_card(copy.deepcopy(yours), tmp_path / "t.png", workdir=work)
    assert sizes and min(sizes) >= card.MIN_TEXT_PX
    with pytest.raises(ValueError):
        real(card.MIN_TEXT_PX - 1)


def test_shot_frames_in_an_awkward_workdir(reports: tuple[TeardownReport, TeardownReport, Path], tmp_path: Path) -> None:
    yours = reports[0]
    odd = tmp_path / "a b%c d"
    frames = teardown.extract_shot_frames(yours, odd)
    assert len(frames) == len(yours.shots) == 3
    for f in frames:
        assert f is not None and f.parent == odd.resolve()
        with Image.open(f) as im:
            assert im.height == 360


def test_custom_labels_reach_the_difference_text(reports: tuple[TeardownReport, TeardownReport, Path]) -> None:
    assert card._diff_text("none in reference", "Yours", "Take B") == "none in Take B"
    assert card._diff_text("none in yours", "Take A", "Reference") == "none in Take A"
    assert "silencedetect" in card.method_text([reports[0]]) and "scdet" in card.method_text([reports[0]])
