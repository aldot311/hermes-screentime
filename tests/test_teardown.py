"""Teardown and Compare on clips generated with ffmpeg lavfi at test time: no network, no key, no files on disk."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from screentime_core import teardown
from screentime_core.errors import ScreentimeError
from screentime_core.models import Segment, Shot, TeardownReport, VideoInfo, from_dict, to_dict

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe on PATH")

FRAME = 1 / 15  # one analysis frame at the default 15 fps
SIZE = "320x240"
TRUE_CUTS = [1.0, 2.6, 4.0]
CONCAT4 = "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,format=yuv420p[v]"


def _ffmpeg(out: Path, *args: str) -> Path:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args, str(out)], check=True, timeout=120)
    return out


def _four_shots(rate: int = 30) -> list[str]:
    """testsrc2 1.0 s, red 1.6 s, mandelbrot 1.4 s, smptebars 1.5 s: true cuts at 1.0, 2.6 and 4.0 s (5.5 s long)."""
    args: list[str] = []
    for dur, src in ((1.0, "testsrc2="), (1.6, "color=c=red:"), (1.4, "mandelbrot="), (1.5, "smptebars=")):
        args += ["-f", "lavfi", "-t", f"{dur}", "-i", f"{src}size={SIZE}:rate={rate}"]
    return args


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    mpeg4 = ["-c:v", "mpeg4", "-q:v", "3"]
    aac = ["-c:a", "aac", "-b:a", "96k"]
    return {
        # 4 visually distinct shots at 30 fps, no audio stream
        "cuts": _ffmpeg(d / "cuts.mp4", *_four_shots(), "-filter_complex", CONCAT4, "-map", "[v]", *mpeg4),
        # the same 4 shots with audio: 0.8 s of digital silence, then a 440 Hz tone
        "full": _ffmpeg(
            d / "full.mp4", *_four_shots(),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=0.8", "-f", "lavfi", "-i", "sine=f=440:r=48000:d=4.7",
            "-filter_complex", CONCAT4 + ";[4:a][5:a]concat=n=2:v=0:a=1[a]", "-map", "[v]", "-map", "[a]",
            *mpeg4, *aac,
        ),
        # one shot, 0.8 s of silence then a tone (3 s)
        "tone": _ffmpeg(
            d / "tone.mp4", "-f", "lavfi", "-i", f"color=c=gray:size={SIZE}:rate=15:duration=3",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=0.8", "-f", "lavfi", "-i", "sine=f=440:r=48000:d=2.2",
            "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[a]", "-map", "0:v", "-map", "[a]", *mpeg4, *aac,
        ),
        # testsrc2 for 30 frames, a 2-frame white flash, mandelbrot for 30 frames (15 fps)
        "flash": _ffmpeg(
            d / "flash.mp4",
            "-f", "lavfi", "-i", f"testsrc2=size={SIZE}:rate=15,trim=end_frame=30",
            "-f", "lavfi", "-i", f"color=c=white:size={SIZE}:rate=15,trim=end_frame=2",
            "-f", "lavfi", "-i", f"mandelbrot=size={SIZE}:rate=15,trim=end_frame=30",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]", "-map", "[v]",
            "-c:v", "mpeg4", "-q:v", "2",
        ),
        # 100 Hz, 1 kHz (quieter) and 4 kHz tones for 2 s each: exercises both K-weighting filters
        "tones": _ffmpeg(
            d / "tones.mp4", "-f", "lavfi", "-i", f"color=c=black:size={SIZE}:rate=15:duration=6",
            "-f", "lavfi", "-i", "sine=f=100:r=48000:d=2", "-f", "lavfi", "-i", "sine=f=1000:r=48000:d=2",
            "-f", "lavfi", "-i", "sine=f=4000:r=48000:d=2",
            "-filter_complex", "[2:a]volume=0.3[q];[1:a][q][3:a]concat=n=3:v=0:a=1[a]", "-map", "0:v",
            "-map", "[a]", *mpeg4, "-c:a", "aac", "-b:a", "128k",
        ),
    }


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("work")


@pytest.fixture(scope="module")
def reports(clips: dict[str, Path], work: Path) -> dict[str, TeardownReport]:
    return {
        "cuts": teardown.analyze(clips["cuts"], work),
        "full": teardown.analyze(clips["full"], work, product_first_seen_sec=1.2, brand_first_said_sec=2.0),
        "tone": teardown.analyze(clips["tone"], work),
    }


# --------------------------------------------------------------------------------------------------------------------
# Cuts and shots


@needs_ffmpeg
def test_cuts_within_one_analysis_frame_of_the_truth(reports: dict[str, TeardownReport]) -> None:
    r = reports["cuts"]
    assert len(r.cuts_sec) == len(TRUE_CUTS), r.cuts_sec
    for got, want in zip(r.cuts_sec, TRUE_CUTS, strict=True):
        assert abs(got - want) <= FRAME + 1e-6, (got, want)
    assert r.first_cut_sec == r.cuts_sec[0]
    assert r.cuts_first_3s == 2
    assert r.cuts_per_10s == [3]


@needs_ffmpeg
def test_shots_tile_the_video(reports: dict[str, TeardownReport]) -> None:
    r = reports["cuts"]
    assert abs(r.info.duration_sec - 5.5) < 0.05
    assert r.info.fps == 15 and r.info.width == 320 and r.info.height == 240
    assert abs(r.info.frame_count - 5.5 * 15) <= 1
    assert [s.index for s in r.shots] == [0, 1, 2, 3]
    assert r.shots[0].start_sec == 0.0 and r.shots[-1].end_sec == r.info.duration_sec
    for a, b in zip(r.shots, r.shots[1:], strict=False):
        assert a.end_sec == b.start_sec
    assert abs(sum(s.duration_sec for s in r.shots) - r.info.duration_sec) < 0.01
    assert abs(r.median_shot_sec - 1.45) <= FRAME + 1e-6
    assert abs(r.longest_shot_sec - 1.6) <= FRAME + 1e-6 and abs(r.longest_shot_start_sec - 1.0) <= FRAME + 1e-6


@needs_ffmpeg
def test_pixel_change_is_zero_on_a_still_shot(reports: dict[str, TeardownReport]) -> None:
    red, bars = reports["cuts"].shots[1], reports["cuts"].shots[3]
    moving = reports["cuts"].shots[0]  # testsrc2 animates
    assert red.pixel_change is not None and red.pixel_change < 0.2
    assert bars.pixel_change is not None and bars.pixel_change < 0.2
    assert moving.pixel_change is not None and moving.pixel_change > red.pixel_change + 0.3


@needs_ffmpeg
def test_flash_frame_shorter_than_min_shot_is_merged(clips: dict[str, Path], tmp_path: Path) -> None:
    # Negative control: with merging off, scdet sees both edges of the 2-frame flash.
    raw = teardown.analyze(clips["flash"], tmp_path, min_shot_sec=0.0)
    assert len(raw.cuts_sec) == 2, raw.cuts_sec
    assert abs(raw.cuts_sec[0] - 2.0) <= FRAME + 1e-6 and abs(raw.cuts_sec[1] - 32 / 15) <= FRAME + 1e-6
    merged = teardown.analyze(clips["flash"], tmp_path)
    assert len(merged.cuts_sec) == 1 and abs(merged.cuts_sec[0] - 2.0) <= FRAME + 1e-6
    assert len(merged.shots) == 2
    assert any("merged" in n for n in merged.notes)
    assert merged.thresholds["min_shot_sec"] == 0.2


def test_merge_cuts_drops_slivers_at_start_flash_and_end() -> None:
    kept, merged = teardown.merge_cuts([0.067, 1.0, 1.1, 2.0, 5.45], duration=5.5, min_shot_sec=0.2)
    assert kept == [1.0, 2.0] and merged == 3
    assert teardown.merge_cuts([1.0, 1.1], duration=5.5, min_shot_sec=0.0) == ([1.0, 1.1], 0)


@needs_ffmpeg
def test_scores_do_not_depend_on_the_analysis_copy(clips: dict[str, Path], tmp_path: Path) -> None:
    """The MJPEG copy wants full-range pixels; scdet must still see the same frames with or without it."""
    ffmpeg = teardown._which("ffmpeg")
    caps = teardown._caps(ffmpeg)
    out = {}
    for copy in (True, False):
        tag = f"copy{int(copy)}"
        argv = teardown._pass_argv(
            ffmpeg, caps, clips["full"].resolve(), tag, 0, 1, tmp_path / f"{tag}.mkv" if copy else None,
            fps=15, threshold=10, silence_db=-40, silence_d=0.2,
        )
        assert teardown._run(argv, 120, what="test", cwd=tmp_path).returncode == 0
        out[copy] = (tmp_path / f"{tag}.scd.txt").read_text()
    assert out[True] == out[False]
    assert (tmp_path / "copy1.mkv").stat().st_size > 0


# --------------------------------------------------------------------------------------------------------------------
# Audio


@needs_ffmpeg
def test_first_sound_after_silence(reports: dict[str, TeardownReport]) -> None:
    r = reports["tone"]
    assert r.first_sound_sec is not None and abs(r.first_sound_sec - 0.8) <= 0.05
    assert len(r.silences) == 1 and r.silences[0].start_sec == 0.0 and abs(r.silences[0].end_sec - 0.8) <= 0.05
    assert r.integrated_lufs is not None and -30 < r.integrated_lufs < -15
    assert r.loudness_range_lu is None  # 3 s of audio: too short for a loudness range (ebur128 says 20 LU)
    assert r.cuts_sec == [] and r.first_cut_sec is None and len(r.shots) == 1


@needs_ffmpeg
def test_no_audio_stream_gives_none_not_zero(reports: dict[str, TeardownReport]) -> None:
    r = reports["cuts"]
    assert r.info.has_audio is False
    assert r.integrated_lufs is None and r.loudness_range_lu is None and r.first_sound_sec is None
    assert r.loudness_curve == [] and r.silences == []
    assert any("No audio" in n for n in r.notes)


@needs_ffmpeg
def test_loudness_curve_covers_the_first_3_s(reports: dict[str, TeardownReport]) -> None:
    curve = reports["full"].loudness_curve
    times = [t for t, _ in curve]
    assert times[0] == 0.5 and all(abs(b - a - 0.5) < 1e-9 for a, b in zip(times, times[1:], strict=False))
    assert times[-1] >= 5.0
    values = dict(curve)
    assert values[0.5] == teardown.LOUDNESS_FLOOR_LUFS  # digital silence, floored
    assert all(-70.0 <= v <= 0.0 for v in values.values())
    assert values[2.0] > -40 and values[2.5] > -40  # inside ebur128's unfilled 3 s window, still measured
    assert reports["full"].first_sound_sec is not None and abs(reports["full"].first_sound_sec - 0.8) <= 0.05


@needs_ffmpeg
def test_short_term_from_k_weighted_blocks_matches_ebur128(clips: dict[str, Path], tmp_path: Path) -> None:
    r = teardown.analyze(clips["tones"], tmp_path)
    blocks = teardown.parse_metadata_print((tmp_path / f"{r.run_id}.audio.txt").read_text())
    pairs = [(own, ff) for _, own, ff in teardown.block_loudness(blocks) if ff is not None and ff > -70]
    assert len(pairs) >= 25
    worst = max(abs(own - ff) for own, ff in pairs if own is not None)
    assert worst < 0.1, worst


# --------------------------------------------------------------------------------------------------------------------
# Inputs and safety


@needs_ffmpeg
def test_hls_playlist_named_mp4_is_refused(tmp_path: Path) -> None:
    playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n"
    for name in ("evil.mp4", "evil.m3u8"):
        evil = tmp_path / name
        evil.write_text(playlist)
        with pytest.raises(ScreentimeError) as err:
            teardown.analyze(evil, tmp_path / "work")
        assert err.value.code == "bad_input"
        assert "localhost" not in err.value.message  # nothing read from /etc/hosts reaches the user
    assert not list((tmp_path / "work").glob("*.analysis.mkv"))  # refused before any decoding


def test_container_check_allows_only_video_containers() -> None:
    for ok in ("mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "avi", "mpegts"):
        teardown.check_container(ok)
    for bad in ("hls", "concat", "image2", "lavfi", "wav", "mov,hls", ""):
        with pytest.raises(ScreentimeError) as err:
            teardown.check_container(bad)
        assert err.value.code == "bad_input"


@needs_ffmpeg
def test_every_read_of_the_video_is_whitelisted(
    clips: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []
    real = teardown._run

    def spy(argv: list[str], *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seen.append(list(argv))
        return real(argv, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(teardown, "_run", spy)
    r = teardown.analyze(clips["full"], tmp_path)
    (tmp_path / f"{r.run_id}.analysis.mkv").unlink()  # force a read of the original too
    assert all(p is not None for p in teardown.extract_shot_frames(r, tmp_path))
    reads = [a for a in seen if "-i" in a]
    assert len(reads) >= 3  # ffprobe, the analysis pass, the frame extraction
    for argv in reads:
        i = argv.index("-i")
        assert argv[argv.index("-protocol_whitelist") + 1] == "file,pipe" and argv.index("-protocol_whitelist") < i
        assert argv.index("-format_whitelist") < i
        assert argv[i + 1].startswith("file:/")


@needs_ffmpeg
def test_missing_file_directory_bad_params_and_too_long(clips: dict[str, Path], tmp_path: Path) -> None:
    for target in (tmp_path / "nope.mp4", tmp_path):
        with pytest.raises(ScreentimeError) as err:
            teardown.analyze(target, tmp_path / "work")
        assert err.value.code == "bad_input"
    for bad in ({"scdet_threshold": 150}, {"analysis_fps": "fast"}, {"silence_db": 6}, {"min_shot_sec": True}):
        with pytest.raises(ScreentimeError) as err:
            teardown.analyze(clips["cuts"], tmp_path, **bad)  # type: ignore[arg-type]
        assert err.value.code == "bad_input" and next(iter(bad)) in err.value.message
    with pytest.raises(ScreentimeError) as err:
        teardown.analyze(clips["cuts"], tmp_path, max_duration_sec=2.0)
    assert err.value.code == "too_long"


def test_parse_metadata_print_tolerates_odd_lines() -> None:
    text = (
        "frame:0    pts:0       pts_time:0\r\nlavfi.scd.mafd=0.000\r\nlavfi.scd.score=0.000\r\n"
        "garbage line\nframe:1    pts:NOPTS   pts_time:N/A\nlavfi.scd.mafd=12.5\nlavfi.scd.time=0.0666\n\n"
    )
    frames = teardown.parse_metadata_print(text)
    assert [f["n"] for f in frames] == [0, 1]
    assert frames[0]["t"] == 0.0 and frames[1]["t"] is None
    assert frames[1]["lavfi.scd.mafd"] == "12.5" and "lavfi.scd.time" in frames[1]


@needs_ffmpeg
def test_report_round_trips_through_json(reports: dict[str, TeardownReport]) -> None:
    r = reports["full"]
    assert from_dict(TeardownReport, to_dict(r)) == r
    assert {"scdet_threshold", "min_shot_sec", "silence_db", "silence_min_sec", "first_sound_min_sec",
            "analysis_fps", "ffmpeg_version"} <= set(r.thresholds)


# --------------------------------------------------------------------------------------------------------------------
# Compare

ROWS = ["Duration", "First cut", "Cuts in first 3 s", "Shots", "Median shot", "Longest shot", "Cuts per 10 s",
        "Pixel change (mean)", "Integrated loudness", "First sound"]
JUDGEMENT_WORDS = ("better", "worse", "score", "good", "bad", "best", "worst", "should", "%")


@needs_ffmpeg
def test_compare_with_itself_is_same_everywhere(reports: dict[str, TeardownReport]) -> None:
    c = teardown.compare(reports["full"], reports["full"])
    assert [row.metric for row in c.rows] == ROWS + ["Product first seen", "Brand first said"]
    assert all(row.diff == "same" for row in c.rows), [(row.metric, row.diff) for row in c.rows]
    assert c.mode == "compare" and c.run_id.startswith("cmp-")


@needs_ffmpeg
def test_compare_differences_are_neutral_and_specific(reports: dict[str, TeardownReport]) -> None:
    c = teardown.compare(reports["cuts"], reports["tone"])
    rows = {row.metric: row for row in c.rows}
    assert [row.metric for row in c.rows] == ROWS  # no Measure data on either side: no product or brand rows
    assert rows["Shots"].yours == 4 and rows["Shots"].theirs == 1 and rows["Shots"].diff == "3 more"
    assert rows["Duration"].diff == "2.5 s longer"
    assert rows["First cut"].theirs is None and rows["First cut"].diff == "none in reference"
    assert rows["Integrated loudness"].yours is None and rows["Integrated loudness"].diff == "none in yours"
    assert rows["Cuts per 10 s"].yours == [3] and rows["Cuts per 10 s"].diff == "+3"
    for row in c.rows:
        assert not any(w in row.diff.lower() for w in JUDGEMENT_WORDS), row


def _report(**kw: object) -> TeardownReport:
    base: dict[str, object] = dict(
        run_id="td-test", mode="teardown", video_path="/nowhere.mp4", video_sha256="0" * 64,
        info=VideoInfo(width=1080, height=1920, frame_count=300, duration_sec=20.0),
        cuts_sec=[1.0], shots=[Shot(0, 0.0, 1.0, 1.0, 2.0), Shot(1, 1.0, 20.0, 19.0, 4.0)], first_cut_sec=1.0,
        cuts_first_3s=1, cuts_per_10s=[1, 0], median_shot_sec=10.0, longest_shot_sec=19.0,
        longest_shot_start_sec=1.0, integrated_lufs=-14.0, loudness_range_lu=4.0, loudness_curve=[],
        first_sound_sec=0.0, silences=[Segment(5.0, 6.0)], thresholds={},
    )
    base.update(kw)
    return TeardownReport(**base)  # type: ignore[arg-type]


def test_compare_wording_rounds_like_the_display() -> None:
    y = _report(first_cut_sec=2.04, integrated_lufs=-16.04, cuts_per_10s=[2, 1, 0], first_sound_sec=1.5,
                longest_shot_sec=12.0, longest_shot_start_sec=4.0)
    t = _report(first_cut_sec=1.96, integrated_lufs=-14.0, cuts_per_10s=[1, 2], first_sound_sec=0.0)
    rows = {row.metric: row for row in teardown.compare(y, t).rows}
    assert rows["First cut"].yours == 2.0 and rows["First cut"].theirs == 2.0 and rows["First cut"].diff == "same"
    assert rows["Integrated loudness"].diff == "2.0 LU quieter"
    assert rows["First sound"].diff == "1.5 s later"
    assert rows["Cuts per 10 s"].diff == "+1, -1 (first 20 s)"
    assert rows["Longest shot"].yours == "12.0 s at 4.0 s"
    assert rows["Longest shot"].diff == "7.0 s shorter, starts 3.0 s later"
    assert rows["Pixel change (mean)"].yours == round((2.0 * 1 + 4.0 * 19) / 20, 1)
    assert teardown.format_value("Integrated loudness", -16.04) == "-16.0 LUFS"
    assert teardown.format_value("First cut", None) == "none"
