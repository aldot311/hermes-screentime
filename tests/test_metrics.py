"""screentime_core.metrics against Screen-Time's compute.ts.

tests/parity/compute.json and real_reports.json hold inputs and the TypeScript's outputs (dev/ts-parity/dump.mts).
Every field is compared (floats to 1e-9) except the deliberate differences, which have their own tests below.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path

import pytest

from screentime_core.errors import ScreentimeError
from screentime_core.metrics import (
    MAX_GAP_FRAMES,
    FrameSample,
    compute,
    frame_samples,
    frames_to_segments,
    longest_strict_run,
)
from screentime_core.models import Box, Detection, Mention, Segment, VideoInfo, VideoMetrics

PARITY = Path(__file__).parent / "parity"
COMPUTE = json.loads((PARITY / "compute.json").read_text(encoding="utf-8"))
REAL = json.loads((PARITY / "real_reports.json").read_text(encoding="utf-8"))
CASES = COMPUTE["cases"] + [r["compute"] for r in REAL["reports"]]
TOL = 1e-9


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOL)


def info_of(js: dict, **over) -> VideoInfo:
    return VideoInfo(width=js["width"], height=js["height"], frame_count=js["frameCount"], duration_sec=js["durationSec"], **over)


def detection(row: list, object_id: int | None = None) -> Detection:
    frame, oid, area = row
    return Detection(frame=frame, object_id=oid if object_id is None else object_id, box=Box(0, 0, 10, 10),
                     area_fraction=area, centroid_x=0.5, centroid_y=0.5)


def run(case: dict, **kw) -> VideoMetrics:
    mentions = [Mention(m["startSec"], m["endSec"], m["text"]) for m in case["mentions"]]
    return compute([detection(r) for r in case["detections"]], set(case["counted"]), info_of(case["info"]),
                   mentions=mentions, **kw)


def ids(cases: list[dict]) -> list[str]:
    return [f"{i}-{c['name'][:48]}" for i, c in enumerate(cases)]


def counted_instances(case: dict) -> dict[int, list[float]]:
    """Independent reference: per on-screen frame, every counted non-empty instance's area (clamped to 0..1)."""
    counted, frame_count = set(case["counted"]), case["info"]["frameCount"]
    out: dict[int, list[float]] = {}
    for frame, oid, area in case["detections"]:
        if oid not in counted or not float(frame).is_integer() or not 0 <= frame < frame_count:
            continue
        a = min(1.0, max(0.0, area))
        if a > 0:
            out.setdefault(int(frame), []).append(a)
    return out


def several_in_a_frame(case: dict) -> set[int]:
    return {f for f, areas in counted_instances(case).items() if len(areas) > 1}


def mean_in_frame_order(values_by_frame: dict[int, float]) -> float:
    total = 0.0
    for f in sorted(values_by_frame):
        total += values_by_frame[f]
    return total / len(values_by_frame)


# ----------------------------------------------------------------------------------------------------------------------
# Parity


def test_parity_files_cover_what_they_claim():
    assert len(COMPUTE["cases"]) >= 60 and len(REAL["reports"]) == 3
    assert all(src["sameAsCommit"] for src in COMPUTE["provenance"]["sources"].values())
    names = " ".join(c["name"] for c in COMPUTE["cases"])
    for needed in ("gap of 1 frame", "gap of 2 frames", "gap of 3 frames", "empty video", "two objects in one frame"):
        assert needed in names


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_compute_matches_typescript(case):
    m = run(case)
    ts = case["ts"]["metrics"]
    assert close(m.seconds_on_screen, ts["secondsOnScreen"])
    assert close(m.share_of_video, ts["shareOfVideo"])
    assert close(m.first_seen_sec, ts["firstSeenSec"])
    assert m.visible_frames == ts["visibleFrames"]
    assert len(m.segments) == len(ts["segments"])
    for got, want in zip(m.segments, ts["segments"]):
        assert close(got.start_sec, want["startSec"]) and close(got.end_sec, want["endSec"])
    assert [(x.start_sec, x.end_sec, x.text) for x in m.mentions] == [
        (x["startSec"], x["endSec"], x["text"]) for x in ts["mentions"]
    ]
    assert m.spoken_checked is None

    samples = frame_samples([detection(r) for r in case["detections"]], case["counted"], info_of(case["info"]))
    assert [s.frame for s in samples] == [f for f, _ in case["ts"]["samples"]]
    several = several_in_a_frame(case)
    for got, (frame, area) in zip(samples, case["ts"]["samples"]):
        if frame not in several:  # one instance: the TypeScript's sum is that instance
            assert close(got.area_fraction, area)
    if not several:
        assert close(m.avg_area_fraction, ts["avgAreaFraction"])

    # The strict run, measured by the TypeScript's own framesToSegments(frames, 0).
    strict = case["ts"]["strict"]
    assert close(m.longest_run_sec, strict["longestRunSec"])
    assert close(m.longest_run_start_sec, strict["longestRunStartSec"])
    if strict["longestRunSec"] == ts["longestRunSec"]:
        assert close(m.longest_run_sec, ts["longestRunSec"])


@pytest.mark.parametrize(
    "case", COMPUTE["framesToSegments"], ids=[f"{c['frames'][:6]}-gap{c['maxGap']}" for c in COMPUTE["framesToSegments"]]
)
def test_frames_to_segments_matches_typescript(case):
    got = frames_to_segments(case["frames"], max_gap=case["maxGap"])
    assert [(s.start_sec, s.end_sec) for s in got] == [(s["startSec"], s["endSec"]) for s in case["ts"]]


# ----------------------------------------------------------------------------------------------------------------------
# Deliberate differences


def test_longest_run_is_strict_where_the_typescript_bridged_gaps():
    bridged = [c for c in CASES if not close(c["ts"]["metrics"]["longestRunSec"], c["ts"]["strict"]["longestRunSec"])]
    assert len(bridged) >= 5, "the parity set should exercise bridged runs"
    for case in bridged:
        m = run(case)
        assert m.longest_run_sec < case["ts"]["metrics"]["longestRunSec"]
        assert close(m.longest_run_sec, case["ts"]["strict"]["longestRunSec"])


def test_strict_run_example_and_segments_still_bridge():
    # Seen 0-7, 10-17, 20-27: two frames missing twice. The TypeScript's longest run was 28 frames (1.87 s).
    frames = [*range(0, 8), *range(10, 18), *range(20, 28)]
    m = compute([detection([f, 1, 0.1]) for f in frames], {1}, VideoInfo(1080, 1920, 60, 4.0))
    assert m.longest_run_sec == 8 / 15
    assert m.longest_run_start_sec == 0.0
    assert m.segments == [Segment(0.0, 28 / 15)]  # the navigation aid still bridges
    assert m.seconds_on_screen == 24 / 15


def test_longest_run_tie_goes_to_the_first_run():
    assert longest_strict_run([*range(0, 10), *range(20, 30)]) == (10, 0)
    assert longest_strict_run([3, 5, 6, 9]) == (2, 5)
    assert longest_strict_run([]) == (0, None)
    m = compute([detection([f, 1, 0.1]) for f in [*range(0, 10), *range(20, 30)]], {1}, VideoInfo(1, 1, 60, 4.0))
    assert m.longest_run_start_sec == 0.0


def test_area_is_the_largest_instance_not_the_sum():
    # Two accepted objects in frame 3 (0.1 and 0.3): the TypeScript said 0.4. Two at 0.7 in frame 0: it said 1.0.
    m = compute([detection([3, 1, 0.1]), detection([3, 2, 0.3])], {1, 2}, VideoInfo(1, 1, 10, 10 / 15))
    assert m.avg_area_fraction == 0.3 and m.peak_area_fraction == 0.3
    m = compute([detection([0, 1, 0.7]), detection([0, 2, 0.7])], {1, 2}, VideoInfo(1, 1, 10, 10 / 15))
    assert m.avg_area_fraction == 0.7


def test_area_in_parity_cases_with_several_instances_per_frame():
    overlapping = [c for c in CASES if several_in_a_frame(c)]
    assert len(overlapping) >= 4, "the parity set should exercise several accepted objects in one frame"
    differs = 0
    for case in overlapping:
        m = run(case)
        largest = {f: max(areas) for f, areas in counted_instances(case).items()}
        assert m.visible_frames == sorted(largest)
        assert close(m.avg_area_fraction, mean_in_frame_order(largest))
        assert close(m.peak_area_fraction, max(largest.values()))
        differs += not close(m.avg_area_fraction, case["ts"]["metrics"]["avgAreaFraction"])
    assert differs >= 3


def test_peak_area_and_its_time():
    rows = [[0, 1, 0.1], [1, 1, 0.4], [2, 1, 0.2], [5, 1, 0.4], [5, 2, 0.9]]
    m = compute([detection(r) for r in rows], {1}, VideoInfo(1, 1, 30, 2.0))
    assert m.peak_area_fraction == 0.4 and m.peak_area_sec == 1 / 15  # the first frame at the peak; object 2 not counted
    empty = compute([], {1}, VideoInfo(1, 1, 30, 2.0))
    assert (empty.peak_area_fraction, empty.peak_area_sec, empty.avg_area_fraction) == (0.0, None, 0.0)
    assert (empty.longest_run_sec, empty.longest_run_start_sec, empty.first_seen_sec) == (0.0, None, None)


def test_no_prominence_or_centre_weighting():
    names = {f.name for f in fields(VideoMetrics)}
    assert not names & {"prominence_score", "prominence_sized", "center_weighted_sec", "centre_weighted_sec"}


def test_times_use_the_videos_fps():
    m = compute([detection([f, 1, 0.1]) for f in range(30, 60)], {1}, VideoInfo(1, 1, 90, 3.0, fps=30))
    assert m.seconds_on_screen == 1.0 and m.first_seen_sec == 1.0 and m.share_of_video == 1 / 3
    assert m.segments == [Segment(1.0, 2.0)] and m.longest_run_sec == 1.0


@pytest.mark.parametrize("fps", [0, -15, float("nan"), float("inf"), True])
def test_an_unusable_fps_is_bad_input(fps):
    with pytest.raises(ScreentimeError) as err:
        compute([], {1}, VideoInfo(1, 1, 10, 1.0, fps=fps))
    assert err.value.code == "bad_input"


# ----------------------------------------------------------------------------------------------------------------------
# Behaviour kept from the TypeScript, spelled out


def test_a_frame_counts_once_and_only_counted_objects_count():
    rows = [[0, 1, 0.1], [0, 2, 0.1], [1, 3, 0.5], [2, 3, 0.5]]
    m = compute([detection(r) for r in rows], {1, 2}, VideoInfo(1, 1, 15, 1.0))
    assert m.visible_frames == [0] and m.seconds_on_screen == 1 / 15


def test_frames_outside_the_video_empty_masks_and_bad_numbers_are_not_on_screen():
    rows = [[-1, 1, 0.2], [30, 1, 0.2], [2.5, 1, 0.2], [3.0, 1, 0.2], [4, 1, 0.0], [5, 1, -0.1], [6, 1, float("nan")],
            [7, 1, float("inf")], [8, 1, 1.7]]
    m = compute([detection(r) for r in rows], {1}, VideoInfo(1, 1, 30, 2.0))
    assert m.visible_frames == [3, 8]
    assert m.peak_area_fraction == 1.0  # an area above 1 is clamped


def test_mentions_are_sorted_and_spoken_checked_passes_through():
    said = [Mention(5, 6, "b"), Mention(1, 2, "a"), Mention(1, 1.5, "a2")]
    m = compute([], {1}, VideoInfo(1, 1, 150, 10.0), mentions=said, spoken_checked=False)
    assert [x.text for x in m.mentions] == ["a", "a2", "b"]
    assert m.spoken_checked is False


def test_frame_samples_are_sorted_and_hold_the_largest_instance():
    rows = [[9, 1, 0.2], [2, 2, 0.3], [2, 1, 0.5], [9, 2, 0.1]]
    assert frame_samples([detection(r) for r in rows], [1, 2], VideoInfo(1, 1, 30, 2.0)) == [
        FrameSample(2, 0.5),
        FrameSample(9, 0.2),
    ]


def test_max_gap_is_two_frames():
    assert MAX_GAP_FRAMES == 2
    assert frames_to_segments([0, 3]) == [Segment(0.0, 4 / 15)]
    assert frames_to_segments([0, 4]) == [Segment(0.0, 1 / 15), Segment(4 / 15, 5 / 15)]
