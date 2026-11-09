"""Exposure metrics for one video: a port of Screen-Time's src/lib/metrics/compute.ts.

Only detections of counted objects (the ones the look-alike check accepted) are measured. A frame counts once however
many counted objects are in it; frames outside the video, non-integer frame numbers and empty masks (area <= 0) are
not on screen. Times are frame / info.fps; a segment's end is exclusive.

Kept as in the TypeScript (checked by tests/test_metrics.py against tests/parity/*.json): unique-frame exposure, share
of video, first seen, segments with gaps of up to MAX_GAP_FRAMES missing frames bridged (a navigation aid, never a
contract number), visible_frames, mentions sorted by time.

Deliberate differences, each also marked where it happens:
- longest_run_sec is strict: consecutive counted frames only. The TypeScript measured it on the bridged segments.
- Per-frame area is the largest counted instance's mask fraction, not the sum over instances.
- No prominence score and no centre weighting (not shipped).
- Times use info.fps; the TypeScript used its FPS constant (15, the rate Measure normalizes every video to).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Any, Optional

from .errors import ScreentimeError
from .models import FPS, Detection, Mention, Segment, VideoInfo, VideoMetrics

MAX_GAP_FRAMES = 2
"""Gaps of up to this many missing frames inside a run are bridged into one segment. SAM drops single frames on motion
blur or a quick hand pass; at 15 fps two frames is 0.13 s. Segments only: no contract number uses the bridging."""

# VideoMetrics has no per-frame area field yet (proposed to the lead as `visible_areas`, parallel to visible_frames).
# compute() fills it once it exists, and rules.evaluate() reads it for min_area; until then pass samples= to evaluate.
_HAS_VISIBLE_AREAS = "visible_areas" in {f.name for f in fields(VideoMetrics)}


@dataclass(frozen=True)
class FrameSample:
    """One counted frame: its index and the largest counted instance's mask fraction in it (0..1)."""

    frame: int
    area_fraction: float


def frame_samples(detections: Iterable[Detection], counted_object_ids: Iterable[int], info: VideoInfo) -> list[FrameSample]:
    """One sample per frame with at least one counted, non-empty detection, sorted by frame.

    Deliberate difference: the TypeScript (frameSamples) summed the instances' areas, capped at 1. Two instances of one
    product (a can and its reflection, overlapping masks of one object tracked twice) are not twice the screen space,
    so a frame's area here is its largest counted instance.
    """
    counted = set(counted_object_ids)
    largest: dict[int, float] = {}
    for d in detections:
        if d.object_id not in counted:
            continue
        frame = _frame_index(d.frame)
        if frame is None or frame < 0 or frame >= info.frame_count:
            continue
        area = _clamp01(d.area_fraction)
        if area <= 0:
            continue
        if area > largest.get(frame, 0.0):
            largest[frame] = area
    return [FrameSample(frame, largest[frame]) for frame in sorted(largest)]


def frames_to_segments(frames: Sequence[int], *, fps: float = FPS, max_gap: int = MAX_GAP_FRAMES) -> list[Segment]:
    """Merge frame indices (sorted ascending) into segments, bridging gaps of up to ``max_gap`` missing frames.

    ``max_gap=0`` gives the strict runs of consecutive frames.
    """
    segments: list[Segment] = []
    start = last = -1
    for f in frames:
        if start < 0:
            start = last = f
        elif f - last - 1 <= max_gap:
            last = f
        else:
            segments.append(_frame_segment(start, last, fps))
            start = last = f
    if start >= 0:
        segments.append(_frame_segment(start, last, fps))
    return segments


def compute(
    detections: Iterable[Detection],
    counted_object_ids: Iterable[int],
    info: VideoInfo,
    *,
    mentions: Optional[Iterable[Mention]] = None,
    spoken_checked: Optional[bool] = None,
) -> VideoMetrics:
    """Metrics for one video from every tracked detection and the ids of the objects to count.

    ``mentions`` are the brand mentions heard (find_mentions); ``spoken_checked`` is passed through (False when the
    audio could not be transcribed, so an empty ``mentions`` is unknown, not "never said").
    """
    fps = video_fps(info)
    samples = frame_samples(detections, counted_object_ids, info)
    frames = [s.frame for s in samples]
    areas = [s.area_fraction for s in samples]

    seconds = len(samples) / fps
    share = min(1.0, seconds / info.duration_sec) if info.duration_sec > 0 else 0.0
    # Added sequentially, as the TypeScript's reduce does: Python's sum() of floats is compensated since 3.12 and can
    # differ in the last bit.
    avg_area = _sequential_sum(areas) / len(areas) if areas else 0.0

    # Deliberate difference: strict. The TypeScript's longest run was the longest bridged segment, so up to two frames
    # that were never seen could sit inside "one continuous run"; a contract term must not be met by unseen frames.
    run_frames, run_start = longest_strict_run(frames)

    peak_area, peak_frame = 0.0, None
    for s in samples:
        if s.area_fraction > peak_area:
            peak_area, peak_frame = s.area_fraction, s.frame

    extra: dict[str, Any] = {"visible_areas": areas} if _HAS_VISIBLE_AREAS else {}
    return VideoMetrics(
        seconds_on_screen=seconds,
        share_of_video=share,
        first_seen_sec=frames[0] / fps if frames else None,
        longest_run_sec=run_frames / fps,
        longest_run_start_sec=run_start / fps if run_start is not None else None,
        avg_area_fraction=avg_area,
        peak_area_fraction=peak_area,
        peak_area_sec=peak_frame / fps if peak_frame is not None else None,
        segments=frames_to_segments(frames, fps=fps),
        visible_frames=frames,
        mentions=sorted(mentions or [], key=lambda m: m.start_sec),
        spoken_checked=spoken_checked,
        **extra,
    )


def longest_strict_run(frames: Sequence[int]) -> tuple[int, Optional[int]]:
    """(length in frames, first frame) of the longest run of consecutive frames; the earliest wins a tie.

    ``frames`` must be sorted ascending without repeats. (0, None) when there are none.
    """
    best_len, best_start = 0, None
    run_len, run_start, prev = 0, None, None
    for f in frames:
        if prev is not None and f == prev + 1:
            run_len += 1
        else:
            run_len, run_start = 1, f
        if run_len > best_len:
            best_len, best_start = run_len, run_start
        prev = f
    return best_len, best_start


def video_fps(info: VideoInfo) -> float:
    """The analysis frame rate, or a bad_input error when it cannot turn frames into seconds."""
    fps = info.fps
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ScreentimeError(
            "bad_input",
            f"The video's frame rate ({fps!r}) cannot turn frames into seconds.",
            fix="Normalize the video to a fixed frame rate before measuring it.",
        )
    return fps


def _frame_segment(first: int, last: int, fps: float) -> Segment:
    return Segment(start_sec=first / fps, end_sec=(last + 1) / fps)


def _frame_index(value: Any) -> Optional[int]:
    """The frame as an int when it is a whole number (2 and 2.0 both are, as in JavaScript), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _clamp01(n: Any) -> float:
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n):
        return 0.0
    return min(1.0, max(0.0, float(n)))


def _sequential_sum(xs: Iterable[float]) -> float:
    total = 0.0
    for x in xs:
        total += x
    return total
