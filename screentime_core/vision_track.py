"""Tracking with any vision model: the keyless alternative to SAM 3.1.

The normalized video is sampled at a few frames a second; each batch of frames goes to the user's own vision model
(through Hermes) with one question: box every visible <noun>. Boxes in consecutive samples are linked into objects by
overlap, and each sample stands for the frames around it, so the rest of Measure (the product check, metrics, terms,
proof reel) runs unchanged on the result.

What this gives up against SAM 3.1, and the report says so: seconds are accurate to one sample (1/3 s at the default
3 samples a second), sizes come from boxes rather than masks (so they read larger), and a model can miss or invent a
box. Accuracy on the labelled Reels: docs/ACCURACY.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .errors import ScreentimeError
from .models import FPS, Box, Detection, ReceiptCall, VideoInfo

# llm(instructions, images: list[(bytes, mime)], json_schema, schema_name) -> {"parsed", "provider", "model", "usd"}
StructuredCall = Callable[[str, list[tuple[bytes, str]], dict, str], dict]

DEFAULT_SAMPLES_PER_SEC = 3
BATCH = 4  # frames per model call
MAX_SIDE = 640  # longest side of a frame sent to the model
MAX_CALLS = 120  # per run; a 120 s video at 3 samples a second is 90 calls
WORKERS = 8
LINK_GAP = 2  # an object may be missed in this many samples in a row and keep its identity
MIN_IOU = 0.1

DETECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer"},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "integer"},
                                "y1": {"type": "integer"},
                                "x2": {"type": "integer"},
                                "y2": {"type": "integer"},
                            },
                            "required": ["x1", "y1", "x2", "y2"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["frame", "objects"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["frames"],
    "additionalProperties": False,
}


def detect_prompt(noun: str, n: int) -> str:
    return (
        f"These {n} images are frames of one video, in order: frame 1 to frame {n}. In each frame, find every visible "
        f'"{noun}", including small, partly hidden or blurred ones, and give a tight box around the whole object. '
        "Coordinates are integers from 0 to 1000: x across the image width from the left edge, y down the image height "
        "from the top edge; (x1, y1) is the top-left corner and (x2, y2) the bottom-right. Return one entry per frame "
        f'with "frame" set to its number (1 to {n}) and "objects" listing the boxes, or an empty list when there is '
        f'no "{noun}" in that frame. Do not box anything that is not a "{noun}".'
    )


@dataclass
class _Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _iou(a: _Box, b: _Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def parse_boxes(parsed: Any, n: int) -> list[list[_Box]]:
    """Per frame (0..n-1) the boxes as fractions 0..1 of width and height. Malformed entries are dropped."""
    out: list[list[_Box]] = [[] for _ in range(n)]
    frames = parsed.get("frames") if isinstance(parsed, dict) else None
    for entry in frames or []:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("frame")
        if isinstance(idx, bool) or not isinstance(idx, int) or not 1 <= idx <= n:
            continue
        for o in entry.get("objects") or []:
            if not isinstance(o, dict):
                continue
            try:
                vals = [min(1000.0, max(0.0, float(o[k]))) / 1000.0 for k in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                continue
            x1, x2 = sorted((vals[0], vals[2]))
            y1, y2 = sorted((vals[1], vals[3]))
            b = _Box(x1, y1, x2, y2)
            if b.area() < 1e-5 or b.area() > 0.98:
                continue  # a point, or the whole frame
            out[idx - 1].append(b)
    return out


def link(samples: list[list[_Box]]) -> list[list[tuple[int, _Box]]]:
    """Objects as [(sample index, box)], linked across samples by overlap (greedy, best overlap first)."""
    tracks: list[list[tuple[int, _Box]]] = []
    for s, boxes in enumerate(samples):
        live = [t for t in range(len(tracks)) if s - tracks[t][-1][0] <= LINK_GAP + 1]
        pairs = sorted(
            ((_iou(tracks[t][-1][1], b), t, j) for t in live for j, b in enumerate(boxes)), reverse=True
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, t, j in pairs:
            if score < MIN_IOU:
                break
            if t in used_t or j in used_b or tracks[t][-1][0] == s:
                continue
            tracks[t].append((s, boxes[j]))
            used_t.add(t)
            used_b.add(j)
        for j, b in enumerate(boxes):
            if j not in used_b:
                tracks.append([(s, b)])
    return tracks


def to_detections(
    tracks: list[list[tuple[int, _Box]]], sample_frames: list[int], every: int, offset: int, info: VideoInfo
) -> list[Detection]:
    """Each sample stands for the `every` frames around it; boxes become pixel boxes in the normalized video."""
    out: list[Detection] = []
    w, h = info.width, info.height
    for oid, track in enumerate(tracks):
        for s, b in track:
            start = sample_frames[s] - offset
            for f in range(max(0, start), min(info.frame_count, start + every)):
                out.append(
                    Detection(
                        frame=f,
                        object_id=oid,
                        box=Box(b.x1 * w, b.y1 * h, max(b.x1 * w, b.x2 * w - 1), max(b.y1 * h, b.y2 * h - 1)),
                        area_fraction=b.area(),
                        centroid_x=(b.x1 + b.x2) / 2,
                        centroid_y=(b.y1 + b.y2) / 2,
                        confidence=None,
                    )
                )
    out.sort(key=lambda d: (d.frame, d.object_id))
    return out


@dataclass
class VisionTrackResult:
    detections: list[Detection]
    call: ReceiptCall
    notes: list[str]
    label: str


class VisionTracker:
    """Box a noun in sampled frames with the user's own vision model, then link the boxes into objects."""

    def __init__(self, llm: StructuredCall, samples_per_sec: int = DEFAULT_SAMPLES_PER_SEC):
        self.llm = llm
        self.samples_per_sec = max(1, min(int(samples_per_sec or DEFAULT_SAMPLES_PER_SEC), FPS))
        self.label = "vision"

    def track(self, video: str, info: VideoInfo, noun: str, workdir: Path,
              progress: Optional[Callable[[int, int], None]] = None) -> VisionTrackResult:
        from . import media

        every = max(1, round(FPS / self.samples_per_sec))
        offset = every // 2
        frames = media.sample_frames(video, workdir / "samples", every=every, offset=offset, max_side=MAX_SIDE)
        if not frames:
            raise ScreentimeError("bad_input", "No frames could be read from the video.")
        batches = [frames[i:i + BATCH] for i in range(0, len(frames), BATCH)]
        if len(batches) > MAX_CALLS:
            raise ScreentimeError("too_long", f"This video needs {len(batches)} model calls to track; the limit is "
                                  f"{MAX_CALLS}.", fix="Trim the video, or use a Meta key (SAM 3.1 tracks every frame).")
        t0 = time.monotonic()
        usd = 0.0
        usd_known = True
        labels: set[str] = set()
        failed: list[int] = []
        per_sample: list[list[_Box]] = [[] for _ in frames]
        done = 0

        def run(bi: int) -> tuple[int, Optional[dict]]:
            images = [(p.read_bytes(), "image/jpeg") for _, p in batches[bi]]
            prompt = detect_prompt(noun, len(images))
            for _attempt in range(2):
                try:
                    return bi, self.llm(prompt, images, DETECT_SCHEMA, "frame_boxes")
                except ScreentimeError as exc:
                    if exc.code in ("auth", "billing", "model_not_found"):
                        raise
                except Exception:  # noqa: S112 - one retry, then the batch counts as unread
                    continue
            return bi, None

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for bi, res in pool.map(run, range(len(batches))):
                done += 1
                if progress:
                    progress(done, len(batches))
                if res is None or not isinstance(res.get("parsed"), dict):
                    failed.append(bi)
                    continue
                labels.add(f"{res.get('provider') or '?'}/{res.get('model') or '?'}")
                if res.get("usd") is None:
                    usd_known = False
                else:
                    usd += float(res["usd"])
                for k, boxes in enumerate(parse_boxes(res["parsed"], len(batches[bi]))):
                    per_sample[bi * BATCH + k] = boxes
        if len(failed) == len(batches):
            raise ScreentimeError(
                "engine_unavailable",
                "Your Hermes model could not look at the frames (no answer it could read).",
                fix="Set a vision-capable model for the screentime_track task (auxiliary.screentime_track), "
                "or add a Meta Model API key so SAM 3.1 tracks the video.",
            )
        tracks = link(per_sample)
        detections = to_detections(tracks, [f for f, _ in frames], every, offset, info)
        self.label = "hermes:" + ("+".join(sorted(labels)) if labels else "?")
        notes = [
            (f"Tracked by your own model ({self.label.removeprefix('hermes:')}) at {self.samples_per_sec} frames a "
            f"second, not by SAM 3.1: seconds are accurate to about {1 / self.samples_per_sec:.1f} s and sizes come "
            "from boxes, so they read larger than SAM's masks.")
        ]
        if failed:
            unread = sum(len(batches[b]) for b in failed)
            notes.append(f"{unread} of {len(frames)} sampled frames got no readable answer from the model and count "
                         "as not seen.")
        call = ReceiptCall(
            model=self.label,
            purpose=f'Find "{noun}" in sampled frames',
            input=f"{len(frames)} frames ({self.samples_per_sec} a second) in {len(batches)} calls",
            output=f"{len(tracks)} object{'s' if len(tracks) != 1 else ''}",
            ms=int((time.monotonic() - t0) * 1000),
            usd=usd if usd_known else 0.0,
        )
        if not usd_known:
            notes.append("Your model's provider did not report a cost for the tracking calls.")
        return VisionTrackResult(detections, call, notes, self.label)
