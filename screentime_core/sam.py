"""SAM 3.1 video output text to detections, with Meta's official parser (meta-sam-parser).

Port of the Screen-Time app's src/lib/sam/parse.ts and geometry.ts: the same record pairing (the latest box and mask
for each frame and object), the same area and centroid arithmetic, and the same fallback to the box when a mask cannot
be decoded. Only parser APIs present in both 0.1.x and 0.2.x are used (``video_segmentation_format``,
``decode_mask_to_raster`` and the record types), so either release works.

The input is the special-token text SAM streams in its output_text lane (``SamResult.text``): one line per frame,
``<Nf>`` then comma-separated ``id<|box;...|><|mask;...|>`` records. Frames with no line have no detections.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import compress
from typing import Optional

from meta_sam_parser import (
    CompletedOutcome,
    InvalidSegmentationMaskError,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    video_segmentation_format,
)

from .errors import ScreentimeError
from .models import Box, Detection


@dataclass
class SamParse:
    detections: list[Detection]  # sorted by frame, then object id
    diagnostics: list[str] = field(default_factory=list)  # recoverable format problems, for logs and report notes


def parse_sam_text(text: str, width: int, height: int) -> list[Detection]:
    """Detections from SAM 3.1 video output text. `width` x `height` is the frame size SAM saw (the normalized video)."""
    return parse_sam_output(text, width, height).detections


def parse_sam_output(text: str, width: int, height: int) -> SamParse:
    """Like parse_sam_text, plus the parser's diagnostics and any frame-size mismatch."""
    if not (width > 0 and height > 0):
        raise ScreentimeError("bad_input", "Parsing SAM output needs a positive frame width and height.")
    if not isinstance(text, str):
        raise TypeError("SAM output must be text")
    parser = video_segmentation_format().create_parser()
    parser.push(text, emit=False)
    result = parser.finish(CompletedOutcome()).result
    boxes = [r for r in result.records if isinstance(r, SegmentationBoxRecord)]
    masks = [r for r in result.records if isinstance(r, SegmentationMaskRecord)]
    diagnostics = [f"{d.severity} {d.code} on line {d.line}: {d.message}" for d in result.diagnostics]
    ignored = sum(1 for r in result.records if r.kind == "text" and r.text.strip())
    if ignored:
        diagnostics.append(f"warning ignored_text: {ignored} line(s) that are not frame records")
    diagnostics.extend(frame_size_mismatches(result.raw_output, width, height))
    return SamParse(detections=to_detections(boxes, masks, width, height), diagnostics=diagnostics)


def to_detections(
    boxes: Iterable[SegmentationBoxRecord],
    masks: Iterable[SegmentationMaskRecord],
    width: int,
    height: int,
) -> list[Detection]:
    """One detection per (frame, object): later records for the same key replace earlier ones, as in the app."""
    masks_by_key = {_key(m): m for m in masks}
    boxes_by_key = {_key(b): b for b in boxes}
    detections = [_to_detection(b, masks_by_key.get(k), width, height) for k, b in boxes_by_key.items()]
    detections.sort(key=lambda d: (d.frame, d.object_id))
    return detections


def group_tracks(detections: Iterable[Detection]) -> dict[int, list[Detection]]:
    """Detections by SAM object id (ascending), each list sorted by frame."""
    by_id: dict[int, list[Detection]] = {}
    for d in detections:
        by_id.setdefault(d.object_id, []).append(d)
    return {oid: sorted(by_id[oid], key=lambda d: d.frame) for oid in sorted(by_id)}


def frame_to_sec(frame: int, fps: float) -> float:
    return frame / fps


# --------------------------------------------------------------------------------------------------------------------
# Geometry (geometry.ts)


def mask_footprint(
    raster: bytes,
    mask_width: int,
    mask_height: int,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    """(area_fraction, centroid_x, centroid_y) of a row-major 0/1 raster covering `bounds` (left, top, right, bottom,
    half-open) in a width x height frame. The raster is scaled to the bounds if its size differs.
    An empty mask has zero area and the box centre as its centroid."""
    left, top, right, bottom = bounds
    scale_x = (right - left) / mask_width
    scale_y = (bottom - top) / mask_height
    count = 0
    sum_col = 0
    sum_row = 0
    cols = range(mask_width)
    for row in range(mask_height):
        line = raster[row * mask_width : (row + 1) * mask_width]
        n = line.count(1)
        if not n:
            continue
        count += n
        sum_row += row * n
        sum_col += sum(compress(cols, line))
    if count == 0:
        cx, cy = _box_centroid(bounds, width, height)
        return 0.0, cx, cy
    area = count * scale_x * scale_y
    return (
        _ratio01(area, width * height),
        _ratio01(left + (sum_col / count + 0.5) * scale_x, width),
        _ratio01(top + (sum_row / count + 0.5) * scale_y, height),
    )


def box_footprint(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float]:
    """The box's own area and centre, used when its mask cannot be decoded."""
    left, top, right, bottom = bounds
    area = (right - left) * (bottom - top)
    cx, cy = _box_centroid(bounds, width, height)
    return _ratio01(area, width * height), cx, cy


def frame_size_mismatches(raw_output: str, width: int, height: int) -> list[str]:
    """Boxes carry the frame size SAM saw (w/h). Area and centroid divide by width x height, so a different size
    would skew them silently; report it instead."""
    seen: list[str] = []
    for fields in _BOX_TOKEN.findall(raw_output):
        w = _W_FIELD.search(fields)
        h = _H_FIELD.search(fields)
        if w and h and (int(w.group(1)) != width or int(h.group(1)) != height):
            size = f"{w.group(1)}x{h.group(1)}"
            if size not in seen:
                seen.append(size)
    return [f"warning frame_size_mismatch: boxes are in a {s} frame but info is {width}x{height}" for s in seen]


# --------------------------------------------------------------------------------------------------------------------

_BOX_TOKEN = re.compile(r"<\|box;([^|]*)\|>")
_W_FIELD = re.compile(r"(?:^|;)w=(\d+)")
_H_FIELD = re.compile(r"(?:^|;)h=(\d+)")


def _key(record: SegmentationBoxRecord | SegmentationMaskRecord) -> tuple[int, str]:
    return (_frame_index(record), record.object_id)


def _frame_index(record: SegmentationBoxRecord | SegmentationMaskRecord) -> int:
    return record.frame.frame_index if record.frame is not None else 0


def _to_detection(
    box: SegmentationBoxRecord, mask: Optional[SegmentationMaskRecord], width: int, height: int
) -> Detection:
    bounds = (box.left, box.top, box.right, box.bottom)
    raster = _try_decode(mask) if mask is not None else None
    if mask is not None and raster is not None:
        b = mask.bounds
        area, cx, cy = mask_footprint(raster, mask.mask.width, mask.mask.height, (b.left, b.top, b.right, b.bottom),
                                      width, height)
    else:
        area, cx, cy = box_footprint(bounds, width, height)
    confidence = box.confidence if box.confidence is not None else (mask.confidence if mask is not None else None)
    return Detection(
        frame=_frame_index(box),
        object_id=int(box.object_id),
        # The wire format has inclusive x2/y2; the parser makes them half-open, so step back by one.
        box=Box(x1=box.left, y1=box.top, x2=box.right - 1, y2=box.bottom - 1),
        area_fraction=area,
        centroid_x=cx,
        centroid_y=cy,
        confidence=confidence,
    )


def _try_decode(mask: SegmentationMaskRecord) -> Optional[bytes]:
    try:
        return decode_mask_to_raster(mask.mask)
    except InvalidSegmentationMaskError:
        return None


def _box_centroid(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float]:
    left, top, right, bottom = bounds
    return _ratio01((left + right) / 2, width), _ratio01((top + bottom) / 2, height)


def _ratio01(num: float, den: float) -> float:
    """clamp01(num / den) as the app computes it: a division by zero (Infinity or NaN in JavaScript) gives 0."""
    if not den:
        return 0.0
    value = num / den
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return min(1.0, max(0.0, value))
