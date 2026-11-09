"""Shareable PNG cards for Teardown and Compare, drawn with Pillow alone (1080 x 1350, dark, made to be screenshotted).

A card shows each video as a filmstrip on one shared time axis (one thumbnail per shot, the middle frame, repeated
across the shot's width so shot length reads at a glance), cut ticks, a short-term loudness line with silences shaded,
markers for product first seen and brand first said when a Measure run supplied them, the measured table, and a footer
with the method and run ids. Text is never smaller than 22 px on the 1080 px canvas, so it stays readable on a phone.
The difference column is neutral: plain text, no red or green.
"""

from __future__ import annotations

import contextlib
import math
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from . import teardown
from .models import CompareReport, TeardownReport

W, H = 1080, 1350
PAD = 56
INNER = W - 2 * PAD
MIN_TEXT_PX = 22

Color = tuple[int, int, int]
BG: Color = (11, 13, 18)
PANEL: Color = (19, 23, 31)
RULE: Color = (40, 46, 60)
TEXT: Color = (238, 241, 246)
MUTED: Color = (150, 158, 175)
SILENCE: Color = (52, 61, 80)
PLACEHOLDER: Color = (30, 35, 46)
CUT: Color = (255, 255, 255)
YOURS: Color = (108, 196, 255)
THEIRS: Color = (255, 180, 84)
PRODUCT: Color = (74, 222, 128)
BRAND: Color = (200, 150, 255)

# Vertical rhythm, px. One set of numbers drives both the layout plan and the drawing.
OVERLINE_Y = 60  # overline baseline
TITLE_STEP = 62  # overline baseline -> title baseline
AFTER_TITLE = 40  # title baseline -> top of the strips block
LABEL_H = 48  # side label row: baseline at +26, filmstrip from +48
STRIP_GAP = 8  # filmstrip -> loudness lane
SIDE_GAP = 22  # between two sides
AXIS_GAP = 12  # last lane (or marker row) -> axis line
AXIS_H = 40  # axis line -> bottom of its labels (baseline at +32)
LEGEND_LINE = 32  # one legend line (baseline at +26)
LEGEND_GLYPH = 31  # glyph box before each legend label
LEGEND_SPACE = 22  # between legend items
AFTER_STRIPS = 16  # legend -> table
TABLE_HEAD = 38  # header row (baseline at +26, rule at +38)
AFTER_TABLE = 24
FOOTER_HEAD = 38  # rule -> "Measured, not predicted." baseline
FOOTER_LINE = 28
BOTTOM = 26
CUT_GAP = 4  # px left empty at every cut, with the white tick inside it


@dataclass(frozen=True)
class _Style:
    value_px: int
    metric_px: int
    row_pad: int
    lane_h: int
    marker_h: int


STYLES = (
    _Style(value_px=25, metric_px=24, row_pad=7, lane_h=56, marker_h=38),
    _Style(value_px=24, metric_px=23, row_pad=5, lane_h=46, marker_h=34),
    _Style(value_px=22, metric_px=22, row_pad=4, lane_h=40, marker_h=12),
)
MARKER_LABEL_MIN_H = 30  # marker rows shorter than this get the line and a tick, not the label


@contextlib.contextmanager
def _workdir(workdir: Optional[teardown.PathLike]) -> Iterator[Path]:
    if workdir is not None:
        path = Path(workdir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix="screentime-card-") as tmp:
            yield Path(tmp)


_FONTS: dict[int, Any] = {}


def _font(size: int) -> Any:
    if size < MIN_TEXT_PX:
        raise ValueError(f"text under {MIN_TEXT_PX} px is not readable on a phone ({size} px asked)")
    if size not in _FONTS:
        _FONTS[size] = ImageFont.load_default(size=size)  # Aileron Regular, scalable (Pillow 10.1+ with FreeType)
    return _FONTS[size]


def _width(draw: ImageDraw.ImageDraw, text: str, size: int) -> float:
    return draw.textlength(text, font=_font(size))


def _text(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: int, fill: Color, *,
    bold: bool = False, anchor: str = "ls",
) -> float:
    """Draws one line; returns its width. Bold is a 1 px stroke in the text colour (the bundled font has one weight)."""
    draw.text(xy, text, font=_font(size), fill=fill, anchor=anchor, stroke_width=1 if bold else 0, stroke_fill=fill)
    return _width(draw, text, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, size: int, max_w: float) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        trial = f"{line} {word}".strip()
        if not line or _width(draw, trial, size) <= max_w:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def _fit_size(draw: ImageDraw.ImageDraw, text: str, size: int, max_w: float, floor: int = MIN_TEXT_PX) -> int:
    while size > floor and _width(draw, text, size) > max_w:
        size -= 1
    return size


# --------------------------------------------------------------------------------------------------------------------
# Strips: filmstrip, loudness lane, markers, shared time axis, legend


@dataclass
class _Axis:
    x0: float
    width: float
    max_dur: float

    def x(self, t: float) -> float:
        return self.x0 + max(0.0, min(t, self.max_dur)) / self.max_dur * self.width


@dataclass
class _Side:
    report: TeardownReport
    label: str
    color: Color
    frames: list[Optional[Path]]


def _load_thumb(path: Optional[Path]) -> Optional[Image.Image]:
    if path is None:
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except OSError:
        return None


def _draw_filmstrip(
    img: Image.Image, draw: ImageDraw.ImageDraw, side: _Side, axis: _Axis, top: int, height: int
) -> None:
    shots = side.report.shots
    for shot, frame in zip(shots, side.frames, strict=False):
        a = int(round(axis.x(shot.start_sec))) + (CUT_GAP // 2 if shot.index > 0 else 0)
        b = int(round(axis.x(shot.end_sec))) - (CUT_GAP - CUT_GAP // 2 if shot.index < len(shots) - 1 else 0)
        if b - a < 1:
            continue
        thumb = _load_thumb(frame)
        if thumb is None:
            draw.rectangle([a, top, b - 1, top + height - 1], fill=PLACEHOLDER)
            continue
        if b - a < 6:
            avg = thumb.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
            draw.rectangle([a, top, b - 1, top + height - 1], fill=avg)
            continue
        tile_w = max(1, int(round(height * thumb.width / thumb.height)))
        tile = thumb.resize((tile_w, height), Image.Resampling.LANCZOS)
        if b - a <= tile_w:  # narrower than one frame: show its centre, where the subject usually is
            left = (tile_w - (b - a)) // 2
            img.paste(tile.crop((left, 0, left + b - a, height)), (a, top))
            continue
        x = a
        while x < b:
            w = min(tile_w, b - x)
            img.paste(tile.crop((0, 0, w, height)), (x, top))
            x += w
    for c in side.report.cuts_sec:
        x = axis.x(c)
        draw.line([(x, top - 10), (x, top + height + 3)], fill=CUT, width=2)
        draw.polygon([(x - 6, top - 16), (x + 6, top - 16), (x, top - 8)], fill=CUT)


def _loudness_scale(reports: Sequence[TeardownReport]) -> tuple[float, float]:
    """Shared LUFS range for every lane on a card: the measured values plus a margin, at least 15 LU tall."""
    floor = teardown.LOUDNESS_FLOOR_LUFS
    values = [v for r in reports for _, v in r.loudness_curve if v > floor + 0.05]
    if not values:
        return -50.0, -10.0
    lo = math.floor((min(values) - 2) / 5) * 5
    hi = math.ceil((max(values) + 2) / 5) * 5
    if hi - lo < 15:
        lo = hi - 15
    return float(max(lo, floor)), float(min(hi, 0))


def _draw_loudness(
    draw: ImageDraw.ImageDraw, side: _Side, axis: _Axis, top: int, height: int, scale: tuple[float, float]
) -> None:
    report = side.report
    end = axis.x(report.info.duration_sec)
    draw.rectangle([axis.x0, top, end, top + height], fill=PANEL)
    if not report.loudness_curve:
        msg = "no audio" if not report.info.has_audio else "no loudness measured"
        _text(draw, (axis.x0 + 12, top + height / 2), msg, 22, MUTED, anchor="lm")
        return
    for seg in report.silences:
        draw.rectangle([axis.x(seg.start_sec), top, axis.x(seg.end_sec), top + height], fill=SILENCE)
    lo, hi = scale

    def y(v: float) -> float:
        v = max(lo, min(hi, v))
        return top + 4 + (1 - (v - lo) / (hi - lo)) * (height - 8)

    pts = [(axis.x(t), y(v)) for t, v in report.loudness_curve]
    if len(pts) == 1:
        pts = [(axis.x0, pts[0][1]), pts[0]]
    draw.line(pts, fill=side.color, width=3, joint="curve")


def _marker_items(report: TeardownReport) -> list[tuple[float, str, Color]]:
    items = []
    if report.product_first_seen_sec is not None:
        items.append((report.product_first_seen_sec, f"product {report.product_first_seen_sec:.1f} s", PRODUCT))
    if report.brand_first_said_sec is not None:
        items.append((report.brand_first_said_sec, f"said {report.brand_first_said_sec:.1f} s", BRAND))
    return items


def _draw_markers(
    draw: ImageDraw.ImageDraw, side: _Side, axis: _Axis, strip_top: int, lane_bottom: int, labels: bool
) -> None:
    """A line through the strip and lane at each moment, with a label under the lane when there is room (labels
    never overlap)."""
    placed_right = axis.x0 - 100.0
    for t, label, color in sorted(_marker_items(side.report)):
        x = axis.x(t)
        draw.line([(x, strip_top), (x, lane_bottom + 4)], fill=color, width=3)
        draw.polygon([(x - 6, lane_bottom + 4), (x + 6, lane_bottom + 4), (x, lane_bottom + 11)], fill=color)
        if not labels:
            continue
        w = _width(draw, label, 22)
        left = min(max(x - 8, axis.x0, placed_right + 16), axis.x0 + axis.width - w)
        _text(draw, (left, lane_bottom + 32), label, 22, color)
        placed_right = left + w


def _nice_step(max_dur: float) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if max_dur / step <= 8:
            return float(step)
    return 600.0


def _draw_axis(draw: ImageDraw.ImageDraw, axis: _Axis, y: int) -> None:
    draw.line([(axis.x0, y), (axis.x0 + axis.width, y)], fill=RULE, width=2)
    step = _nice_step(axis.max_dur)
    k = 0
    while k * step <= axis.max_dur + 1e-6:
        x = axis.x(k * step)
        draw.line([(x, y), (x, y + 8)], fill=MUTED, width=2)
        label = f"{k * step:g} s"
        anchor = "ls" if k == 0 else "ms"
        if x + _width(draw, label, 22) / 2 > axis.x0 + axis.width:
            anchor = "rs"
        _text(draw, (x, y + 32), label, 22, MUTED, anchor=anchor)
        k += 1


Glyph = Callable[[ImageDraw.ImageDraw, float, float], None]


def _glyph_cut(draw: ImageDraw.ImageDraw, x: float, cy: float) -> None:
    draw.line([(x + 11, cy - 6), (x + 11, cy + 12)], fill=CUT, width=2)
    draw.polygon([(x + 5, cy - 12), (x + 17, cy - 12), (x + 11, cy - 4)], fill=CUT)


def _glyph_line(colors: Sequence[Color]) -> Glyph:
    def draw_it(draw: ImageDraw.ImageDraw, x: float, cy: float) -> None:
        for i, color in enumerate(colors):
            dy = (i - (len(colors) - 1) / 2) * 7
            draw.line([(x, cy + 4 + dy), (x + 8, cy - 3 + dy), (x + 15, cy + 2 + dy), (x + 23, cy - 5 + dy)],
                      fill=color, width=3)

    return draw_it


def _glyph_box(color: Color) -> Glyph:
    def draw_it(draw: ImageDraw.ImageDraw, x: float, cy: float) -> None:
        draw.rectangle([x, cy - 9, x + 23, cy + 9], fill=color)

    return draw_it


def _glyph_marker(color: Color) -> Glyph:
    def draw_it(draw: ImageDraw.ImageDraw, x: float, cy: float) -> None:
        draw.line([(x + 11, cy - 12), (x + 11, cy + 12)], fill=color, width=3)

    return draw_it


def _legend_items(sides: Sequence[_Side], scale: tuple[float, float]) -> list[tuple[str, Glyph]]:
    reports = [s.report for s in sides]
    items: list[tuple[str, Glyph]] = [("cut", _glyph_cut)]
    if any(r.loudness_curve for r in reports):
        items.append((f"loudness, {scale[0]:g} to {scale[1]:g} LUFS",
                      _glyph_line([s.color for s in sides if s.report.loudness_curve])))
    if any(r.silences for r in reports):
        items.append(("silence", _glyph_box(SILENCE)))
    if any(r.product_first_seen_sec is not None for r in reports):
        items.append(("product first seen", _glyph_marker(PRODUCT)))
    if any(r.brand_first_said_sec is not None for r in reports):
        items.append(("brand first said", _glyph_marker(BRAND)))
    return items


def _legend_rows(draw: ImageDraw.ImageDraw, items: Sequence[tuple[str, Glyph]]) -> list[list[tuple[str, Glyph]]]:
    rows: list[list[tuple[str, Glyph]]] = [[]]
    x = 0.0
    for label, glyph in items:
        w = LEGEND_GLYPH + _width(draw, label, 22) + LEGEND_SPACE
        if rows[-1] and x + w - LEGEND_SPACE > INNER:
            rows.append([])
            x = 0.0
        rows[-1].append((label, glyph))
        x += w
    return rows


def _draw_legend(draw: ImageDraw.ImageDraw, y: int, rows: Sequence[Sequence[tuple[str, Glyph]]]) -> None:
    for row in rows:
        x = float(PAD)
        for label, glyph in row:
            glyph(draw, x, y + 18)
            _text(draw, (x + LEGEND_GLYPH, y + 26), label, 22, MUTED)
            x += LEGEND_GLYPH + _width(draw, label, 22) + LEGEND_SPACE
        y += LEGEND_LINE


def _strips_height(n_sides: int, strip_h: int, style: _Style, markers: bool, legend_rows: int) -> int:
    per_side = LABEL_H + strip_h + STRIP_GAP + style.lane_h + (style.marker_h if markers else 0)
    return n_sides * per_side + (n_sides - 1) * SIDE_GAP + AXIS_GAP + AXIS_H + legend_rows * LEGEND_LINE


def _draw_strips(
    img: Image.Image, draw: ImageDraw.ImageDraw, sides: Sequence[_Side], y: int, strip_h: int, style: _Style,
    legend: Sequence[Sequence[tuple[str, Glyph]]], scale: tuple[float, float],
) -> int:
    """Draws every side, the shared axis and the legend from `y`; returns the y below the legend."""
    max_dur = max((s.report.info.duration_sec for s in sides), default=1.0) or 1.0
    axis = _Axis(PAD, INNER, max_dur)
    markers = any(_marker_items(s.report) for s in sides)
    for i, side in enumerate(sides):
        if i:
            y += SIDE_GAP
        if len(sides) > 1:
            draw.rectangle([PAD, y + 8, PAD + 16, y + 24], fill=side.color)
            _text(draw, (PAD + 28, y + 26), side.label, 26, TEXT, bold=True)
            _text(draw, (W - PAD, y + 26), _side_stats(side.report), 22, MUTED, anchor="rs")
        else:  # the title already names the video
            _text(draw, (PAD, y + 26), _side_stats(side.report), 24, MUTED)
        strip_top = y + LABEL_H
        _draw_filmstrip(img, draw, side, axis, strip_top, strip_h)
        lane_top = strip_top + strip_h + STRIP_GAP
        _draw_loudness(draw, side, axis, lane_top, style.lane_h, scale)
        y = lane_top + style.lane_h
        if markers:
            _draw_markers(draw, side, axis, strip_top, y, style.marker_h >= MARKER_LABEL_MIN_H)
            y += style.marker_h
    y += AXIS_GAP
    _draw_axis(draw, axis, y)
    y += AXIS_H
    _draw_legend(draw, y, legend)
    return y + len(legend) * LEGEND_LINE


# --------------------------------------------------------------------------------------------------------------------
# Table, title, footer


@dataclass
class _Cell:
    text: str
    color: Color = TEXT
    metric: bool = False  # metric-name cell (smaller, muted)


def _cell_px(cell: _Cell, style: _Style) -> int:
    return style.metric_px if cell.metric else style.value_px


def _row_heights(
    draw: ImageDraw.ImageDraw, rows: Sequence[Sequence[_Cell]], widths: Sequence[int], style: _Style
) -> list[int]:
    out = []
    for row in rows:
        lines = max(len(_wrap(draw, c.text, _cell_px(c, style), w - 18)) for c, w in zip(row, widths, strict=True))
        out.append(2 * style.row_pad + lines * (style.value_px + 8))
    return out


def _draw_table(
    draw: ImageDraw.ImageDraw, y: int, widths: Sequence[int], header: Sequence[_Cell],
    rows: Sequence[Sequence[_Cell]], heights: Sequence[int], style: _Style,
) -> int:
    x = float(PAD)
    for cell, w in zip(header, widths, strict=True):
        _text(draw, (x, y + 26), cell.text, 22, cell.color, bold=True)
        x += w
    y += TABLE_HEAD
    draw.line([(PAD, y), (W - PAD, y)], fill=RULE, width=2)
    for i, (row, h) in enumerate(zip(rows, heights, strict=True)):
        if i % 2 == 0:
            draw.rectangle([PAD - 12, y + 1, W - PAD + 12, y + h], fill=PANEL)
        x = float(PAD)
        for cell, w in zip(row, widths, strict=True):
            px = _cell_px(cell, style)
            line_mid = y + style.row_pad + (style.value_px + 8) / 2
            for line in _wrap(draw, cell.text, px, w - 18):
                _text(draw, (x, line_mid), line, px, cell.color, anchor="lm")
                line_mid += style.value_px + 8
            x += w
        y += h
    return y


def _title_height() -> int:
    return OVERLINE_Y + TITLE_STEP + AFTER_TITLE


def _draw_title(draw: ImageDraw.ImageDraw, overline: str, parts: Sequence[tuple[str, Color, bool]]) -> None:
    _text(draw, (PAD, OVERLINE_Y), overline, 22, MUTED)
    size = _fit_size(draw, "".join(p for p, _, _ in parts), 54, INNER, floor=30)
    x = float(PAD)
    for text, color, bold in parts:
        x += _text(draw, (x, OVERLINE_Y + TITLE_STEP), text, size, color, bold=bold)


def _threshold_text(reports: Sequence[TeardownReport], key: str) -> str:
    seen: list[str] = []
    for r in reports:
        v = r.thresholds.get(key)
        s = "?" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
        if s not in seen:
            seen.append(s)
    return " / ".join(seen)


def method_text(reports: Sequence[TeardownReport]) -> str:
    """The method line printed under every card: tool, thresholds and filters actually used."""

    def t(key: str) -> str:
        return _threshold_text(reports, key)

    return (
        f"ffmpeg {t('ffmpeg_version')}: scdet cuts at threshold {t('scdet_threshold')} ({t('analysis_fps')} fps), "
        f"merged under {t('min_shot_sec')} s; pixel change = scdet mafd (0-100); silencedetect {t('silence_db')} dB "
        f"for {t('silence_min_sec')} s; ebur128 loudness (EBU R128), short-term line."
    )


def _footer_lines(draw: ImageDraw.ImageDraw, reports: Sequence[TeardownReport], run_ids: Sequence[str]) -> list[str]:
    return _wrap(draw, method_text(reports), 22, INNER) + _wrap(draw, "Runs " + ", ".join(run_ids), 22, INNER)


def _footer_height(lines: Sequence[str]) -> int:
    return FOOTER_HEAD + FOOTER_LINE * len(lines) + BOTTOM


def _draw_footer(draw: ImageDraw.ImageDraw, lines: Sequence[str]) -> None:
    y = H - _footer_height(lines)
    draw.line([(PAD, y), (W - PAD, y)], fill=RULE, width=2)
    y += FOOTER_HEAD
    _text(draw, (PAD, y), "Measured, not predicted.", 26, TEXT, bold=True)
    for line in lines:
        y += FOOTER_LINE
        _text(draw, (PAD, y), line, 22, MUTED)


# --------------------------------------------------------------------------------------------------------------------
# Values


def _value_text(report: TeardownReport, metric: str, value: Any) -> str:
    if value is None:
        if metric in ("Integrated loudness", "First sound") and not report.info.has_audio:
            return "no audio"
        if metric == "First cut":
            return "none (one shot)"
        if metric == "Integrated loudness":
            return "not measured"
    return teardown.format_value(metric, value)


def _diff_text(diff: str, yours_label: str, theirs_label: str) -> str:
    if yours_label != "Yours":
        diff = diff.replace("in yours", f"in {yours_label}")
    if theirs_label != "Reference":
        diff = diff.replace("in reference", f"in {theirs_label}")
    return diff


def _side_stats(report: TeardownReport) -> str:
    n = len(report.shots)
    parts = [f"{report.info.duration_sec:.1f} s", f"{n} shot{'s' if n != 1 else ''}"]
    if report.integrated_lufs is not None:
        parts.append(f"{report.integrated_lufs:.1f} LUFS")
    elif not report.info.has_audio:
        parts.append("no audio")
    return " · ".join(parts)


# --------------------------------------------------------------------------------------------------------------------
# Page assembly


def _render(
    sides: Sequence[_Side], overline: str, title: Sequence[tuple[str, Color, bool]], widths: Sequence[int],
    header: Sequence[_Cell], rows: Sequence[Sequence[_Cell]], run_ids: Sequence[str], *, min_strip: int,
    max_strip: int,
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    reports = [s.report for s in sides]
    scale = _loudness_scale(reports)
    legend = _legend_rows(draw, _legend_items(sides, scale))
    markers = any(_marker_items(r) for r in reports)
    footer = _footer_lines(draw, reports, run_ids)
    # Densest style last: take the first that leaves the filmstrips at least `min_strip` px tall.
    for style in STYLES:
        heights = _row_heights(draw, rows, widths, style)
        fixed = (_title_height() + _strips_height(len(sides), 0, style, markers, len(legend)) + AFTER_STRIPS
                 + TABLE_HEAD + sum(heights) + AFTER_TABLE + _footer_height(footer))
        strip_h = min(max_strip, (H - fixed) // len(sides))
        if strip_h >= min_strip:
            break
    spare = H - fixed - strip_h * len(sides)
    if spare > 0 and strip_h >= max_strip:  # room left over: a taller loudness lane, up to 60 px more
        style = replace(style, lane_h=style.lane_h + min(60, spare // len(sides)))
    strip_h = max(strip_h, 48)
    _draw_title(draw, overline, title)
    y = _draw_strips(img, draw, sides, _title_height(), strip_h, style, legend, scale)
    _draw_table(draw, y + AFTER_STRIPS, widths, header, rows, heights, style)
    _draw_footer(draw, footer)
    return img


def _save(img: Image.Image, out_png: teardown.PathLike) -> str:
    out = Path(out_png).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG", optimize=True)
    return str(out)


def render_compare_card(
    report: CompareReport,
    out_png: teardown.PathLike,
    *,
    yours_label: str = "Yours",
    theirs_label: str = "Reference",
    workdir: Optional[teardown.PathLike] = None,
) -> str:
    """Draws the Compare card (1080 x 1350 PNG) and returns its path; also sets ``report.card_path``.

    Thumbnails are extracted with ffmpeg into `workdir` (a temporary folder when None); pass analyze()'s workdir to
    reuse its analysis copies. A video that has moved since the analysis gets grey placeholders, not an error.
    """
    widths = [250, 200, 200, INNER - 250 - 2 * 200]
    header = [_Cell("Metric", MUTED), _Cell(yours_label, YOURS), _Cell(theirs_label, THEIRS),
              _Cell("Difference", MUTED)]
    rows = [
        [
            _Cell(r.metric, MUTED, metric=True),
            _Cell(_value_text(report.yours, r.metric, r.yours)),
            _Cell(_value_text(report.theirs, r.metric, r.theirs)),
            _Cell(_diff_text(r.diff, yours_label, theirs_label)),
        ]
        for r in report.rows
    ]
    title = [(yours_label, YOURS, True), ("  vs  ", MUTED, False), (theirs_label, THEIRS, True)]
    with _workdir(workdir) as work:  # thumbnails must outlive the drawing, so render inside
        sides = [
            _Side(report.yours, yours_label, YOURS, teardown.extract_shot_frames(report.yours, work)),
            _Side(report.theirs, theirs_label, THEIRS, teardown.extract_shot_frames(report.theirs, work)),
        ]
        img = _render(sides, "SCREEN-TIME  ·  COMPARE", title, widths, header, rows,
                      [report.run_id, report.yours.run_id, report.theirs.run_id], min_strip=96, max_strip=170)
    path = _save(img, out_png)
    report.card_path = path
    return path


def _teardown_stats(r: TeardownReport) -> list[tuple[str, str]]:
    silence_total = sum(s.end_sec - s.start_sec for s in r.silences)
    pc = teardown.mean_pixel_change(r)
    stats = [
        ("Duration", f"{r.info.duration_sec:.1f} s"),
        ("Shots", str(len(r.shots))),
        ("First cut", _value_text(r, "First cut", r.first_cut_sec)),
        ("Cuts in first 3 s", str(r.cuts_first_3s)),
        ("Cuts per 10 s", teardown.format_value("Cuts per 10 s", r.cuts_per_10s)),
        ("Median shot", teardown.format_value("Median shot", r.median_shot_sec)),
        ("Longest shot", teardown.format_value("", teardown.longest_shot_text(r))),
        ("Pixel change", "none" if pc is None else f"{pc:.1f}"),
        ("Loudness", _value_text(r, "Integrated loudness", r.integrated_lufs)),
        ("Loudness range", "none" if r.loudness_range_lu is None else f"{r.loudness_range_lu:.1f} LU"),
        ("First sound", _value_text(r, "First sound", r.first_sound_sec)),
        ("Silences", "none" if not r.silences else f"{len(r.silences)}, {silence_total:.1f} s in all"),
    ]
    if r.product_first_seen_sec is not None:
        stats.append(("Product first seen", f"{r.product_first_seen_sec:.1f} s"))
    if r.brand_first_said_sec is not None:
        stats.append(("Brand first said", f"{r.brand_first_said_sec:.1f} s"))
    return stats


def render_teardown_card(
    report: TeardownReport,
    out_png: teardown.PathLike,
    *,
    label: str = "Teardown",
    workdir: Optional[teardown.PathLike] = None,
) -> str:
    """Draws the single-video Teardown card (1080 x 1350 PNG) and returns its path; also sets ``report.card_path``."""
    stats = _teardown_stats(report)
    half = (len(stats) + 1) // 2
    rows = []
    for i in range(half):
        cells: list[_Cell] = []
        for name, value in stats[i:i + 1] + stats[half + i:half + i + 1]:
            cells += [_Cell(name, MUTED, metric=True), _Cell(value)]
        rows.append(cells + [_Cell("")] * (4 - len(cells)))
    widths = [230, INNER // 2 - 230, 230, INNER - INNER // 2 - 230]
    header = [_Cell("Metric", MUTED), _Cell("Measured", MUTED), _Cell("Metric", MUTED), _Cell("Measured", MUTED)]
    with _workdir(workdir) as work:  # thumbnails must outlive the drawing, so render inside
        side = _Side(report, label, YOURS, teardown.extract_shot_frames(report, work))
        img = _render([side], "SCREEN-TIME  ·  TEARDOWN", [(label, TEXT, True)], widths, header, rows,
                      [report.run_id], min_strip=160, max_strip=300)
    path = _save(img, out_png)
    report.card_path = path
    return path
