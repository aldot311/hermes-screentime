"""Plain-text summaries of results, the same on every surface (Telegram, Desktop, CLI). No em dashes."""

from __future__ import annotations

from typing import Any, Optional

from .models import CompareReport, MeasureReport, TeardownReport


def _s(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f} s"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def short_label(product: str, limit: int = 64) -> str:
    """A short name for headlines: the description up to its first colon, comma or semicolon, without 'a/an/the'."""
    text = " ".join((product or "").split())
    for sep in (":", ";", ","):
        if sep in text:
            text = text.split(sep, 1)[0]
    for article in ("a ", "an ", "the "):
        if text.lower().startswith(article):
            text = text[len(article):]
    return (text[: limit - 1] + "\u2026") if len(text) > limit else text


def measure_summary(r: MeasureReport) -> str:
    m = r.metrics
    headline = (
        f"{short_label(r.product) or r.prompt}: {m.seconds_on_screen:.1f} s on screen of {r.info.duration_sec:.1f} s "
        f"({_pct(m.share_of_video)})"
    )
    lines = [headline]
    if m.first_seen_sec is not None:
        lines.append(
            f"First seen {_s(m.first_seen_sec)} · longest continuous run {_s(m.longest_run_sec)}"
            + (f" (from {_s(m.longest_run_start_sec)})" if m.longest_run_start_sec is not None else "")
            + f" · {len(m.segments)} segment{'s' if len(m.segments) != 1 else ''}"
        )
        if m.peak_area_fraction:
            lines.append(
                f"Size in frame: average {_pct(m.avg_area_fraction)}, peak {_pct(m.peak_area_fraction)}"
                + (f" at {_s(m.peak_area_sec)}" if m.peak_area_sec is not None else "")
            )
    if m.mentions:
        said = ", ".join(f"{x.start_sec:.1f} s" for x in m.mentions[:6])
        lines.append(f"Said out loud: {len(m.mentions)} time{'s' if len(m.mentions) != 1 else ''} (about {said})")
    elif m.spoken_checked is False:
        lines.append("Speech: not checked (the audio could not be transcribed)")
    counted = [t for t in r.tracks if t.verdict == "match"]
    rejected = [t for t in r.tracks if t.verdict == "lookalike"]
    unclear = [t for t in r.tracks if t.verdict == "unclear"]
    lines.append(
        f"Checked {r.info.frame_count} frames at {r.info.fps:g} fps. Objects tracked: {len(r.tracks)}; counted "
        f"{len(counted)}, look-alikes rejected {len(rejected)}, unclear (not counted) {len(unclear)}."
    )
    for t in rejected + unclear:
        lines.append(f"  object {t.object_id + 1}: {t.verdict}: {t.reason}")
    for tr in r.terms:
        mark = {True: "PASS", False: "FAIL", None: "NEEDS A PERSON"}[tr.passed]
        lines.append(f"{mark}  {tr.term.text}: {tr.detail}")
    if r.terms:
        if any(t.passed is False for t in r.terms):
            verdict = "HOLD (a term failed)"
        elif any(t.passed is None for t in r.terms):
            verdict = "NEEDS REVIEW"
        else:
            verdict = "DELIVERED (every term passed)"
        lines.append(f"Verdict: {verdict}")
    check = r.verifier + ("" if r.verifier_validated else " (not validated on this task)")
    cost = r.receipt.usd if r.receipt else 0.0
    replay = " Replay of a recorded run: no API calls." if r.engine == "replay" else ""
    lines.append(f"Product check: {check}. Cost: ${cost:.3f}. Run {r.run_id}.{replay}")
    for n in r.notes:
        lines.append(f"Note: {n}")
    return "\n".join(lines)


def teardown_summary(t: TeardownReport) -> str:
    lines = [
        f"{t.info.duration_sec:.1f} s, {len(t.shots)} shot{'s' if len(t.shots) != 1 else ''}; first cut "
        + (_s(t.first_cut_sec) if t.first_cut_sec is not None else "none (single shot)")
        + f"; {t.cuts_first_3s} cut{'s' if t.cuts_first_3s != 1 else ''} in the first 3 s",
        f"Median shot {_s(t.median_shot_sec)}, longest {_s(t.longest_shot_sec)}"
        + (f" at {_s(t.longest_shot_start_sec)}" if t.longest_shot_start_sec is not None else ""),
    ]
    if t.integrated_lufs is not None:
        lines.append(f"Loudness {t.integrated_lufs:.1f} LUFS; first sound {_s(t.first_sound_sec)}")
    else:
        lines.append("No audio track")
    return "\n".join(lines)


def compare_summary(c: CompareReport) -> str:
    from .teardown import format_value

    lines = ["Metric | yours | reference | difference"]
    for row in c.rows:
        lines.append(f"{row.metric} | {format_value(row.metric, row.yours)} | {format_value(row.metric, row.theirs)} | {row.diff}")
    lines.append("Measured, not predicted: no score, no retention or virality claims.")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}"
    if isinstance(v, (list, tuple)):
        return "/".join(_fmt(x) for x in v)
    return str(v)


def media_instructions(path: str) -> str:
    return (
        "To show the user this file, put the media_tag value on its own line in your reply, exactly as given "
        "(chat apps attach it). In a plain terminal, give the path instead."
    )
