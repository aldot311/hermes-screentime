"""Contract terms checked against a video's measurements: a port of Screen-Time's src/lib/metrics/rules.ts.

A term is plain English plus a typed rule (RULE_KINDS) that a language model fills from RULE_JSON_SCHEMA. Each result
is three-state: True (met), False (not met) or None (needs a person: not checkable, or the evidence is unknown), with
the moments that prove it (evidence segments to jump to) and a plain-English detail. Thresholds, the 1e-6 tolerance
and the rounding of every number in a detail are the TypeScript's (tests/test_rules.py replays its outputs).

Deliberate differences from the TypeScript:
- min_continuous uses the strict longest run (metrics.longest_run_sec, no gap bridging), and its evidence is that run.
- "disclosed" reads only the transcript in v0.1 (no frames are read for on-screen #ad): a spoken disclosure passes;
  nothing said, or audio not transcribed, is None with a detail saying on-screen text was not checked. The TypeScript
  also read sampled frames and failed the term when neither showed anything.
- "mentioned" is None when the audio was not transcribed: spoken_checked False, or None on a video that has sound.
  The TypeScript treated a missing flag as "transcribed".
- Rules are validated first (validate_rule); an invalid one is None with the reason instead of being evaluated.
- Per-frame data comes from metrics.visible_frames (always set by metrics.compute) or from ``samples``. When the
  metrics show nothing on screen, an empty frame list is known data (the TypeScript, given no frames, said "at most
  0 s" and left min_area unchecked).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Optional

from .metrics import FrameSample, frames_to_segments, video_fps
from .models import ContractTerm, Disclosure, Mention, Segment, TermResult, VideoInfo, VideoMetrics

EPS = 1e-6
"""Seconds are multiples of 1/fps, so float sums can land a hair under a whole-number threshold."""

RULE_KINDS: dict[str, dict[str, Any]] = {
    "min_seconds_total": {
        "fields": ("seconds",),
        "optional": (),
        "means": "the product is on screen for at least `seconds` in total",
    },
    "min_seconds_within": {
        "fields": ("seconds", "within_first_sec"),
        "optional": (),
        "means": "the product is on screen for at least `seconds` within the first `within_first_sec` seconds",
    },
    "min_continuous": {
        "fields": ("seconds",),
        "optional": (),
        "means": "one continuous appearance of at least `seconds`, every frame of it seen",
    },
    "min_area": {
        "fields": ("fraction", "for_seconds"),
        "optional": (),
        "means": "the product fills at least `fraction` (0 to 1) of the frame for at least `for_seconds` in total",
    },
    "mentioned": {
        "fields": ("min_times",),
        "optional": ("within_first_sec",),
        "means": "the brand name is said aloud at least `min_times` times, within the first `within_first_sec` "
        "seconds when given",
    },
    "first_seen_before": {
        "fields": ("seconds",),
        "optional": (),
        "means": "the product first appears before `seconds`",
    },
    "disclosed": {
        "fields": (),
        "optional": (),
        "means": "the creator says the video is an ad or sponsored",
    },
}
"""Every rule kind: the number fields it needs, the ones it may take, and what it means."""

_NUMBER_FIELDS = ("seconds", "within_first_sec", "fraction", "for_seconds", "min_times")
_CAMEL_CASE = {"withinFirstSec": "within_first_sec", "forSeconds": "for_seconds", "minTimes": "min_times"}

RULE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "One machine-checkable contract term. Fill only the number fields its kind uses and leave the rest out. "
        + " ".join(
            f"{kind}: {spec['means']}"
            + (f" (needs {', '.join(spec['fields'])}" if spec["fields"] else "")
            + (f"; optional {', '.join(spec['optional'])}" if spec["optional"] else "")
            + (")" if spec["fields"] else "")
            + "."
            for kind, spec in RULE_KINDS.items()
        )
    ),
    "properties": {
        "kind": {"type": "string", "enum": list(RULE_KINDS), "description": "Which check this term is."},
        "seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Seconds, greater than 0 (min_seconds_total, min_seconds_within, min_continuous, "
            "first_seen_before).",
        },
        "within_first_sec": {
            "type": "number",
            "minimum": 0,
            "description": "Length of the opening window in seconds, greater than 0 (min_seconds_within; optional "
            "for mentioned, left out means anywhere in the video).",
        },
        "fraction": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Share of the frame from 0 to 1, greater than 0: 0.1 is 10% (min_area).",
        },
        "for_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Seconds at or above that size, in total, greater than 0 (min_area).",
        },
        "min_times": {
            "type": "integer",
            "minimum": 1,
            "description": "How many times the brand name must be said (mentioned).",
        },
    },
    "required": ["kind"],
    "additionalProperties": False,
}
"""JSON schema for one typed rule, flat so that tool-calling models that reject oneOf can fill it."""


class _RuleError(ValueError):
    pass


def validate_rule(rule: Any) -> tuple[bool, Optional[str]]:
    """(True, None) for a rule evaluate() can check, else (False, the reason in plain English).

    Rejects unknown kinds, missing fields, numbers that are not finite and greater than 0, a fraction above 1, a
    min_times that is not a whole number, and a window or duration in a field the kind does not use (it would be
    dropped and loosen the term, as the TypeScript's parser also refused). An unused field set to 0 or null is ignored.
    """
    try:
        _normalized(rule)
    except _RuleError as exc:
        return False, str(exc)
    return True, None


def describe_rule(rule: Any) -> str:
    """How the rule was read, in plain English, for the user to confirm ("at least 3.0 s on screen in total")."""
    if rule is None:
        return "not checkable as written"
    try:
        r = _normalized(rule)
    except _RuleError as exc:
        return f"not checkable as written ({exc})"
    kind = r["kind"]
    if kind == "min_seconds_total":
        return f"at least {_amount(r['seconds'])} s on screen in total"
    if kind == "min_seconds_within":
        return f"at least {_amount(r['seconds'])} s on screen within the first {_amount(r['within_first_sec'])} s"
    if kind == "min_continuous":
        return f"at least {_amount(r['seconds'])} s on screen in one continuous run"
    if kind == "min_area":
        return f"at least {_pct(r['fraction'])} of the frame for at least {_amount(r['for_seconds'])} s in total"
    if kind == "mentioned":
        window = f" within the first {_amount(r['within_first_sec'])} s" if "within_first_sec" in r else ""
        return f"brand name said aloud at least {_times(r['min_times'])}{window}"
    if kind == "first_seen_before":
        return f"first on screen before {_amount(r['seconds'])} s"
    return "sponsorship disclosed out loud (“ad”, “sponsored”, “paid partnership”); on-screen text is not read in this version"


def evaluate(
    terms: Sequence[ContractTerm],
    metrics: VideoMetrics,
    info: VideoInfo,
    disclosure: Optional[Disclosure] = None,
    *,
    samples: Optional[Sequence[FrameSample]] = None,
) -> list[TermResult]:
    """Check each term against the metrics, in order.

    ``samples`` (metrics.frame_samples of the counted detections) gives min_area its per-frame sizes. Without it,
    min_area uses metrics.visible_areas when that field exists, and is otherwise None (needs a person).
    """
    fps = video_fps(info)
    frames, sized = _per_frame(metrics, samples)
    results: list[TermResult] = []
    for term in terms:
        if not term.rule:
            results.append(TermResult(term, None, [], "Could not be turned into a checkable rule."))
            continue
        try:
            rule = _normalized(term.rule)
        except _RuleError as exc:
            results.append(TermResult(term, None, [], f"Could not be checked: {exc}."))
            continue
        passed, evidence, detail = _check(rule, metrics, info, fps, disclosure, frames, sized)
        results.append(TermResult(term, passed, evidence, detail))
    return results


def clock(sec: float) -> str:
    """1.25 -> "0:01.2", 75.5 -> "1:15.5"; tenths round down like the other details."""
    tenths = math.floor(max(0.0, sec) * 10 + EPS)
    minutes = tenths // 600
    rest = (tenths % 600) / 10
    return f"{minutes}:{'0' if rest < 10 else ''}{rest:.1f}"


def clip_segments(segments: Sequence[Segment], limit_sec: float) -> list[Segment]:
    """The parts of the segments that fall inside [0, limit_sec)."""
    clipped = [
        Segment(max(0.0, s.start_sec), min(s.end_sec, limit_sec)) for s in segments if s.start_sec < limit_sec
    ]
    return [s for s in clipped if s.end_sec > s.start_sec]


# ----------------------------------------------------------------------------------------------------------------------
# Checks


_Outcome = tuple[Optional[bool], list[Segment], str]


def _check(
    r: dict[str, Any],
    m: VideoMetrics,
    info: VideoInfo,
    fps: float,
    disclosure: Optional[Disclosure],
    frames: Optional[list[int]],
    sized: Optional[list[FrameSample]],
) -> _Outcome:
    kind = r["kind"]
    if kind == "min_seconds_total":
        return (
            _at_least(m.seconds_on_screen, r["seconds"]),
            list(m.segments),
            f"On screen {_fmt(m.seconds_on_screen)} s in total (needs {_need(r['seconds'])} s).",
        )

    if kind == "min_seconds_within":
        limit, need = r["within_first_sec"], r["seconds"]
        in_window = clip_segments(m.segments, limit)
        evidence = in_window if in_window else list(m.segments[:1])
        window = f"in the first {_need(limit)} s (needs {_need(need)} s)"
        if frames is not None:
            seconds = _seconds_within(frames, limit, fps)
            return _at_least(seconds, need), evidence, f"On screen {_fmt(seconds)} s {window}."
        # Without per-frame data only an upper bound is known: bridged segments include gaps that were not detected.
        # It can prove a fail, never a pass.
        upper = min(_sum_lengths(in_window), m.seconds_on_screen)
        if not _at_least(upper, need):
            return False, evidence, f"On screen at most {_fmt(upper)} s {window}."
        return None, evidence, f"Needs per-frame data, which was not provided: segments allow up to {_fmt(upper)} s {window}."

    if kind == "min_continuous":
        # Deliberate difference: the strict run (no bridged gaps) and its own moment as evidence; the TypeScript used
        # its bridged longest run and pointed at the longest bridged segment.
        run, start = m.longest_run_sec, m.longest_run_start_sec
        return (
            _at_least(run, r["seconds"]),
            [Segment(start, start + run)] if start is not None else [],
            f"Longest continuous run {_fmt(run)} s (needs {_need(r['seconds'])} s).",
        )

    if kind == "min_area":
        if sized is None:
            return None, [], "Needs per-frame size data, which was not provided."
        fraction, need = r["fraction"], r["for_seconds"]
        big = [s for s in sized if s.area_fraction + EPS >= fraction]
        seconds = len(big) / fps
        evidence = frames_to_segments([s.frame for s in big], fps=fps) if big else _closest_to_area(sized, fps)
        return (
            _at_least(seconds, need),
            evidence,
            f"At least {_pct(fraction)} of the frame for {_fmt(seconds)} s (needs {_need(need)} s).",
        )

    if kind == "mentioned":
        # No transcript is not "never said": the term waits for a person.
        if m.spoken_checked is False:
            return None, [], "Not checked: the audio could not be transcribed."
        if m.spoken_checked is None and info.has_audio:
            # Deliberate difference: the TypeScript read a missing flag as "transcribed".
            return None, [], "Not checked: the audio was not transcribed for brand names."
        within = r.get("within_first_sec")
        counted = list(m.mentions) if within is None else [x for x in m.mentions if x.start_sec < within]
        where = "" if within is None else f" in the first {_need(within)} s"
        return (
            len(counted) >= r["min_times"],
            [_mention_segment(x) for x in (counted or m.mentions[:1])],
            f"Mentioned {_times(len(counted))}{where} (needs {_times(r['min_times'])}).",
        )

    if kind == "first_seen_before":
        first, before = m.first_seen_sec, r["seconds"]
        detail = (
            f"Never seen (needs to appear before {_need(before)} s)."
            if first is None
            else f"First seen at {_fmt(first)} s (needs to appear before {_need(before)} s)."
        )
        return first is not None and first < before - EPS, list(m.segments[:1]), detail

    return _check_disclosed(disclosure, info)


def _check_disclosed(d: Optional[Disclosure], info: VideoInfo) -> _Outcome:
    """Disclosed when the creator says it ("ad", "sponsored", "paid partnership"). Deliberate difference: v0.1 reads no
    frames for on-screen text (#ad), so nothing said leaves the term to a person instead of failing it."""
    if d is None:
        return None, [], "Not checked: the transcript was not searched for a disclosure on this run."
    if d.spoken:
        evidence = sorted((_mention_segment(x) for x in d.spoken), key=lambda s: s.start_sec)
        return True, evidence, f"Disclosed: said “{d.spoken[0].text}” at {clock(d.spoken[0].start_sec)}"
    not_read = "on-screen text such as #ad is not read in this version"
    if not info.has_audio:
        return None, [], f"Not checked: the video has no sound, and {not_read}."
    if d.spoken_checked is False:
        return None, [], f"Not checked: the audio could not be transcribed, and {not_read}."
    if d.spoken_checked is None:
        return None, [], f"Not checked: the audio was not transcribed, and {not_read}."
    return None, [], f"Not said out loud (no “ad” or “sponsored” heard); {not_read}, so a person should check the video."


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _normalized(rule: Any) -> dict[str, Any]:
    """The rule with only its kind's fields, as floats (min_times as int); raises _RuleError when it is not valid."""
    if not isinstance(rule, Mapping):
        raise _RuleError("a rule must be an object with a kind")
    kind = rule.get("kind")
    if not isinstance(kind, str) or kind not in RULE_KINDS:
        raise _RuleError(f"unknown kind {kind!r}; use one of {', '.join(RULE_KINDS)}")
    spec = RULE_KINDS[kind]
    uses = set(spec["fields"]) | set(spec["optional"])
    for key, value in rule.items():
        if key == "kind" or key in uses:
            continue
        if key in _CAMEL_CASE:
            raise _RuleError(f"write {_CAMEL_CASE[key]} instead of {key}")
        if key not in _NUMBER_FIELDS:
            raise _RuleError(f"unknown field {key!r}")
        if value is None or (_is_number(value) and value == 0):
            continue  # an unused field left empty
        raise _RuleError(f"{kind} does not use {key}; a {key} here would be dropped and loosen the term")
    out: dict[str, Any] = {"kind": kind}
    for name in spec["fields"]:
        if rule.get(name) is None:
            raise _RuleError(f"{kind} needs {name}")
        out[name] = _rule_number(name, rule[name])
    for name in spec["optional"]:
        value = rule.get(name)
        if value is None or (_is_number(value) and value == 0):
            continue  # mentioned without a window: anywhere in the video (0 meant the same to the TypeScript's parser)
        out[name] = _rule_number(name, value)
    return out


def _rule_number(name: str, value: Any) -> Any:
    if not _is_number(value):
        raise _RuleError(f"{name} must be a number")
    if not math.isfinite(value):
        raise _RuleError(f"{name} must be a finite number")
    if value < 0:
        raise _RuleError(f"{name} must not be negative")
    if value == 0:
        raise _RuleError(f"{name} must be greater than 0")
    if name == "fraction" and value > 1:
        raise _RuleError("fraction is a share of the frame from 0 to 1 (10% is 0.1)")
    if name == "min_times":
        if not float(value).is_integer():
            raise _RuleError("min_times must be a whole number")
        return int(value)
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _per_frame(
    m: VideoMetrics, samples: Optional[Sequence[FrameSample]]
) -> tuple[Optional[list[int]], Optional[list[FrameSample]]]:
    """(counted frame indices, per-frame sizes); either is None when not known."""
    if samples is not None:
        return [s.frame for s in samples], list(samples)
    frames = list(m.visible_frames or [])
    if frames:
        areas = getattr(m, "visible_areas", None)
        if areas and len(areas) == len(frames):
            return frames, [FrameSample(f, a) for f, a in zip(frames, areas, strict=True)]
        return frames, None
    if m.seconds_on_screen <= 0:
        return [], []  # nothing on screen: an empty frame list is the per-frame data
    return None, None  # seconds without frames: hand-built or older metrics


def _seconds_within(frames: Sequence[int], limit: float, fps: float) -> float:
    """Counted seconds in [0, limit). Each counted frame is 1/fps, as in seconds_on_screen; a frame that straddles the
    window end counts only its overlap, so for a window on a frame boundary this is the frames before it / fps."""
    total = 0.0
    for f in frames:
        start = f / fps
        total += max(0.0, min(start + 1 / fps, limit) - start)
    return total


def _closest_to_area(sized: Sequence[FrameSample], fps: float) -> list[Segment]:
    """For a failed area rule, point at the frame where the product was largest (the first one, on a tie)."""
    if not sized:
        return []
    largest = sized[0]
    for s in sized[1:]:
        if s.area_fraction > largest.area_fraction:
            largest = s
    return frames_to_segments([largest.frame], fps=fps)


def _mention_segment(x: Mention) -> Segment:
    return Segment(x.start_sec, x.end_sec)


def _sum_lengths(segments: Sequence[Segment]) -> float:
    total = 0.0
    for s in segments:
        total += s.end_sec - s.start_sec
    return total


def _at_least(value: float, needed: float) -> bool:
    return value + EPS >= needed


def _fmt(n: float) -> str:
    """A measured number, rounded down to tenths so a near miss never shows as the threshold it missed (2.96 -> 2.9)."""
    if not math.isfinite(n):
        return _js_number(n)
    return _js_number(math.floor(n * 10 + EPS) / 10)


def _need(n: float) -> str:
    """A contract threshold as written, to 0.01, never rounded down: 0.25 s must not read as "needs 0.2 s"."""
    return _js_number(_js_round(n * 100) / 100)


def _times(n: int) -> str:
    return "1 time" if n == 1 else f"{n} times"


def _amount(x: float) -> str:
    """A threshold for describe_rule: at least one decimal, at most two ("3.0", "2.5", "0.25")."""
    text = f"{x:.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _pct(fraction: float) -> str:
    """A share of the frame as a percentage to 0.1, as the TypeScript shows it (0.125 -> "12.5%")."""
    return f"{_js_number(_js_round(fraction * 1000) / 10)}%"


def _js_round(x: float) -> float:
    """JavaScript's Math.round: halves go up (Python's round() sends them to the even neighbour)."""
    if not math.isfinite(x):
        return x
    floor = math.floor(x)
    return floor + 1 if x - floor >= 0.5 else floor


def _js_number(x: float) -> str:
    """JavaScript's String(number): integers without ".0", other numbers in their shortest round-trip form."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    text = repr(float(x))
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    power = int(exponent)
    if -7 < power < 21:
        return format(Decimal(text), "f")
    return f"{mantissa}e{'+' if power > 0 else '-'}{abs(power)}"
