"""screentime_core.rules against Screen-Time's rules.ts.

tests/parity/rules.json and real_reports.json hold metrics, frames, disclosures, rules and the TypeScript's results
(dev/ts-parity/dump.mts). Every result is compared (pass, evidence to 1e-9, detail exactly) except the deliberate
differences, which have their own tests below: min_continuous on the strict run (compared with the TypeScript's own
check run on that strict run, "tsStrict"), and "disclosed" when nothing was said.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import pytest

from screentime_core.metrics import FrameSample, compute, frame_samples
from screentime_core.models import Box, ContractTerm, Detection, Disclosure, Mention, Segment, VideoInfo, VideoMetrics
from screentime_core.rules import (
    EPS,
    RULE_JSON_SCHEMA,
    RULE_KINDS,
    clip_segments,
    clock,
    describe_rule,
    evaluate,
    validate_rule,
)

PARITY = Path(__file__).parent / "parity"
RULES = json.loads((PARITY / "rules.json").read_text(encoding="utf-8"))
REAL = json.loads((PARITY / "real_reports.json").read_text(encoding="utf-8"))
CASES = RULES["cases"] + [r["rules"] for r in REAL["reports"]]
TOL = 1e-9
SNAKE = {"withinFirstSec": "within_first_sec", "forSeconds": "for_seconds", "minTimes": "min_times"}
INFO = VideoInfo(width=1080, height=1920, frame_count=0, duration_sec=0.0)


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOL)


def snake(rule: Optional[dict]) -> Optional[dict]:
    return None if rule is None else {SNAKE.get(k, k): v for k, v in rule.items()}


def mention(js: dict) -> Mention:
    return Mention(js["startSec"], js["endSec"], js["text"])


def metrics_of(case: dict) -> VideoMetrics:
    """The TypeScript's metrics as Python VideoMetrics. A missing spokenChecked meant "transcribed" to the TypeScript."""
    js, strict = case["metrics"], case["strict"]
    return VideoMetrics(
        seconds_on_screen=js["secondsOnScreen"],
        share_of_video=js.get("shareOfVideo", 0.0),
        first_seen_sec=js["firstSeenSec"],
        longest_run_sec=strict["longestRunSec"],
        longest_run_start_sec=strict["longestRunStartSec"],
        avg_area_fraction=js.get("avgAreaFraction", 0.0),
        peak_area_fraction=0.0,
        peak_area_sec=None,
        segments=[Segment(s["startSec"], s["endSec"]) for s in js["segments"]],
        visible_frames=list(js.get("visibleFrames") or []),
        mentions=[mention(x) for x in js["mentions"]],
        spoken_checked=js.get("spokenChecked") is not False,
    )


def samples_of(case: dict) -> Optional[list[FrameSample]]:
    return None if case["frames"] is None else [FrameSample(f, a) for f, a in case["frames"]]


def disclosure_of(case: dict) -> Optional[Disclosure]:
    d = case["disclosure"]
    if d is None:
        return None
    return Disclosure(spoken=[mention(x) for x in d["spoken"]], spoken_checked=d.get("spokenChecked") is not False)


def check(case: dict, rule: Optional[dict], info: VideoInfo = INFO) -> Any:
    term = ContractTerm("contract wording", snake(rule))
    return evaluate([term], metrics_of(case), info, disclosure_of(case), samples=samples_of(case))[0]


def differences(got, want: dict) -> list[str]:
    out = []
    if got.passed != want["pass"]:
        out.append(f"pass {got.passed!r} != {want['pass']!r}")
    if got.detail != want["detail"]:
        out.append(f"detail {got.detail!r} != {want['detail']!r}")
    ev = want["evidence"]
    if len(got.evidence) != len(ev) or not all(
        close(g.start_sec, w["startSec"]) and close(g.end_sec, w["endSec"]) for g, w in zip(got.evidence, ev)
    ):
        out.append(f"evidence {[(s.start_sec, s.end_sec) for s in got.evidence]} != {[(s['startSec'], s['endSec']) for s in ev]}")
    return out


def spoke(case: dict) -> bool:
    return bool(case["disclosure"] and case["disclosure"]["spoken"])


def spoke_only(case: dict) -> bool:
    """Said aloud and nothing read on screen: the one disclosure input where v0.1 and the TypeScript see the same."""
    return spoke(case) and not case["disclosure"]["onScreen"]


# ----------------------------------------------------------------------------------------------------------------------
# Parity


def test_parity_files_cover_every_kind_at_and_around_its_threshold():
    kinds = {ch["rule"]["kind"] for c in CASES for ch in c["checks"] if ch["rule"]}
    assert kinds == set(RULE_KINDS)
    outcomes = {(ch["rule"]["kind"], ch["ts"]["pass"]) for c in CASES for ch in c["checks"] if ch["rule"]}
    for kind in RULE_KINDS:
        if kind != "disclosed":
            assert (kind, True) in outcomes and (kind, False) in outcomes, kind
    assert sum(len(c["checks"]) for c in CASES) > 2000


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-{c['name'][:48]}" for i, c in enumerate(CASES)])
def test_evaluate_matches_typescript(case):
    failures = []
    for n, ch in enumerate(case["checks"]):
        rule = ch["rule"]
        kind = rule["kind"] if rule else None
        if kind == "disclosed" and not spoke_only(case):
            continue  # deliberate difference (v0.1 reads no on-screen text), tested below
        want = ch["tsStrict"] if kind == "min_continuous" else ch["ts"]
        diff = differences(check(case, rule), want)
        if diff:
            failures.append(f"check {n} {rule}: {'; '.join(diff)}")
    assert not failures, "\n".join(failures[:10])


@pytest.mark.parametrize("sec,want", RULES["clock"])
def test_clock_matches_typescript(sec, want):
    assert clock(sec) == want


@pytest.mark.parametrize("case", RULES["clipSegments"])
def test_clip_segments_matches_typescript(case):
    got = clip_segments([Segment(s["startSec"], s["endSec"]) for s in case["segments"]], case["limit"])
    assert [(s.start_sec, s.end_sec) for s in got] == [(s["startSec"], s["endSec"]) for s in case["ts"]]


# ----------------------------------------------------------------------------------------------------------------------
# Deliberate differences


def test_min_continuous_follows_the_strict_run_where_the_typescript_bridged():
    flipped = 0
    for case in CASES:
        for ch in case["checks"]:
            if ch["rule"] and ch["rule"]["kind"] == "min_continuous" and ch["ts"]["pass"] != ch["tsStrict"]["pass"]:
                assert check(case, ch["rule"]).passed is ch["tsStrict"]["pass"] is False
                flipped += 1
    assert flipped >= 5, "the parity set should hold terms the bridged run met and the strict run does not"


def test_min_continuous_example():
    # Seen 0-7 and 10-17: the TypeScript's bridged run was 18 frames (1.2 s) and met "1 s continuous".
    frames = [*range(0, 8), *range(10, 18)]
    dets = [Detection(f, 1, Box(0, 0, 1, 1), 0.1, 0.5, 0.5) for f in frames]
    info = VideoInfo(1080, 1920, 30, 2.0)
    m = compute(dets, {1}, info)
    [r] = evaluate([ContractTerm("1 s in one shot", {"kind": "min_continuous", "seconds": 1})], m, info)
    assert r.passed is False
    assert r.evidence == [Segment(0.0, 8 / 15)]
    assert r.detail == "Longest continuous run 0.5 s (needs 1 s)."


def test_disclosed_needs_a_person_when_nothing_was_said():
    seen = 0
    for case in CASES:
        for ch in case["checks"]:
            if ch["rule"] and ch["rule"]["kind"] == "disclosed" and not spoke(case):
                r = check(case, ch["rule"])
                assert r.passed is None and r.evidence == []
                assert "not checked" in r.detail.lower() or "not read in this version" in r.detail
                seen += 1
    assert seen >= 10


def test_disclosed_said_and_shown_passes_on_the_spoken_part_alone():
    # The TypeScript also listed the on-screen moments; v0.1 never reads them, so its evidence is the spoken part.
    seen = 0
    for case in CASES:
        for ch in case["checks"]:
            if ch["rule"] and ch["rule"]["kind"] == "disclosed" and spoke(case) and not spoke_only(case):
                r = check(case, ch["rule"])
                assert r.passed is True and ch["ts"]["pass"] is True
                spoken = sorted((x["startSec"], x["endSec"]) for x in case["disclosure"]["spoken"])
                assert [(s.start_sec, s.end_sec) for s in r.evidence] == spoken
                assert ch["ts"]["detail"].startswith(r.detail + "; on screen")
                seen += 1
    assert seen >= 1


def test_disclosed_cases_one_by_one():
    rule = ContractTerm("Say it is an ad", {"kind": "disclosed"})
    m = compute([], {1}, INFO, spoken_checked=True)
    said = Disclosure(spoken=[Mention(9.0, 9.5, "this is an ad"), Mention(3.0, 3.2, "#ad")], spoken_checked=True)
    r = evaluate([rule], m, INFO, said)[0]
    assert r.passed is True
    assert r.evidence == [Segment(3.0, 3.2), Segment(9.0, 9.5)]
    assert r.detail == "Disclosed: said “this is an ad” at 0:09.0"  # the first as given, as the TypeScript

    heard_nothing = evaluate([rule], m, INFO, Disclosure(spoken=[], spoken_checked=True))[0]
    assert heard_nothing.passed is None
    assert "on-screen text such as #ad is not read in this version" in heard_nothing.detail

    failed = evaluate([rule], m, INFO, Disclosure(spoken=[], spoken_checked=False))[0]
    assert failed.passed is None and "could not be transcribed" in failed.detail

    no_sound = VideoInfo(1080, 1920, 0, 0.0, has_audio=False)
    silent = evaluate([rule], m, no_sound, Disclosure(spoken=[], spoken_checked=None))[0]
    assert silent.passed is None and "no sound" in silent.detail

    missing = evaluate([rule], m, INFO, None)[0]
    assert missing.passed is None and missing.detail.startswith("Not checked")


def test_mentioned_is_unknown_without_a_transcript():
    rule = ContractTerm("Say LaCroix", {"kind": "mentioned", "min_times": 1})
    for flag, info, want in [
        (False, INFO, None),  # transcription failed (as the TypeScript)
        (None, INFO, None),  # never transcribed although there is sound (the TypeScript would have failed it)
        (None, VideoInfo(1, 1, 0, 0.0, has_audio=False), False),  # no sound: nothing was said
        (True, INFO, False),  # heard, never said
    ]:
        m = compute([], {1}, info, spoken_checked=flag)
        assert evaluate([rule], m, info)[0].passed is want, (flag, info.has_audio)


def test_empty_video_without_samples_is_known_empty():
    # The TypeScript, given no frames, said "at most 0 s" and left min_area unchecked; nothing on screen is known data.
    m = compute([], {1}, INFO)
    within, area = evaluate(
        [
            ContractTerm("2 s in the first 5 s", {"kind": "min_seconds_within", "seconds": 2, "within_first_sec": 5}),
            ContractTerm("10% of the frame for 1 s", {"kind": "min_area", "fraction": 0.1, "for_seconds": 1}),
        ],
        m,
        INFO,
    )
    assert (within.passed, within.detail) == (False, "On screen 0 s in the first 5 s (needs 2 s).")
    assert (area.passed, area.detail) == (False, "At least 10% of the frame for 0 s (needs 1 s).")


# ----------------------------------------------------------------------------------------------------------------------
# Per-frame data


def test_min_area_needs_samples_and_uses_them():
    dets = [Detection(f, 1, Box(0, 0, 1, 1), 0.3 if f < 15 else 0.05, 0.5, 0.5) for f in range(30)]
    info = VideoInfo(1080, 1920, 30, 2.0)
    m = compute(dets, {1}, info)
    term = ContractTerm("a quarter of the frame for 1 s", {"kind": "min_area", "fraction": 0.25, "for_seconds": 1})
    with_samples = evaluate([term], m, info, samples=frame_samples(dets, {1}, info))[0]
    assert with_samples.passed is True and with_samples.evidence == [Segment(0.0, 1.0)]
    assert with_samples.detail == "At least 25% of the frame for 1 s (needs 1 s)."
    if not hasattr(m, "visible_areas"):
        without = evaluate([term], m, info)[0]
        assert without.passed is None and without.detail == "Needs per-frame size data, which was not provided."


def test_min_area_works_without_samples_once_metrics_carry_visible_areas(monkeypatch):
    # Proposed models field (VideoMetrics.visible_areas): simulated here with a subclass; compute() fills it and
    # evaluate() then needs no samples=, so a stored report can be checked again for min_area.
    from screentime_core import metrics as metrics_mod

    @dataclass
    class WithAreas(VideoMetrics):
        visible_areas: list[float] = field(default_factory=list)

    monkeypatch.setattr(metrics_mod, "VideoMetrics", WithAreas)
    monkeypatch.setattr(metrics_mod, "_HAS_VISIBLE_AREAS", True)
    dets = [Detection(f, 1, Box(0, 0, 1, 1), 0.3 if f < 15 else 0.05, 0.5, 0.5) for f in range(30)]
    info = VideoInfo(1080, 1920, 30, 2.0)
    m = compute(dets, {1}, info)
    assert m.visible_areas == [0.3] * 15 + [0.05] * 15
    term = ContractTerm("a quarter of the frame for 1 s", {"kind": "min_area", "fraction": 0.25, "for_seconds": 1})
    [r] = evaluate([term], m, info)
    assert (r.passed, r.detail) == (True, "At least 25% of the frame for 1 s (needs 1 s).")


def test_min_seconds_within_uses_visible_frames_without_samples():
    dets = [Detection(f, 1, Box(0, 0, 1, 1), 0.1, 0.5, 0.5) for f in [0, 3, 30, 31, 32, 33]]
    info = VideoInfo(1080, 1920, 60, 4.0)
    m = compute(dets, {1}, info)
    term = ContractTerm("0.2 s in the first second", {"kind": "min_seconds_within", "seconds": 0.2, "within_first_sec": 1})
    r = evaluate([term], m, info)[0]
    assert r.passed is False and r.detail == "On screen 0.1 s in the first 1 s (needs 0.2 s)."


def test_terms_without_a_rule_or_with_a_bad_one_need_a_person():
    m = compute([], {1}, INFO)
    none, bad = evaluate([ContractTerm("good lighting", None), ContractTerm("x", {"kind": "min_area", "fraction": 2})], m, INFO)
    assert (none.passed, none.detail) == (None, "Could not be turned into a checkable rule.")
    assert bad.passed is None and bad.detail.startswith("Could not be checked: ")


def test_eps_tolerates_float_sums_just_under_a_threshold():
    m = compute([Detection(f, 1, Box(0, 0, 1, 1), 0.1, 0.5, 0.5) for f in range(90)], {1}, VideoInfo(1, 1, 90, 6.0))
    summed = 0.0
    for _ in range(90):
        summed += 1 / 15
    m = replace(m, seconds_on_screen=summed)  # 5.999999999999998: 90 frames added up one at a time
    assert m.seconds_on_screen < 6
    total = ContractTerm("6 s", {"kind": "min_seconds_total", "seconds": 6})
    assert evaluate([total], m, INFO)[0].passed is True
    assert evaluate([total], m, INFO)[0].detail == "On screen 6 s in total (needs 6 s)."
    above = ContractTerm("6 s and a bit", {"kind": "min_seconds_total", "seconds": 6 + 2 * EPS})
    assert evaluate([above], m, INFO)[0].passed is False


# ----------------------------------------------------------------------------------------------------------------------
# Rule validation, schema and description


GOOD_RULES = [
    {"kind": "min_seconds_total", "seconds": 3},
    {"kind": "min_seconds_within", "seconds": 2, "within_first_sec": 5},
    {"kind": "min_continuous", "seconds": 1.5},
    {"kind": "min_area", "fraction": 0.1, "for_seconds": 2},
    {"kind": "mentioned", "min_times": 1},
    {"kind": "mentioned", "min_times": 2, "within_first_sec": 4},
    {"kind": "mentioned", "min_times": 1, "within_first_sec": 0},  # 0: anywhere, as the TypeScript's parser read it
    {"kind": "first_seen_before", "seconds": 2},
    {"kind": "disclosed"},
    {"kind": "min_seconds_total", "seconds": 3, "within_first_sec": 0, "fraction": None},  # unused fields left empty
    {"kind": "mentioned", "min_times": 2.0},
]


@pytest.mark.parametrize("rule", GOOD_RULES)
def test_validate_rule_accepts(rule):
    assert validate_rule(rule) == (True, None)


@pytest.mark.parametrize(
    "rule,why",
    [
        ({"kind": "loudness", "seconds": 3}, "unknown kind"),
        ({"seconds": 3}, "unknown kind"),
        ("min_seconds_total", "must be an object"),
        (None, "must be an object"),
        ({"kind": "min_seconds_total"}, "needs seconds"),
        ({"kind": "min_seconds_within", "seconds": 2}, "needs within_first_sec"),
        ({"kind": "min_area", "fraction": 0.1}, "needs for_seconds"),
        ({"kind": "mentioned"}, "needs min_times"),
        ({"kind": "min_seconds_total", "seconds": -3}, "must not be negative"),
        ({"kind": "mentioned", "min_times": 1, "within_first_sec": -2}, "must not be negative"),
        ({"kind": "min_seconds_total", "seconds": 0}, "greater than 0"),
        ({"kind": "mentioned", "min_times": 0}, "greater than 0"),
        ({"kind": "min_area", "fraction": 10, "for_seconds": 1}, "from 0 to 1"),
        ({"kind": "mentioned", "min_times": 1.5}, "whole number"),
        ({"kind": "min_seconds_total", "seconds": "3"}, "must be a number"),
        ({"kind": "min_seconds_total", "seconds": True}, "must be a number"),
        ({"kind": "min_seconds_total", "seconds": float("nan")}, "finite"),
        ({"kind": "min_seconds_total", "seconds": float("inf")}, "finite"),
        ({"kind": "first_seen_before", "seconds": 3, "within_first_sec": 3}, "does not use within_first_sec"),
        ({"kind": "disclosed", "seconds": 3}, "does not use seconds"),
        ({"kind": "mentioned", "min_times": 1, "seconds": 5}, "does not use seconds"),
        ({"kind": "min_seconds_within", "seconds": 2, "withinFirstSec": 5}, "within_first_sec instead"),
        ({"kind": "min_seconds_total", "seconds": 3, "colour": "red"}, "unknown field"),
    ],
)
def test_validate_rule_rejects(rule, why):
    ok, error = validate_rule(rule)
    assert ok is False and why in error


def test_schema_lists_every_kind_and_field():
    json.dumps(RULE_JSON_SCHEMA)  # plain JSON
    assert RULE_JSON_SCHEMA["properties"]["kind"]["enum"] == list(RULE_KINDS)
    assert RULE_JSON_SCHEMA["required"] == ["kind"] and RULE_JSON_SCHEMA["additionalProperties"] is False
    used = {name for spec in RULE_KINDS.values() for name in (*spec["fields"], *spec["optional"])}
    assert used == set(RULE_JSON_SCHEMA["properties"]) - {"kind"}
    for kind in RULE_KINDS:
        assert kind in RULE_JSON_SCHEMA["description"]
    for rule in GOOD_RULES:  # everything the validator accepts fits the schema's fields
        assert set(rule) <= set(RULE_JSON_SCHEMA["properties"])


@pytest.mark.parametrize(
    "rule,want",
    [
        ({"kind": "min_seconds_total", "seconds": 3}, "at least 3.0 s on screen in total"),
        ({"kind": "min_seconds_within", "seconds": 2.5, "within_first_sec": 8}, "at least 2.5 s on screen within the first 8.0 s"),
        ({"kind": "min_continuous", "seconds": 0.25}, "at least 0.25 s on screen in one continuous run"),
        ({"kind": "min_area", "fraction": 0.125, "for_seconds": 2}, "at least 12.5% of the frame for at least 2.0 s in total"),
        ({"kind": "mentioned", "min_times": 1}, "brand name said aloud at least 1 time"),
        ({"kind": "mentioned", "min_times": 3, "within_first_sec": 5}, "brand name said aloud at least 3 times within the first 5.0 s"),
        ({"kind": "first_seen_before", "seconds": 2}, "first on screen before 2.0 s"),
        (None, "not checkable as written"),
    ],
)
def test_describe_rule(rule, want):
    assert describe_rule(rule) == want


def test_describe_rule_for_disclosed_and_bad_rules():
    assert "on-screen text is not read" in describe_rule({"kind": "disclosed"})
    assert describe_rule({"kind": "min_seconds_total", "seconds": -1}).startswith("not checkable as written (")
