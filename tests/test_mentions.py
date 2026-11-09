"""screentime_core.mentions against Screen-Time's mentions.ts.

tests/parity/mentions.json holds transcripts, brand names and the TypeScript's outputs (dev/ts-parity/dump.mts):
tokenize (code-point offsets), distinctNames, findMentions and findDisclosures. Times compare to 1e-9 (the
TypeScript works in milliseconds, the port in seconds), text exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from screentime_core.errors import ScreentimeError
from screentime_core.mentions import distinct_names, find_mentions, find_spoken_disclosures, tokenize
from screentime_core.models import Mention

PARITY = Path(__file__).parent / "parity"
MENTIONS = json.loads((PARITY / "mentions.json").read_text(encoding="utf-8"))
TOL = 1e-9


def turns_of(js: list[dict]) -> list[dict]:
    return [{"start_sec": t["startMs"] / 1000, "end_sec": t["endMs"] / 1000, "text": t["transcript"]} for t in js]


def same_mentions(got: list[Mention], want: list[dict]) -> bool:
    return len(got) == len(want) and all(
        g.text == w["text"]
        and g.approximate is True
        and math.isclose(g.start_sec, w["startSec"], rel_tol=0.0, abs_tol=TOL)
        and math.isclose(g.end_sec, w["endSec"], rel_tol=0.0, abs_tol=TOL)
        for g, w in zip(got, want)
    )


def ids(cases: list[dict], key: str) -> list[str]:
    return [f"{i}-{c[key][:40]!r}" for i, c in enumerate(cases)]


# ----------------------------------------------------------------------------------------------------------------------
# Parity


def test_parity_file_sizes():
    assert len(MENTIONS["tokenize"]) >= 40
    assert len(MENTIONS["findMentions"]) >= 60
    assert len(MENTIONS["findDisclosures"]) >= 150
    assert sum(bool(c["ts"]) for c in MENTIONS["findDisclosures"]) >= 40  # disclosures found, not only rejected


@pytest.mark.parametrize("case", MENTIONS["tokenize"], ids=ids(MENTIONS["tokenize"], "text"))
def test_tokenize_matches_typescript(case):
    assert [(t.norm, t.start, t.end) for t in tokenize(case["text"])] == [(t["norm"], t["start"], t["end"]) for t in case["ts"]]


@pytest.mark.parametrize("case", MENTIONS["distinctNames"])
def test_distinct_names_matches_typescript(case):
    assert distinct_names(case["names"]) == case["ts"]


@pytest.mark.parametrize("case", MENTIONS["findMentions"], ids=ids(MENTIONS["findMentions"], "name"))
def test_find_mentions_matches_typescript(case):
    got = find_mentions(turns_of(case["turns"]), case["terms"])
    assert same_mentions(got, case["ts"]), [(m.start_sec, m.end_sec, m.text) for m in got]


@pytest.mark.parametrize("case", MENTIONS["findDisclosures"], ids=ids(MENTIONS["findDisclosures"], "name"))
def test_find_spoken_disclosures_matches_typescript(case):
    got = find_spoken_disclosures(turns_of(case["turns"]))
    assert same_mentions(got, case["ts"]), [(m.start_sec, m.end_sec, m.text) for m in got]


# ----------------------------------------------------------------------------------------------------------------------
# Behaviour spelled out


def test_one_name_written_twice_is_one_mention():
    [m] = find_mentions([{"start_sec": 0, "end_sec": 9, "text": "than a La Croix LaCroix. This is an ad"}], ["LaCroix", "La Croix"])
    assert (m.start_sec, m.end_sec, m.text, m.approximate) == (2.0, 5.0, "La Croix LaCroix", True)
    assert [x.text for x in find_mentions([{"start_sec": 0, "end_sec": 1, "text": "Nike, Nike!"}], ["Nike"])] == ["Nike", "Nike"]


def test_denials_are_not_disclosures():
    turn = {"start_sec": 0, "end_sec": 4, "text": ""}
    for text in ["this isn't an ad", "This is not sponsored", "not a paid partnership", "I don't think this is an ad"]:
        assert find_spoken_disclosures([{**turn, "text": text}]) == [], text
    [d] = find_spoken_disclosures([{**turn, "text": "Ad. this is great"}])
    assert (d.start_sec, d.end_sec, d.text) == (0.0, 1.0, "Ad")


def test_turns_can_be_objects():
    turns = [SimpleNamespace(start_sec=2.0, end_sec=6.0, text="we wear nike daily")]
    assert find_mentions(turns, ["Nike"]) == [Mention(4.0, 5.0, "nike", True)]
    assert find_spoken_disclosures([SimpleNamespace(start_sec=0.0, end_sec=2.0, text="#ad")]) == [Mention(0.0, 2.0, "ad", True)]


@pytest.mark.parametrize("bad", [{"text": "Nike"}, {"start_sec": "0", "end_sec": 1, "text": "Nike"}, SimpleNamespace(text="Nike")])
def test_a_turn_without_times_is_bad_input(bad):
    with pytest.raises(ScreentimeError) as err:
        find_mentions([bad], ["Nike"])
    assert err.value.code == "bad_input"


def test_missing_text_and_non_string_aliases_are_skipped():
    assert find_mentions([{"start_sec": 0, "end_sec": 1, "text": None}], ["Nike"]) == []
    assert find_mentions([{"start_sec": 0, "end_sec": 1, "text": "Nike"}], [None, 3, "nike"]) == [Mention(0.0, 1.0, "Nike")]  # type: ignore[list-item]
    assert distinct_names(["LaCroix", None, "La Croix"]) == ["LaCroix"]  # type: ignore[list-item]
