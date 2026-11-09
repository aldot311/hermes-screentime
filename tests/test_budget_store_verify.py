import sqlite3
import threading

import pytest

from screentime_core.budget import Budget, measure_estimate, sam_estimate
from screentime_core.errors import ScreentimeError
from screentime_core.models import Box, Detection, MeasureReport
from screentime_core.store import RunStore, new_run_id
from screentime_core.verify import (
    VERIFY_SCHEMA,
    HermesLlmVerifier,
    ReplayVerifier,
    parse_verdict,
    pick_frames,
    to_checks,
    verify_prompt,
)


def det(frame, area, oid=0):
    return Detection(frame=frame, object_id=oid, box=Box(0, 0, 10, 10), area_fraction=area, centroid_x=0.5, centroid_y=0.5)


def test_pick_frames_first_middle_largest_distinct():
    ds = [det(f, a) for f, a in [(9, 0.1), (3, 0.2), (5, 0.9), (7, 0.1), (1, 0.05)]]
    picks = pick_frames(ds, frame_count=100)
    assert [d.frame for d in picks] == [1, 5]  # first=1, middle(sorted idx 2)=5, largest=5 -> deduped
    assert pick_frames([], 10) == []
    assert pick_frames([det(50, 0.5)], frame_count=10) == []


def test_parse_verdict_malformed_is_unclear_and_checks_cleaned():
    assert parse_verdict(2, {"verdict": "yes"}, "m").verdict == "unclear"
    tv = parse_verdict(1, {"verdict": "lookalike", "visibility": "weird", "reason": "  Poppi  can ",
                           "checks": [{"feature": "pink can", "seen": "no"}, {"feature": 3, "seen": "yes"}]}, "m")
    assert (tv.verdict, tv.visibility, tv.reason, tv.checks) == ("lookalike", "clear", "Poppi can",
                                                                  [{"feature": "pink can", "seen": "no"}])
    assert len(to_checks([{"feature": "x", "seen": "yes"}] * 9)) == 6


def test_verify_prompt_matches_app_wording():
    p = verify_prompt("LaCroix Strawberry Peach can", "drink can", with_photo=True)
    assert p.startswith("The first image is the brand's reference photo of its product.")
    assert 'searched for "drink can"' in p and "and the photo" in p
    assert VERIFY_SCHEMA["properties"]["verdict"]["enum"] == ["match", "lookalike", "unclear"]


def test_hermes_verifier_wraps_failures_and_labels_model():
    def ok(instr, images, schema, name):
        assert images and images[0][1] == "image/jpeg" and name == "product_check"
        return {"parsed": {"verdict": "match", "visibility": "clear", "reason": "pink can", "checks": []},
                "provider": "meta-ai", "model": "muse-spark-1.3", "usd": 0.001}

    tv, call = HermesLlmVerifier(ok).check(object_id=0, product="p", sam_prompt="can", photo_jpeg=b"x", crops=[b"y"])
    assert tv.verdict == "match" and tv.checked_by == "hermes:meta-ai/muse-spark-1.3" and call.usd == 0.001

    def boom(*a):
        raise RuntimeError("model does not support images")

    with pytest.raises(ScreentimeError) as e:
        HermesLlmVerifier(boom).check(object_id=0, product="p", sam_prompt="can", photo_jpeg=None, crops=[b"y"])
    assert e.value.code == "verifier_unavailable"


def test_replay_verifier_marks_replayed():
    tv, call = ReplayVerifier({1: {"verdict": "match", "visibility": "clear", "reason": "r", "checks": []}}).check(
        object_id=1, product="p", sam_prompt="can", photo_jpeg=None, crops=[])
    assert tv.verdict == "match" and call.replayed and call.usd == 0
    tv2, _ = ReplayVerifier({}).check(object_id=7, product="p", sam_prompt="can", photo_jpeg=None, crops=[])
    assert tv2.verdict == "unclear"


def test_budget_refuses_before_reserving_negative_control():
    conn = sqlite3.connect(":memory:")
    b = Budget(conn, per_video_usd=0.05, per_day_usd=5)
    with pytest.raises(ScreentimeError) as e:
        b.reserve("r1", 0.06, "SAM")
    assert e.value.code == "budget"
    assert b.spent_run("r1") == 0  # nothing was reserved
    r = b.reserve("r1", 0.04, "SAM")
    with pytest.raises(ScreentimeError):
        b.reserve("r1", 0.02, "Spark")
    b.settle(r, 0.01)
    assert b.spent_run("r1") == pytest.approx(0.01)
    b.reserve("r1", 0.02, "Spark")  # fits after settling


def test_budget_day_cap_across_runs_and_threads(tmp_path):
    path = tmp_path / "b.db"
    results = []

    def worker(i):
        conn = sqlite3.connect(path, timeout=10)
        b = Budget(conn, per_video_usd=1, per_day_usd=0.10)
        try:
            b.reserve(f"run{i}", 0.03, "x")
            results.append(True)
        except ScreentimeError:
            results.append(False)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 3  # 3 x 0.03 = 0.09 <= 0.10; the 4th would pass the cap


def test_estimates():
    assert sam_estimate(10) == pytest.approx(0.01)  # 50-frame minimum
    assert sam_estimate(900) == pytest.approx(0.18)
    assert measure_estimate(900, objects_to_check=2) == pytest.approx(0.18 + 0.04 + 0.01)


def test_store_owner_isolation():
    conn = sqlite3.connect(":memory:")
    s = RunStore(conn)
    rid = new_run_id()
    rep = {"run_id": rid, "mode": "measure", "video_sha256": "abc", "product": "can"}
    s.save(rep, owner="chatA", summary="7.9 s")
    assert s.get(rid, owner="chatA")["product"] == "can"
    assert s.get(rid, owner="chatB") is None
    assert s.latest_measure_for("abc", owner="chatA")["run_id"] == rid
    assert s.recent(owner="chatB") == []
    assert MeasureReport  # imported for type availability
