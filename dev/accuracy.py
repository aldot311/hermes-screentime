"""Score the plugin's own Measure runs against the human-checked frame labels (hand-run, live, about $1 for 11 Reels).

    ~/ai/bin/with-secrets.sh .venv/bin/python dev/accuracy.py --videos <dir> --labels <dir> --out docs/accuracy.json

Per frame at 15 fps, as the Screen-Time app's scripts/accuracy.mts does: a labelled frame is product on screen; the
prediction is (a) every frame SAM 3.1 drew a box in ("SAM alone"), (b) the frames the plugin counted after the product
check ("with the check"). Pooled precision, recall and F1 over all Reels, seconds per Reel, and the seconds counted on
the look-alike-only Reel. Clips are reported by number, not by the person in them.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screentime_core.budget import Budget  # noqa: E402
from screentime_core.meta_client import MetaClient  # noqa: E402
from screentime_core.models import (  # noqa: E402
    FPS,
    ContractTerm,  # noqa: E402
    to_dict,
)
from screentime_core.pipeline import Engines, MeasureRequest, run_measure  # noqa: E402
from screentime_core.rules import validate_rule  # noqa: E402
from screentime_core.verify import MetaSparkVerifier  # noqa: E402

PRODUCT = (
    "a pink LaCroix Strawberry Peach sparkling water can: blue 'LaCroix' script over a yellow peach, red and yellow "
    "hearts, strawberries and green stripes near the base"
)
TERMS = [
    ("On screen at least 3 seconds in total", {"kind": "min_seconds_total", "seconds": 3}),
    ("Says LaCroix out loud", {"kind": "mentioned", "min_times": 1}),
    ("Discloses the ad: says it is an ad or sponsored", {"kind": "disclosed"}),
]


def label_frames(target: dict) -> set[int]:
    out: set[int] = set()
    for a, b in target["ranges"]:
        out.update(range(int(a), int(b) + 1))
    return out


def counts(pred: set[int], truth: set[int]) -> dict:
    return {"tp": len(pred & truth), "fp": len(pred - truth), "fn": len(truth - pred)}


def prf(c: dict) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"precision": p, "recall": r, "f1": f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--photo", default=str(Path(__file__).resolve().parents[1] / "fixtures" / "demo" / "product.jpg"))
    ap.add_argument("--out", default="docs/accuracy.json")
    ap.add_argument("--max-usd", type=float, default=2.5)
    args = ap.parse_args()
    key = os.environ.get("META_MODEL_API_KEY", "")
    if not key:
        sys.exit("META_MODEL_API_KEY missing (run through ~/ai/bin/with-secrets.sh)")
    client = MetaClient(key)
    labels = sorted(p for p in Path(args.labels).glob("*.json") if p.name not in ("objects.json", "checklist.json"))
    db = sqlite3.connect(Path(tempfile.gettempdir()) / "screentime-accuracy-budget.db")
    budget = Budget(db, per_video_usd=0.6, per_day_usd=args.max_usd)
    rows, totals = [], {"sam": {"tp": 0, "fp": 0, "fn": 0}, "checked": {"tp": 0, "fp": 0, "fn": 0}}
    usd_total = 0.0
    for n, lp in enumerate(labels, 1):
        lab = json.loads(lp.read_text())
        clip = lab["id"].split("/")[-1]
        video = Path(args.videos) / f"{clip}.mp4"
        target = next(t for t in lab["targets"] if t["name"] == "lacroix-can")
        truth = label_frames(target)
        terms = [ContractTerm(t, r if validate_rule(r)[0] else None) for t, r in TERMS]
        req = MeasureRequest(str(video), PRODUCT, "drink can", photo_path=args.photo, brand_names=["LaCroix", "La Croix"],
                             terms=terms)
        eng = Engines("meta", MetaSparkVerifier(client), client, transcribe=True)
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            rep = run_measure(req, eng, workdir=Path(tmp) / "w", outdir=Path(tmp) / "o", budget=budget,
                              run_id=f"acc{n:02d}", render_proof=False)
        secs = time.monotonic() - t0
        sam_pred = {d.frame for d in rep.detections}
        checked_pred = set(rep.metrics.visible_frames)
        c_sam, c_chk = counts(sam_pred, truth), counts(checked_pred, truth)
        for k in ("tp", "fp", "fn"):
            totals["sam"][k] += c_sam[k]
            totals["checked"][k] += c_chk[k]
        usd = rep.receipt.usd if rep.receipt else 0.0
        usd_total += usd
        builder = clip.endswith("-aldo")
        row = {
            "reel": n, "builder_filmed": builder, "frames": rep.info.frame_count,
            "truth_sec": round(len(truth) / FPS, 2), "sam_alone_sec": round(len(sam_pred) / FPS, 2),
            "checked_sec": round(rep.metrics.seconds_on_screen, 2), "sam": c_sam, "checked": c_chk,
            "objects": [{"object": t.object_id + 1, "verdict": t.verdict} for t in rep.tracks],
            "terms": [{"term": t.term.text, "passed": t.passed} for t in rep.terms],
            "usd": usd, "seconds_to_run": round(secs, 1),
        }
        rows.append(row)
        print(f"reel {n:2d} {'(builder)' if builder else '         '} truth {row['truth_sec']:5.1f} s | SAM alone "
              f"{row['sam_alone_sec']:5.1f} s | with check {row['checked_sec']:5.1f} s | ${usd:.3f} | {secs:.0f} s",
              flush=True)
    # Decision check for the first term (on screen at least 3 s in total) against the labels: right, sent to a person,
    # or wrong. A Reel whose labelled seconds are 3 or more should pass; one under 3 should fail.
    decisions = {"right": 0, "to_a_person": 0, "wrong": 0}
    for row in rows:
        got = row["terms"][0]["passed"]
        want = row["truth_sec"] >= 3.0
        key = "to_a_person" if got is None else ("right" if got == want else "wrong")
        decisions[key] += 1
        row["three_second_term"] = {"expected": want, "got": got, "outcome": key}
    summary = {"sam_alone": prf(totals["sam"]), "with_check": prf(totals["checked"]), "frames": totals,
               "three_second_term": decisions, "usd_total": round(usd_total, 4), "reels": len(rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows, "product": PRODUCT,
                                          "photo": "fixtures/demo/product.jpg"}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    _ = to_dict


if __name__ == "__main__":
    main()
