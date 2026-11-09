"""Hand-run: Teardown two local clips, compare them and draw the cards. No model, no key, ffmpeg only.

    .venv/bin/python dev/teardown_try.py YOURS.mp4 REFERENCE.mp4 --out /private/tmp/claude-501/teardown-try

Writes compare.png, teardown-yours.png, teardown-reference.png and compare.json to --out (analysis files under
--out/work) and prints the compare table. Outputs can hold private footage and local paths: never commit them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screentime_core import card, teardown  # noqa: E402
from screentime_core.errors import ScreentimeError  # noqa: E402
from screentime_core.models import to_dict  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yours", help="the draft")
    ap.add_argument("reference", help="the reference take")
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--yours-label", default="Yours")
    ap.add_argument("--reference-label", default="Reference")
    ap.add_argument("--threshold", type=float, default=10.0, help="scdet threshold (default 10)")
    ap.add_argument("--min-shot", type=float, default=0.2, help="merge cuts closer than this, seconds (default 0.2)")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    opts = {"scdet_threshold": args.threshold, "min_shot_sec": args.min_shot}
    try:
        t0 = time.monotonic()
        yours = teardown.analyze(args.yours, work, **opts)
        t1 = time.monotonic()
        theirs = teardown.analyze(args.reference, work, **opts)
        t2 = time.monotonic()
        report = teardown.compare(yours, theirs)
        card.render_compare_card(report, out / "compare.png", yours_label=args.yours_label,
                                 theirs_label=args.reference_label, workdir=work)
        card.render_teardown_card(yours, out / "teardown-yours.png", label=args.yours_label, workdir=work)
        card.render_teardown_card(theirs, out / "teardown-reference.png", label=args.reference_label, workdir=work)
        t3 = time.monotonic()
    except ScreentimeError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1
    (out / "compare.json").write_text(json.dumps(to_dict(report), indent=2), encoding="utf-8")

    print(f"analyze: {t1 - t0:.2f} s + {t2 - t1:.2f} s; cards: {t3 - t2:.2f} s")
    width = max(len(r.metric) for r in report.rows)
    print(f"{'Metric':<{width}} | {args.yours_label} | {args.reference_label} | Difference")
    for r in report.rows:
        y = teardown.format_value(r.metric, r.yours)
        t = teardown.format_value(r.metric, r.theirs)
        print(f"{r.metric:<{width}} | {y} | {t} | {r.diff}")
    for side, rep in (("yours", yours), ("reference", theirs)):
        print(f"{side}: run {rep.run_id}, cuts {rep.cuts_sec}, LRA {rep.loudness_range_lu}, "
              f"silences {[(s.start_sec, s.end_sec) for s in rep.silences]}, notes {rep.notes}")
    print(f"compare run {report.run_id}")
    print(f"cards: {report.card_path}, {yours.card_path}, {theirs.card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
