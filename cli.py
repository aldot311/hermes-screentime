"""`hermes screentime ...`: doctor, demo, selftest, measure, compare from a terminal."""

from __future__ import annotations

import json
from typing import Any


def setup(subparser: Any) -> None:
    subs = subparser.add_subparsers(dest="screentime_command")
    subs.add_parser("doctor", help="Check ffmpeg, the Meta key and settings")
    subs.add_parser("selftest", help="Cut detector on a generated clip + the bundled demo pipeline (no key, no network)")
    d = subs.add_parser("demo", help="Measure on the bundled clip (recorded API run, no key, no cost)")
    d.add_argument("--json", action="store_true")
    m = subs.add_parser("measure", help="Measure a product's seconds on screen (needs the Meta key)")
    m.add_argument("video")
    m.add_argument("--product", required=True)
    m.add_argument("--noun", required=True, help="plain object noun, e.g. 'drink can'")
    m.add_argument("--photo")
    m.add_argument("--brand", action="append", default=[])
    m.add_argument("--term", action="append", default=[], help='JSON, e.g. {"text":"3 s total","kind":"min_seconds_total","seconds":3}')
    m.add_argument("--json", action="store_true")
    c = subs.add_parser("compare", help="Teardown one video, or compare it with a reference (no key)")
    c.add_argument("video")
    c.add_argument("--reference")
    subparser.set_defaults(func=handle)


def handle(args: Any) -> None:
    from . import commands, tools

    sub = getattr(args, "screentime_command", None)
    if sub == "doctor":
        print(commands._status())
    elif sub == "selftest":
        print(commands._selftest())
    elif sub == "demo":
        from .screentime_core.format import measure_summary
        from .screentime_core.models import to_dict

        r = commands.run_demo()
        print(json.dumps(to_dict(r), indent=2, default=str) if args.json else measure_summary(r))
        if r.proof_path and not args.json:
            print(f"Proof reel: {r.proof_path}")
    elif sub == "measure":
        out = tools.measure({
            "video": args.video, "product": args.product, "object_noun": args.noun, "photo": args.photo,
            "brand_names": args.brand, "terms": [json.loads(t) for t in args.term],
        })
        data = json.loads(out)
        print(json.dumps(data, indent=2) if args.json or not data.get("ok") else data["summary"])
        if data.get("media_tag") and not args.json:
            print("Proof reel: " + data["media_tag"].removeprefix("MEDIA:"))
    elif sub == "compare":
        out = json.loads(tools.compare({"video": args.video, "reference": args.reference}))
        print(out.get("summary") if out.get("ok") else json.dumps(out, indent=2))
        if out.get("media_tag"):
            print("Card: " + out["media_tag"].removeprefix("MEDIA:"))
    else:
        print("Usage: hermes screentime <doctor|selftest|demo|measure|compare>")
