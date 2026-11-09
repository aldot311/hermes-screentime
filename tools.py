"""Tool handlers. Each returns a JSON string and never raises: expected failures come back as {"ok": false, ...}."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from . import hermes_bridge as hb
from .screentime_core.errors import ScreentimeError
from .screentime_core.models import ContractTerm

log = logging.getLogger(__name__)

MEASURE_REPLY = (
    "Reply with the measured numbers exactly as given (do not round or estimate): lead with the seconds on screen and, "
    "if terms were checked, the verdict; list each term with PASS, FAIL or NEEDS A PERSON and its time; say which "
    "objects were rejected as look-alikes and why. Then put the media_tag value on its own line so the proof reel is "
    "attached (in a plain terminal, give the path instead)."
)
COMPARE_REPLY = (
    "Reply with the differences exactly as measured, in neutral words (later, earlier, more, fewer); never call one "
    "version better or predict performance. Then put the media_tag value on its own line so the card is attached (in "
    "a plain terminal, give the path instead)."
)


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _fail(exc: ScreentimeError) -> str:
    out = {"ok": False, **exc.to_dict()}
    if exc.code == "engine_unavailable":
        out["price"] = "About $0.003 per second of video for tracking, plus about $0.02 per checked object."
        out["try_now"] = "/screentime demo runs the whole Measure pipeline on a bundled clip with no key."
    return _dump(out)


def _internal(where: str, exc: Exception) -> str:
    log.exception("screentime %s failed", where)
    # Exception text from our own modules is key-free by construction; keep it short anyway.
    return _dump({"ok": False, "error": "internal", "message": f"{where} failed: {type(exc).__name__}: {str(exc)[:200]}"})


def terms_from_args(items: Optional[list]) -> list[ContractTerm]:
    """The model's typed rules, validated; anything malformed or 'uncheckable' is kept as a term for a person."""
    from .screentime_core import rules

    out: list[ContractTerm] = []
    for it in (items or [])[:12]:
        if not isinstance(it, dict):
            continue
        text = " ".join(str(it.get("text") or "").split())[:300]
        if not text:
            continue
        kind = it.get("kind")
        if not kind or kind == "uncheckable":
            out.append(ContractTerm(text, None))
            continue
        rule: dict[str, Any] = {"kind": kind}
        for k in ("seconds", "within_first_sec", "fraction", "for_seconds", "min_times"):
            if it.get(k) is not None:
                rule[k] = it[k]
        ok, _why = rules.validate_rule(rule)
        out.append(ContractTerm(text, rule if ok else None))
    return out


def _engines(require_key: bool = True):
    """Pick the engines from the settings. With a Meta key, auto means SAM 3.1 + Muse Spark + Voice Transcribe; without
    one, auto means the user's own Hermes models (vision for tracking and the check, Hermes speech-to-text)."""
    from .screentime_core.meta_client import MetaClient
    from .screentime_core.pipeline import Engines
    from .screentime_core.verify import HermesLlmVerifier, MetaSparkVerifier
    from .screentime_core.vision_track import VisionTracker

    key = hb.meta_key()
    client = MetaClient(key) if key else None
    no_key_fix = "Add the key in Settings > Plugins > screentime, or set this to auto or hermes."

    def choice(name: str) -> str:
        v = str(hb.setting(name, "auto")).lower()
        return v if v in ("auto", "meta", "hermes") else "auto"

    track, check, speech = choice("tracker"), choice("verifier"), choice("speech")
    for name, v in (("Tracker", track), ("Product check", check), ("Speech-to-text", speech)):
        if v == "meta" and client is None:
            raise ScreentimeError("engine_unavailable", f"{name} is set to Meta, but no Meta key is set.", fix=no_key_fix)
    use_meta = {k: v == "meta" or (v == "auto" and client is not None)
                for k, v in (("track", track), ("check", check), ("speech", speech))}

    verifier = MetaSparkVerifier(client) if use_meta["check"] else HermesLlmVerifier(hb.hermes_structured_call())
    listen = bool(hb.setting("listen", True))
    stt, stt_label = (None, "")
    if listen and not use_meta["speech"]:
        stt, stt_label = hb.hermes_stt()
    if use_meta["track"]:
        tracker, vision = "meta", None
    else:
        tracker = "vision"
        try:
            fps = int(hb.setting("frames_per_second", 3))
        except (TypeError, ValueError):
            fps = 3
        vision = VisionTracker(hb.hermes_structured_call(hb.TRACK_TASK, "screentime tracking", 1500), fps)
    if require_key and tracker == "meta" and client is None:
        raise ScreentimeError("engine_unavailable", "Measure needs a Meta Model API key for SAM 3.1.", fix=no_key_fix)
    off = "" if (not listen or use_meta["speech"] or stt) else stt_label
    return Engines(tracker, verifier, client, transcribe=listen and use_meta["speech"], vision=vision,
                   stt=stt, stt_label=stt_label if stt else "hermes-stt", speech_off_reason=off)


def _measure_payload(report, summary: str) -> dict:
    from .screentime_core import rules
    from .screentime_core.format import media_instructions

    m = report.metrics
    payload = {
        "ok": True,
        "run_id": report.run_id,
        "summary": summary,
        "seconds_on_screen": round(m.seconds_on_screen, 2),
        "duration_sec": round(report.info.duration_sec, 2),
        "share_of_video": round(m.share_of_video, 4),
        "first_seen_sec": m.first_seen_sec,
        "longest_continuous_sec": round(m.longest_run_sec, 2),
        "said_out_loud": [round(x.start_sec, 1) for x in m.mentions],
        "terms": [
            {
                "term": t.term.text,
                "read_as": rules.describe_rule(t.term.rule) if t.term.rule else "not checkable as written",
                "result": {True: "PASS", False: "FAIL", None: "NEEDS A PERSON"}[t.passed],
                "detail": t.detail,
                "moments": [[round(s.start_sec, 2), round(s.end_sec, 2)] for s in t.evidence[:5]],
            }
            for t in report.terms
        ],
        "objects": [
            {"object": t.object_id + 1, "verdict": t.verdict, "reason": t.reason, "checked_by": t.checked_by}
            for t in report.tracks
        ],
        "product_check": report.verifier,
        "product_check_validated": report.verifier_validated,
        "engine": report.engine,
        "cost_usd": report.receipt.usd if report.receipt else 0.0,
        "notes": report.notes,
        "reply_instructions": MEASURE_REPLY,
    }
    if report.proof_path:
        payload["media_tag"] = f"MEDIA:{report.proof_path}"
        payload["media_note"] = media_instructions(report.proof_path)
    return payload


def measure(args: dict, **_kw: Any) -> str:
    try:
        from .screentime_core.budget import Budget
        from .screentime_core.format import measure_summary
        from .screentime_core.pipeline import MeasureRequest, run_measure
        from .screentime_core.store import RunStore, new_run_id

        video = hb.resolve_path(str(args.get("video") or ""))
        photo = hb.resolve_path(str(args["photo"])) if args.get("photo") else None
        product = " ".join(str(args.get("product") or "").split())[:300]
        noun = " ".join(str(args.get("object_noun") or "").split())[:40]
        if not product or not noun:
            raise ScreentimeError("bad_input", "Measure needs the product and a plain object noun (e.g. 'drink can').")
        engines = _engines(require_key=True)
        conn = hb.db()
        budget = Budget(conn, per_video_usd=hb.setting("max_usd_per_video", 0.5),
                        per_day_usd=hb.setting("max_usd_per_day", 5.0))
        run_id = new_run_id()
        req = MeasureRequest(
            video_path=str(video),
            product=product,
            sam_prompt=noun,
            owner=hb.owner_key() or "",
            photo_path=str(photo) if photo else None,
            brand_names=[" ".join(str(b).split())[:60] for b in (args.get("brand_names") or []) if str(b).strip()][:6],
            terms=terms_from_args(args.get("terms")),
            max_video_seconds=hb.setting("max_video_seconds", 120),
        )
        report = run_measure(req, engines, workdir=hb.work_dir() / run_id, outdir=hb.out_dir(run_id), budget=budget,
                             run_id=run_id)
        summary = measure_summary(report)
        if req.owner:
            RunStore(conn).save(report, owner=req.owner, summary=summary.splitlines()[0])
        else:
            report.notes.append("This chat could not be identified, so the result was not saved for follow-ups.")
        return _dump(_measure_payload(report, summary))
    except ScreentimeError as exc:
        return _fail(exc)
    except Exception as exc:  # never raise into the agent loop
        return _internal("measure", exc)


def compare(args: dict, **_kw: Any) -> str:
    try:
        from .screentime_core import card, teardown
        from .screentime_core.format import compare_summary, media_instructions, teardown_summary
        from .screentime_core.store import RunStore, new_run_id

        owner = hb.owner_key()
        store = RunStore(hb.db()) if owner else None
        run_id = new_run_id("tc")
        work = hb.work_dir() / run_id
        out = hb.out_dir(run_id)

        def analyze(p: str, tag: str):
            path = hb.resolve_path(p)
            rep = teardown.analyze(str(path), str(work / tag), run_id=f"{run_id}-{tag}")
            prior = store.latest_measure_for(rep.video_sha256, owner=owner) if store else None
            if prior:  # reuse the paid Measure run of the same file: product and brand timing at no cost
                rep.product_first_seen_sec = prior["metrics"].get("first_seen_sec")
                said = prior["metrics"].get("mentions") or []
                rep.brand_first_said_sec = said[0]["start_sec"] if said else None
            return rep

        yours = analyze(str(args.get("video") or ""), "yours")
        if args.get("reference"):
            theirs = analyze(str(args["reference"]), "ref")
            rep = teardown.compare(yours, theirs, run_id=run_id)
            png = out / f"compare_{run_id}.png"
            card.render_compare_card(rep, str(png))
            rep.card_path = str(png)
            summary = compare_summary(rep)
            if store:
                store.save(rep, owner=owner, summary="compare: " + (summary.splitlines()[1] if len(summary.splitlines()) > 1 else ""))
            payload = {"ok": True, "run_id": run_id, "summary": summary,
                       "rows": [{"metric": r.metric, "yours": r.yours, "reference": r.theirs, "difference": r.diff}
                                for r in rep.rows]}
        else:
            png = out / f"teardown_{run_id}.png"
            card.render_teardown_card(yours, str(png))
            yours.card_path = str(png)
            summary = teardown_summary(yours)
            if store:
                store.save(yours, owner=owner, summary="teardown: " + summary.splitlines()[0])
            payload = {"ok": True, "run_id": run_id, "summary": summary,
                       "cuts_sec": yours.cuts_sec, "shots": len(yours.shots), "first_cut_sec": yours.first_cut_sec}
        payload.update({"media_tag": f"MEDIA:{png}", "media_note": media_instructions(str(png)),
                        "method": "ffmpeg scdet, silencedetect, ebur128; measured, not predicted",
                        "reply_instructions": COMPARE_REPLY})
        return _dump(payload)
    except ScreentimeError as exc:
        return _fail(exc)
    except Exception as exc:
        return _internal("compare", exc)


def runs(args: dict, **_kw: Any) -> str:
    try:
        from .screentime_core.store import RunStore

        owner = hb.owner_key()
        if not owner:
            return _dump({"ok": False, "error": "bad_input",
                          "message": "This chat could not be identified, so earlier results are not shown."})
        store = RunStore(hb.db())
        run_id = str(args.get("run_id") or "").strip()
        if run_id:
            rep = store.get(run_id, owner=owner)
            if not rep:
                return _dump({"ok": False, "error": "bad_input", "message": f"No run {run_id} in this chat."})
            rep.pop("detections", None)  # thousands of boxes: not useful to the model
            return _dump({"ok": True, "report": rep})
        limit = max(1, min(int(args.get("limit") or 10), 50))
        return _dump({"ok": True, "runs": store.recent(owner=owner, limit=limit)})
    except ScreentimeError as exc:
        return _fail(exc)
    except Exception as exc:
        return _internal("runs", exc)
