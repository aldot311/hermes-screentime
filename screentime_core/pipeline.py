"""Measure: one video, one product, every frame.

normalize (15 fps, longest edge 960) -> track (SAM 3.1, or a recorded run) -> check each tracked object against the
product (verify.py) -> listen for the brand (Voice Transcribe, optional) -> metrics -> typed terms -> proof reel.

Ported from Screen-Time's src/lib/pipeline/run.ts, without the web app's campaigns, blob storage and follow-up agent.
Prompt recovery is left to the agent: when SAM finds nothing, the result says so and suggests plainer nouns, and the
agent (the user's Hermes model) can call the tool again with one of them.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from . import budget as budget_mod
from .errors import ScreentimeError
from .format import short_label
from .models import (
    FPS,
    ContractTerm,
    Detection,
    Disclosure,
    MeasureReport,
    PromptAttempt,
    ReceiptCall,
    RunReceipt,
    TermResult,
    TrackVerification,
)
from .store import new_run_id
from .verify import Verifier, pick_frames, unchecked

UNBILLED_CODES = ("auth", "billing", "model_not_found", "bad_input", "rate_limited")
MAX_VERIFY_OBJECTS = 6  # objects checked per run, most-seen first; the rest are "not checked" (never counted)
MAX_VIDEO_SECONDS_DEFAULT = 120
NOUN_SUGGESTIONS = ["can", "bottle", "cup", "box", "bag", "shoe", "phone", "jar", "tube", "package"]

ProgressFn = Callable[[str, dict], None]


@dataclass
class ReplayData:
    """A recorded Measure run of a bundled clip: SAM output, the objects' verdicts and (optionally) transcript turns."""

    video_path: str
    sam_text: str
    sam_meta: dict
    verdicts: dict[int, dict]
    turns: Optional[list[dict]] = None
    product: str = ""
    sam_prompt: str = ""
    photo_path: Optional[str] = None
    brand_names: list[str] = field(default_factory=list)
    audio_path: Optional[str] = None  # the clip's own audio (the normalized sample has none), for the proof reel

    @classmethod
    def load(cls, folder: Path) -> ReplayData:
        meta = json.loads((folder / "sample.sam.json").read_text())
        verdicts_raw = json.loads((folder / "sample.verdicts.json").read_text())
        turns_path = folder / "sample.turns.json"
        return cls(
            video_path=str(folder / "sample.mp4"),
            sam_text=(folder / "sample.sam.txt").read_text(),
            sam_meta=meta,
            verdicts={int(k): v for k, v in verdicts_raw["verdicts"].items()},
            turns=json.loads(turns_path.read_text())["turns"] if turns_path.exists() else None,
            product=verdicts_raw.get("product", ""),
            sam_prompt=meta.get("prompt", "drink can"),
            photo_path=str(folder / "product.jpg") if (folder / "product.jpg").exists() else None,
            brand_names=verdicts_raw.get("brand_names", []),
            audio_path=str(folder / "sample_audio.m4a") if (folder / "sample_audio.m4a").exists() else None,
        )


@dataclass
class MeasureRequest:
    video_path: str
    product: str
    sam_prompt: str
    owner: str = "local"
    photo_path: Optional[str] = None
    brand_names: list[str] = field(default_factory=list)
    terms: list[ContractTerm] = field(default_factory=list)
    max_video_seconds: int = MAX_VIDEO_SECONDS_DEFAULT


@dataclass
class Engines:
    tracker: str  # "meta" | "vision" | "replay"
    verifier: Verifier
    client: Any = None  # MetaClient when tracker == "meta" or when transcribing with Meta
    transcribe: bool = False  # listen with Muse Voice Transcribe (needs client)
    replay: Optional[ReplayData] = None
    vision: Any = None  # VisionTracker when tracker == "vision"
    stt: Optional[Callable[[str], dict]] = None  # Hermes speech-to-text: wav path -> {"success", "transcript", "provider"}
    stt_label: str = "hermes-stt"
    speech_off_reason: str = ""  # why speech is not checked this run (shown as a note when the video has audio)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _photo_jpeg(path: Optional[str], max_side: int = 768) -> Optional[bytes]:
    """The user's product photo as a JPEG no larger than max_side (Pillow)."""
    if not path:
        return None
    from PIL import Image, ImageOps

    p = Path(path)
    if not p.is_file():
        raise ScreentimeError("bad_input", f"Product photo not found: {p.name}", fix="Attach the photo again or give its path.")
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return buf.getvalue()


def _order_tracks(tracks: dict[int, list[Detection]]) -> list[int]:
    """Most-seen objects first (ties: larger first), so the cap drops fleeting detections, not the product."""
    return sorted(tracks, key=lambda oid: (-len(tracks[oid]), -max(d.area_fraction for d in tracks[oid]), oid))


def run_measure(
    req: MeasureRequest,
    engines: Engines,
    *,
    workdir: Path,
    outdir: Path,
    budget: Optional[budget_mod.Budget],
    run_id: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    render_proof: bool = True,
) -> MeasureReport:
    from . import media
    from . import mentions as mentions_mod
    from . import metrics as metrics_mod
    from . import rules as rules_mod
    from . import sam as sam_mod

    run_id = run_id or new_run_id()
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, data: None)
    calls: list[ReceiptCall] = []
    notes: list[str] = []
    started = _now_iso()

    src = Path(req.video_path)
    if engines.tracker == "replay":
        assert engines.replay is not None
        src = Path(engines.replay.video_path)
    if not src.is_file():
        raise ScreentimeError("bad_input", f"Video not found: {src.name}", fix="Attach the video again or give its full path.")

    # 1. Probe and normalize ------------------------------------------------------------------------------------------
    say("normalize", {})
    original = media.probe(str(src))
    if original.duration_sec > req.max_video_seconds:
        est = budget_mod.measure_estimate(int(original.duration_sec * FPS))
        raise ScreentimeError(
            "too_long",
            f"This video is {original.duration_sec:.0f} s; Measure takes up to {req.max_video_seconds} s per run "
            f"(about ${est:.2f} for the whole video).",
            fix="Trim it, or raise max_video_seconds in the plugin settings.",
        )
    sha = media.sha256_file(str(src))
    if engines.tracker == "replay":
        norm = src  # the bundled clip is already normalized
        info = media.probe(str(norm))
    else:
        norm = workdir / f"{run_id}.mp4"
        info = media.normalize(str(src), str(norm), fps=FPS, max_edge=960)
    replay_audio = engines.replay.audio_path if (engines.tracker == "replay" and engines.replay) else None
    info = replace(info, has_audio=original.has_audio or bool(replay_audio))
    say("normalized", {"frames": info.frame_count, "seconds": info.duration_sec, "width": info.width, "height": info.height})

    # 2. Track ---------------------------------------------------------------------------------------------------------
    sam_prompt = req.sam_prompt.strip() or "object"
    say("tracking", {"prompt": sam_prompt, "frames": info.frame_count})
    if engines.tracker == "replay":
        rp = engines.replay
        sam_text = rp.sam_text
        calls.append(ReceiptCall("replay:sam-3.1", f'Track "{rp.sam_prompt}" (recorded {rp.sam_meta.get("recorded_at", "")})',
                                 f"{info.frame_count} frames", "recorded output", 0, 0.0, replayed=True))
        sam_prompt = rp.sam_prompt
    elif engines.tracker == "vision":
        if engines.vision is None:
            raise ScreentimeError("engine_unavailable", "No model is set up to track the product.",
                                  fix="Set a vision-capable model in Hermes, or add a Meta Model API key.")
        vt = engines.vision.track(str(norm), info, sam_prompt, workdir,
                                  progress=lambda done, total: say("tracking", {"calls_done": done, "calls": total}))
        calls.append(vt.call)
        notes.extend(vt.notes)
        sam_text = None
    else:
        if engines.client is None:
            raise ScreentimeError("engine_unavailable", "Measure needs a Meta Model API key to track the product.",
                                  fix="Add it in Settings > Plugins > screentime, or try /screentime demo.")
        est = budget_mod.sam_estimate(info.frame_count)
        res = budget.reserve(run_id, est, f'Track "{sam_prompt}"') if budget else None
        file_id = None
        try:
            try:
                file_id = engines.client.upload_file(str(norm), expires_after_sec=3600)
            except ScreentimeError:
                if budget and res:
                    budget.release(res)  # uploads are not billed
                raise
            try:
                sam = engines.client.sam_track(sam_prompt, file_id=file_id, frame_count=info.frame_count)
            except ScreentimeError as exc:
                if budget and res and exc.code in UNBILLED_CODES:
                    budget.release(res)  # refused before anything was billed
                raise
        finally:
            if file_id:
                try:
                    engines.client.delete_file(file_id)
                except Exception:  # best effort; the upload also expires on its own
                    notes.append("The uploaded copy could not be deleted now; it expires within an hour.")
        if budget and res:
            budget.settle(res, sam.usd)
        sam_text = sam.text
        calls.append(ReceiptCall("sam-3.1", f'Track "{sam_prompt}"', f"{info.frame_count} frames at {FPS} fps",
                                 f"{sam.frames_billed} frames billed", int(sam.ms), float(sam.usd)))
    detections = sam_mod.parse_sam_text(sam_text, info.width, info.height) if sam_text is not None else vt.detections
    tracks = sam_mod.group_tracks(detections)
    say("tracked", {"objects": len(tracks)})

    # 3. Check each object against the product -----------------------------------------------------------------------
    photo = _photo_jpeg(req.photo_path or (engines.replay.photo_path if engines.replay else None))
    product = req.product or (engines.replay.product if engines.replay else "")
    order = _order_tracks(tracks)
    verifications: dict[int, TrackVerification] = {}
    for oid in order[MAX_VERIFY_OBJECTS:]:
        verifications[oid] = unchecked(oid, f"only the {MAX_VERIFY_OBJECTS} most-seen objects are checked")
    to_check = order[:MAX_VERIFY_OBJECTS]
    paid_checks = getattr(engines.verifier, "name", "").startswith("muse-spark")
    reservations = {}
    for oid in to_check:
        if paid_checks and budget:
            try:
                reservations[oid] = budget.reserve(run_id, budget_mod.SPARK_CHECK_RESERVE_USD, f"Check object {oid + 1}")
            except ScreentimeError as exc:
                verifications[oid] = unchecked(oid, exc.message)

    def check_one(oid: int):
        picks = pick_frames(tracks[oid], info.frame_count)
        crops = [media.crop_jpeg(media.frame_jpeg(str(norm), d.frame, fps=FPS), d.box, pad=0.15) for d in picks]
        return engines.verifier.check(object_id=oid, product=product, sam_prompt=sam_prompt, photo_jpeg=photo, crops=crops)

    pending = [oid for oid in to_check if oid not in verifications]
    say("checking", {"objects": len(pending), "verifier": getattr(engines.verifier, "name", "?")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {oid: pool.submit(check_one, oid) for oid in pending}
        for oid, fut in futures.items():
            try:
                tv, call = fut.result()
                verifications[oid] = tv
                calls.append(call)
                if budget and oid in reservations:
                    budget.settle(reservations[oid], call.usd)
            except ScreentimeError as exc:
                verifications[oid] = unchecked(oid, exc.message)
                if exc.code == "verifier_unavailable":
                    notes.append(exc.message + (f" {exc.fix}" if exc.fix else ""))
            say("checked", {"object": oid, "verdict": verifications[oid].verdict})
    track_list = [verifications[oid] for oid in sorted(verifications)]
    verdict_by_object = {str(t.object_id): t.verdict for t in track_list}
    counted = {t.object_id for t in track_list if t.verdict == "match"}
    # Objects nobody could confirm either way (not checked, or checked as "unclear"): never counted, but a term that
    # fails only because of them goes to a person instead of failing (a false FAIL would hold a creator's fee).
    unchecked_ids = {t.object_id for t in track_list if t.checked_by == "unchecked" or t.verdict == "unclear"}

    # 4. Listen ------------------------------------------------------------------------------------------------------
    mentions: list = []
    disclosures: list = []
    spoken_checked: Optional[bool] = None
    brand_names = req.brand_names or (engines.replay.brand_names if engines.replay else [])
    turns = None
    if engines.tracker == "replay" and engines.replay and engines.replay.turns is not None:
        turns = engines.replay.turns
        calls.append(ReceiptCall("replay:muse-voice-transcribe-1.0", "Listen for the brand (recorded)", "audio",
                                 f"{len(turns)} turns", 0, 0.0, replayed=True))
    elif engines.transcribe and engines.client is not None and info.has_audio:
        say("listening", {})
        wav = workdir / f"{run_id}.wav"
        if media.extract_audio_wav(str(src), str(wav)):
            res = None
            try:
                res = budget.reserve(run_id, budget_mod.TRANSCRIBE_RESERVE_USD, "Listen for the brand") if budget else None
                t0 = time.monotonic()
                transcript = engines.client.transcribe(str(wav), keywords=brand_names)
                turns = [{"start_sec": t.start_sec, "end_sec": t.end_sec, "text": t.text} for t in transcript]
                usd = float(getattr(transcript, "usd", None) or budget_mod.TRANSCRIBE_RESERVE_USD)
                ms = int(getattr(transcript, "ms", None) or (time.monotonic() - t0) * 1000)
                if budget and res:
                    budget.settle(res, usd)
                calls.append(ReceiptCall("muse-voice-transcribe-1.0", "Listen for the brand",
                                         f"{original.duration_sec:.1f} s of audio", f"{len(turns)} turns", ms, usd))
            except ScreentimeError as exc:
                if budget and res and exc.code in UNBILLED_CODES:
                    budget.release(res)
                spoken_checked = False
                notes.append(f"Speech was not checked: {exc.message}")
    elif engines.stt is not None and info.has_audio:
        say("listening", {})
        wav = workdir / f"{run_id}.wav"
        if media.extract_audio_wav(str(src), str(wav)):
            t0 = time.monotonic()
            turns, why = listen_in_windows(engines.stt, wav, workdir / "speech")
            if turns is None:
                spoken_checked = False
                notes.append(f"Speech was not checked: {why}")
            else:
                calls.append(ReceiptCall(engines.stt_label, "Listen for the brand",
                                         f"{original.duration_sec:.1f} s of audio in {WINDOW_SEC:.0f} s windows",
                                         f"{len(turns)} windows with speech", int((time.monotonic() - t0) * 1000), 0.0))
    if turns is None and info.has_audio and engines.speech_off_reason and spoken_checked is None:
        notes.append(f"Speech was not checked: {engines.speech_off_reason}")
    if turns is not None:
        mentions = mentions_mod.find_mentions(turns, brand_names) if brand_names else []
        disclosures = mentions_mod.find_spoken_disclosures(turns)
        if engines.stt is not None and not engines.transcribe and engines.tracker != "replay":
            mentions, disclosures = dedupe_mentions(mentions), dedupe_mentions(disclosures)
        spoken_checked = True if brand_names else spoken_checked
    elif not info.has_audio:
        spoken_checked = None

    # 5. Measure and check terms ----------------------------------------------------------------------------------------
    say("measuring", {})
    metrics = metrics_mod.compute(detections, counted, info, mentions=mentions, spoken_checked=spoken_checked)
    disclosure = Disclosure(spoken=disclosures, spoken_checked=(turns is not None) if info.has_audio else None)
    terms = rules_mod.evaluate(req.terms, metrics, info, disclosure) if req.terms else []
    if unchecked_ids and terms:
        if_counted = metrics_mod.compute(detections, counted | unchecked_ids, info, mentions=mentions,
                                         spoken_checked=spoken_checked)
        terms = open_for_unchecked(terms, rules_mod.evaluate(req.terms, if_counted, info, disclosure), len(unchecked_ids))

    tracker_label = vt.label if engines.tracker == "vision" else "sam-3.1"
    if engines.tracker == "replay" or engines.transcribe:
        listen_label = "muse-voice-transcribe-1.0"
    else:
        listen_label = engines.stt_label
    attempts = [PromptAttempt(sam_prompt, len(tracks), len(counted), info.frame_count)]
    if not tracks:
        notes.append(
            f'{"Your model" if engines.tracker == "vision" else "SAM 3.1"} found no object for "{sam_prompt}". Try a plainer one-to-three word noun for the object\'s shape, '
            f"for example: {', '.join(n for n in NOUN_SUGGESTIONS if n not in sam_prompt)[:80]}."
        )

    report = MeasureReport(
        run_id=run_id,
        mode="measure",
        video_path=str(src),
        video_sha256=sha,
        info=info,
        product=product,
        prompt=sam_prompt,
        engine=engines.tracker,
        verifier=getattr(engines.verifier, "name", "none"),
        verifier_validated=bool(getattr(engines.verifier, "validated", False)),
        tracks=track_list,
        metrics=metrics,
        terms=terms,
        detections=detections,
        verdict_by_object=verdict_by_object,
        disclosure=disclosure,
        prompt_attempts=attempts,
        receipt=RunReceipt(run_id, started, FPS,
                           {"sam": tracker_label, "check": getattr(engines.verifier, "name", "none"),
                            "transcribe": listen_label if turns is not None else "none"},
                           calls),
        notes=notes,
    )

    # 6. Proof reel ----------------------------------------------------------------------------------------------------
    if render_proof:
        say("rendering", {})
        from . import proof as proof_mod

        out_mp4 = outdir / f"proof_{run_id}.mp4"
        try:
            proof_mod.render_proof(str(norm), detections, verdict_by_object, metrics, info, str(out_mp4),
                                   product_label=short_label(product) or sam_prompt, audio_source=replay_audio or (str(src) if original.has_audio else None))
            report.proof_path = str(out_mp4)
        except ScreentimeError as exc:
            report.notes.append(f"The proof reel could not be made: {exc.message}")
    say("done", {"seconds_on_screen": metrics.seconds_on_screen})
    return report


WINDOW_SEC = 5.0
STRIDE_SEC = 4.0  # windows overlap by 1 s so a word on a boundary is heard whole in one of them
MAX_WINDOWS = 40


def listen_in_windows(stt: Callable[[str], dict], wav: Path, outdir: Path) -> tuple[Optional[list[dict]], str]:
    """Hermes speech-to-text returns text without timestamps, so the audio goes in short overlapping windows and each
    window becomes one turn with its own times. Returns (turns, "") or (None, why)."""
    import wave

    outdir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "rb") as w:
        rate, width, channels, total = w.getframerate(), w.getsampwidth(), w.getnchannels(), w.getnframes()
        pcm = w.readframes(total)
    dur = total / float(rate)
    starts = []
    t = 0.0
    while t < dur and len(starts) < MAX_WINDOWS:
        starts.append(t)
        if t + WINDOW_SEC >= dur:
            break
        t += STRIDE_SEC
    turns: list[dict] = []
    errors: list[str] = []
    frame_bytes = width * channels
    for i, start in enumerate(starts):
        end = min(dur, start + WINDOW_SEC)
        if end - start < 0.3:
            continue
        part = outdir / f"w{i:03d}.wav"
        with wave.open(str(part), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(pcm[int(start * rate) * frame_bytes:int(end * rate) * frame_bytes])
        try:
            res = stt(str(part)) or {}
        except Exception as exc:  # the host's STT failing is a note, not a failed run
            res = {"success": False, "error": type(exc).__name__}
        if not res.get("success"):
            errors.append(str(res.get("error") or "no transcript"))
            continue
        text = " ".join(str(res.get("transcript") or "").split())
        if text:
            turns.append({"start_sec": round(start, 2), "end_sec": round(end, 2), "text": text})
    if errors and len(errors) == len(starts):
        return None, f"Hermes speech-to-text did not answer ({errors[0][:120]})."
    return turns, ""


def dedupe_mentions(items: list, gap_sec: float = 1.5) -> list:
    """Overlapping windows hear a word twice; keep the first of any two hits closer than gap_sec."""
    out: list = []
    for m in sorted(items, key=lambda m: m.start_sec):
        if out and m.start_sec - out[-1].start_sec < gap_sec:
            continue
        out.append(m)
    return out


def open_for_unchecked(results: list[TermResult], if_counted: list[TermResult], objects: int) -> list[TermResult]:
    """Terms that failed only because unchecked objects were not counted go to a person (run.ts openForUnchecked)."""
    which = ("1 tracked object could not be confirmed (unclear or not checked); counting it" if objects == 1
             else f"{objects} tracked objects could not be confirmed (unclear or not checked); counting them")
    out = []
    for i, t in enumerate(results):
        alt = if_counted[i] if i < len(if_counted) else None
        if t.passed is False and alt is not None and alt.passed is True:
            t = replace(t, passed=None, detail=f"{t.detail} {which} would meet this term.")
        out.append(t)
    return out
