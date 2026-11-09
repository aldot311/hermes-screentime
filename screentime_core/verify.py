"""The look-alike check: is each tracked object the brand's exact product?

Ported from Screen-Time (src/lib/pipeline/prompts.ts: VERIFY_SCHEMA, verifyPrompt, toChecks; run.ts: verifyTrack,
pickFrames). Three interchangeable verifiers:

- MetaSparkVerifier: Muse Spark 1.3 on Meta's Model API, the configuration the app was measured with (validated).
- HermesLlmVerifier: the user's own Hermes model through a structured-output callable the plugin glue provides
  (ctx.llm.complete_structured with image blocks). Not validated on this task, and every report says so.
- ReplayVerifier: recorded verdicts, for the keyless demo and for tests.

A failed or skipped check never counts: the object becomes "unclear" with the reason, and terms that only that object
could have satisfied are left for a person (see pipeline.open_for_unchecked).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .errors import ScreentimeError
from .models import Detection, ReceiptCall, TrackVerification

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "visibility", "reason", "checks"],
    "properties": {
        "verdict": {"type": "string", "enum": ["match", "lookalike", "unclear"]},
        "visibility": {"type": "string", "enum": ["clear", "partial", "blurred"]},
        "reason": {"type": "string"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["feature", "seen"],
                "properties": {
                    "feature": {"type": "string"},
                    "seen": {"type": "string", "enum": ["yes", "no", "unclear"]},
                },
            },
        },
    },
}

VALID_VERDICTS = ("match", "lookalike", "unclear")
VALID_VISIBILITY = ("clear", "partial", "blurred")


def verify_prompt(product_description: str, sam_prompt: str, with_photo: bool = False) -> str:
    """The app's object-check prompt, word for word (prompts.ts verifyPrompt)."""
    parts = [
        (
            f"The first image is the brand's reference photo of its product. The other images are crops from one "
            f'creator video of the object a video tracker returned when it searched for "{sam_prompt}".'
        )
        if with_photo
        else f'A video tracker searched for "{sam_prompt}" and returned the object shown in these crops from one creator video.',
        f"The brand's product is: {product_description}.",
    ]
    if with_photo:
        parts.append("Compare the crops against the reference photo as well as the description.")
    parts += [
        'Decide whether the tracked object is that exact product ("match"), a different product of the same kind',
        '("lookalike", for example another brand\'s can), or impossible to tell ("unclear").',
        "Judge color, shape, logo and label text. If the description names no logo, brand mark or label, judge only the",
        "features it does name (color, shape, material) and never answer \"unclear\" just because no logo is visible;",
        '"unclear" is for when the object itself cannot be made out. Visibility: "clear" if the label is readable or the product is',
        'unmistakable, "partial" if covered or cut off, "blurred" if motion or focus hides it.',
        "Give a one-sentence reason a brand manager would understand.",
        "In \"checks\", list up to five distinguishing features of the brand's product, taken from the description"
        + (" and the photo" if with_photo else ""),
        '(for example "teal can", "pink Alani script", "peach graphic"), each in a few words, and whether the crops show it:',
        '"yes", "no" (a different color, logo or shape is visible instead), or "unclear" (hidden, blurred or out of frame).',
    ]
    return " ".join(parts)


def to_checks(raw: Any) -> list[dict[str, str]]:
    """At most six features, short names, known answers only (prompts.ts toChecks)."""
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if not isinstance(c, dict) or not isinstance(c.get("feature"), str) or not isinstance(c.get("seen"), str):
            continue
        feature = " ".join(c["feature"].split())[:60]
        if feature and c["seen"] in ("yes", "no", "unclear"):
            out.append({"feature": feature, "seen": c["seen"]})
    return out[:6]


def pick_frames(detections: list[Detection], frame_count: int) -> list[Detection]:
    """First, middle and largest appearance, distinct by frame (run.ts pickFrames)."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: d.frame)
    largest = ordered[0]
    for d in ordered:
        if d.area_fraction > largest.area_fraction:
            largest = d
    picks = [ordered[0], ordered[len(ordered) // 2], largest]
    seen: dict[int, Detection] = {}
    for d in picks:
        if d.frame < frame_count and d.frame not in seen:
            seen[d.frame] = d
    return list(seen.values())


def parse_verdict(object_id: int, out: Any, checked_by: str) -> TrackVerification:
    """A model's answer as a TrackVerification; anything malformed becomes "unclear" (never counted)."""
    if not isinstance(out, dict) or out.get("verdict") not in VALID_VERDICTS:
        return TrackVerification(object_id, "unclear", "clear", "The check returned an unreadable answer; not counted.",
                                 [], checked_by)
    visibility = out.get("visibility") if out.get("visibility") in VALID_VISIBILITY else "clear"
    reason = " ".join(str(out.get("reason") or "").split())[:300]
    return TrackVerification(object_id, out["verdict"], visibility, reason, to_checks(out.get("checks")), checked_by)


class Verifier(Protocol):
    name: str
    validated: bool

    def check(self, *, object_id: int, product: str, sam_prompt: str, photo_jpeg: Optional[bytes],
              crops: list[bytes]) -> tuple[TrackVerification, ReceiptCall]: ...


# ---------------------------------------------------------------------------------------------------------------------


class MetaSparkVerifier:
    """Muse Spark 1.3 with the app's exact request: strict schema, low reasoning, photo first, then crops."""

    validated = True

    def __init__(self, client: Any, model: str = "muse-spark-1.3"):
        self.client = client
        self.model = model
        self.name = model

    def check(self, *, object_id, product, sam_prompt, photo_jpeg, crops):
        with_photo = photo_jpeg is not None
        parts: list[dict[str, Any]] = [{"type": "text", "text": verify_prompt(product, sam_prompt, with_photo)}]
        for img in ([photo_jpeg] if with_photo else []) + list(crops):
            parts.append({"type": "image", "jpeg": img})
        res = self.client.spark_json(model=self.model, parts=parts, json_schema=VERIFY_SCHEMA,
                                     schema_name="product_check", reasoning_effort="low")
        tv = parse_verdict(object_id, res.parsed, self.model)
        call = ReceiptCall(
            model=self.model,
            purpose=f"Check object {object_id + 1} against the product",
            input=f"{len(crops)} crop{'s' if len(crops) != 1 else ''} of the object"
            + (" + the product photo" if with_photo else "") + " + the description",
            output=f"{tv.verdict}, {tv.visibility}",
            ms=int(res.ms),
            usd=float(res.usd),
        )
        return tv, call


# A structured-output callable supplied by the Hermes glue:
#   llm(instructions, images: list[tuple[bytes, mime]], json_schema, schema_name) -> dict with keys
#   parsed, text, provider, model, usd (usd may be None)
HermesStructuredCall = Callable[[str, list[tuple[bytes, str]], dict, str], dict]


class HermesLlmVerifier:
    """The user's own Hermes model (or the model the operator set for the screentime_verify auxiliary task)."""

    validated = False

    def __init__(self, llm: HermesStructuredCall, label: str = "hermes"):
        self.llm = llm
        self.name = label

    def check(self, *, object_id, product, sam_prompt, photo_jpeg, crops):
        with_photo = photo_jpeg is not None
        images = ([(photo_jpeg, "image/jpeg")] if with_photo else []) + [(c, "image/jpeg") for c in crops]
        t0 = time.monotonic()
        try:
            res = self.llm(verify_prompt(product, sam_prompt, with_photo), images, VERIFY_SCHEMA, "product_check")
        except ScreentimeError:
            raise
        except Exception as exc:  # the host model may not take images, or may not be configured
            raise ScreentimeError(
                "verifier_unavailable",
                f"Your Hermes model could not check the crops ({type(exc).__name__}).",
                fix="Set a vision-capable model for the screentime_verify task (auxiliary.screentime_verify), "
                "or add a Meta Model API key so Muse Spark checks them.",
            ) from None
        label = f"hermes:{res.get('provider') or '?'}/{res.get('model') or '?'}"
        self.name = label
        tv = parse_verdict(object_id, res.get("parsed"), label)
        call = ReceiptCall(
            model=label,
            purpose=f"Check object {object_id + 1} against the product",
            input=f"{len(crops)} crop{'s' if len(crops) != 1 else ''}" + (" + the product photo" if with_photo else ""),
            output=f"{tv.verdict}, {tv.visibility}",
            ms=int((time.monotonic() - t0) * 1000),
            usd=float(res.get("usd") or 0.0),
        )
        return tv, call


@dataclass
class ReplayVerifier:
    """Recorded verdicts keyed by object id, labelled as replayed in every output."""

    verdicts: dict[int, dict[str, Any]]
    source_model: str = "muse-spark-1.3"
    validated: bool = True

    @property
    def name(self) -> str:
        return f"replay:{self.source_model}"

    def check(self, *, object_id, product, sam_prompt, photo_jpeg, crops):
        rec = self.verdicts.get(object_id)
        tv = parse_verdict(object_id, rec, self.name)
        call = ReceiptCall(model=self.name, purpose=f"Check object {object_id + 1} (recorded)",
                           input=f"{len(crops)} crops", output=f"{tv.verdict}, {tv.visibility}", ms=0, usd=0.0,
                           replayed=True)
        return tv, call


def unchecked(object_id: int, why: str) -> TrackVerification:
    """An object that was not checked: never counted."""
    return TrackVerification(object_id, "unclear", "clear", f"Not checked: {why}; not counted.", [], "unchecked")
