"""Meta Model API client: Files upload and delete, SAM 3.1 video tracking, Muse Spark structured output, and Muse Voice
Transcribe. Port of the Screen-Time app's src/lib/meta/{client,http,sse}.ts.

The API key is passed in by the caller and is never logged, returned or put in an exception: error messages are
scrubbed of it, and errors are raised outside ``except`` blocks so no transport exception (which holds the request and
its headers) is chained to them.

Retries: only 429 and 5xx answers, and connection failures from before the request reached the server, are tried
again (at most MAX_TRIES attempts, full-jitter backoff, Retry-After as a floor). A billed call (SAM, Spark, transcribe)
is never sent again once it may have reached the server: after a read or write failure, a timeout, or once a 200
arrived. Uploads are not billed, so they are also retried after a dropped connection, as in the app.
"""

from __future__ import annotations

import base64
import email.utils
import json
import math
import random
import re
import socket
import threading
import time
import wave
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx

from .errors import ScreentimeError

BASE_URL = "https://api.meta.ai/v1"

SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"
MODELS = {"sam": SAM_MODEL, "spark": SPARK_MODEL, "transcribe": TRANSCRIBE_MODEL}

# Prices, Standard mode (never Contributor). Source: dev.meta.ai/docs pricing, model and speech-to-text pages, read and
# cross-checked 2026-10-28 (screen-time docs/meta-api-notes.md); the SAM figures match the app's cost guard
# (src/lib/meta/cost-guard.ts).
SAM_USD_PER_FRAME = 0.20 / 1000  # $0.20 per 1,000 video frames
SAM_MIN_FRAMES = 50  # each video request is billed for at least 50 frames
SPARK_USD_PER_M_INPUT = 1.25  # muse-spark-1.3, per million input tokens
SPARK_USD_PER_M_CACHED_INPUT = 0.15
SPARK_USD_PER_M_OUTPUT = 4.25  # output includes reasoning tokens
TRANSCRIBE_USD_PER_HOUR = 0.18  # per hour of processed audio, rounded down to whole seconds
SPARK_RESERVE_USD = 0.02
"""The app's per-call Spark reserve. Used as a Spark call's usd only when the response carries no usage."""

UPLOAD_TIMEOUT_SEC = 60.0  # per attempt (run.ts UPLOAD_TIMEOUT_MS)
SPARK_TIMEOUT_SEC = 60.0  # run.ts SPARK_IMAGES_TIMEOUT_MS
DELETE_TIMEOUT_SEC = 30.0
CONNECT_TIMEOUT_SEC = 10.0
MAX_TRIES = 3
BASE_DELAY_SEC = 0.5
MAX_DELAY_SEC = 30.0
RETRY_AFTER_CEILING_SEC = 60.0
"""Longest a retry may wait; a longer Retry-After fails the call instead of hanging (as in the app)."""

UPLOAD_MAX_BYTES = 1 << 30  # Files API limit, 1 GiB
UPLOAD_EXPIRY_RANGE_SEC = (3600, 2_592_000)  # expires_after.seconds: 1 hour to 30 days
MAX_SPARK_IMAGES = 50  # more images in one request is a 400
TRANSCRIBE_MAX_SEC = 600.0  # /asr/transcribe takes at most 10 minutes of audio
TRANSCRIBE_MAX_BYTES = 32 * 1024 * 1024
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
USER_AGENT = "hermes-screentime/0.1.0"


def sam_timeout_sec(frame_count: int) -> float:
    """30 s plus 0.15 s per frame (run.ts samTimeoutMs), from the start of the attempt to the end of the stream."""
    return 30.0 + 0.15 * frame_count


def transcribe_timeout_sec(audio_sec: float) -> float:
    """30 s plus 0.5 s per second of audio (run.ts transcribeTimeoutMs)."""
    return 30.0 + 0.5 * audio_sec


def sam_usd(frames: int) -> float:
    """What a SAM 3.1 video request is billed for `frames` frames (50-frame minimum)."""
    return round(max(math.ceil(frames), SAM_MIN_FRAMES) * SAM_USD_PER_FRAME, 6)


def spark_usd(usage: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Muse Spark 1.3 Standard cost from a Responses usage block; None when the usage is missing or unreadable."""
    if not isinstance(usage, Mapping):
        return None
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if not _is_count(inp) or not _is_count(out):
        return None
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, Mapping) else 0
    cached = min(int(cached), int(inp)) if _is_count(cached) else 0
    usd = ((int(inp) - cached) * SPARK_USD_PER_M_INPUT + cached * SPARK_USD_PER_M_CACHED_INPUT
           + int(out) * SPARK_USD_PER_M_OUTPUT) / 1e6
    return round(usd, 6)


def transcribe_usd(audio_sec: float) -> float:
    """Muse Voice Transcribe cost: $0.18 per hour, rounded down to whole seconds."""
    return round(math.floor(max(0.0, audio_sec)) * TRANSCRIBE_USD_PER_HOUR / 3600, 6)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# --------------------------------------------------------------------------------------------------------------------
# Results


@dataclass
class SamResult:
    text: str  # the output_text lane: one special-token line per frame with a visible object (sam.parse_sam_text)
    frames_billed: int  # usage.video_frames_processed, or the frame count when the response omitted it
    ms: int
    usd: float
    frames_estimated: bool = False  # True when frames_billed is the frame count, not the API's own figure


@dataclass
class SparkResult:
    parsed: dict[str, Any]
    text: str
    ms: int
    usd: float  # from usage; SPARK_RESERVE_USD when the response had no usage (then usd_estimated is True)
    usage: dict[str, Any] = field(default_factory=dict)
    usd_estimated: bool = False
    model: str = SPARK_MODEL


@dataclass
class Turn:
    """One speech turn. Times are when the turn was detected in the audio, not exact word boundaries."""

    start_sec: float
    end_sec: float
    text: str


class Transcript(list):
    """The turns (a list of Turn), with the call's details: text, audio_sec, ms, usd."""

    def __init__(self, turns: Iterable[Turn] = (), *, text: str = "", audio_sec: float = 0.0, ms: int = 0,
                 usd: float = 0.0) -> None:
        super().__init__(turns)
        self.text = text
        self.audio_sec = audio_sec
        self.ms = ms
        self.usd = usd


# --------------------------------------------------------------------------------------------------------------------
# Request bodies and response readers (pure, shared with tests)


def build_sam_body(prompt: str, file_id: str, include_confidence: bool = True,
                   mask_encoding: Optional[str] = "one_bit") -> dict[str, Any]:
    """A streamed SAM 3.1 video request; metadata values are strings (client.ts buildSamVideoBody)."""
    metadata = {"include_confidence": "true" if include_confidence else "false"}
    if mask_encoding:
        metadata["mask_encoding"] = mask_encoding
    return {
        "model": SAM_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}],
            }
        ],
        "metadata": metadata,
        "stream": True,
    }


def build_spark_body(content: list[dict[str, Any]], json_schema: Mapping[str, Any], schema_name: str, *,
                     model: str = SPARK_MODEL, system: Optional[str] = None,
                     reasoning_effort: Optional[str] = "low", strict: Optional[bool] = None,
                     max_output_tokens: Optional[int] = None) -> dict[str, Any]:
    """A Responses request constrained to `json_schema` via text.format (client.ts buildSparkJsonBody)."""
    fmt: dict[str, Any] = {"type": "json_schema", "name": schema_name, "schema": dict(json_schema)}
    if strict is not None:
        fmt["strict"] = strict
    body: dict[str, Any] = {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": content}],
        "text": {"format": fmt},
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if system:
        body["instructions"] = system
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    return body


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Server-sent events to JSON event objects (sse.ts parseSseEvents). Comments, keepalives and [DONE] are skipped.
    An unterminated, unparseable final event is a dropped connection and is skipped (the missing terminal event is
    what fails the call); an unparseable event anywhere else raises."""
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                event = _decode_event("\n".join(data), final=False)
                data = []
                if event is not None:
                    yield event
            continue
        if line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        event = _decode_event("\n".join(data), final=True)
        if event is not None:
            yield event


def _decode_event(payload: str, *, final: bool) -> Optional[dict[str, Any]]:
    if payload == "[DONE]":
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        event = _MALFORMED
    if event is _MALFORMED:
        if final:
            return None
        raise ScreentimeError("server", "The Meta Model API sent a malformed stream event.", fix=_FIX["server"],
                              retryable=True)
    if isinstance(event, dict) and isinstance(event.get("type"), str):
        return event
    return None


_MALFORMED = object()


def extract_output_text(body: Mapping[str, Any]) -> str:
    """The output_text of a non-streamed Responses body (client.ts extractJsonOutput, before the JSON parse)."""
    status = body.get("status")
    error = body.get("error") if isinstance(body.get("error"), Mapping) else None
    if (status and status != "completed") or (not status and body.get("error")):
        details = body.get("incomplete_details") if isinstance(body.get("incomplete_details"), Mapping) else {}
        why = details.get("reason") or (error or {}).get("message") or status or "error"
        code = "incomplete" if status == "incomplete" else _code_for_error_code((error or {}).get("code"), "server")
        raise ScreentimeError(code, f"Muse Spark response not completed: {why}", fix=_FIX[code],
                              retryable=_RETRYABLE[code])
    text = ""
    for item in body.get("output") or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "refusal":
                raise ScreentimeError("refused", f"Muse Spark refused: {part.get('refusal') or ''}".strip(),
                                      fix=_FIX["refused"])
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                text += part["text"]
    if not text:
        raise ScreentimeError("incomplete", "Muse Spark response had no output_text.", fix=_FIX["incomplete"],
                              retryable=True)
    return text


def parse_retry_after(value: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Retry-After as delta-seconds or an HTTP date, in seconds; None when absent or unreadable."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        seconds = None
    if seconds is not None:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def backoff_sec(attempt: int, retry_after: Optional[float], rand: Callable[[], float] = random.random) -> float:
    """Full-jitter exponential backoff; Retry-After is a floor, capped at RETRY_AFTER_CEILING_SEC (http.ts)."""
    ceiling = min(MAX_DELAY_SEC, BASE_DELAY_SEC * 2**attempt)
    jittered = ceiling * rand()
    if retry_after is None:
        return jittered
    return min(max(retry_after, jittered), RETRY_AFTER_CEILING_SEC)


# --------------------------------------------------------------------------------------------------------------------
# Errors

_FIX = {
    "auth": "Check the Meta Model API key (dev.meta.ai, API keys).",
    "billing": "Add a payment method for the Meta Model API at dev.meta.ai/billing.",
    "model_not_found": "This key's team cannot use that model; check the models enabled at dev.meta.ai.",
    "rate_limited": "Wait a minute, then try again.",
    "server": "The Meta Model API had a problem; try again in a few minutes.",
    "timeout": "Try again; if it keeps timing out, try a shorter video.",
    "refused": "The model declined this input; try another video or wording.",
    "incomplete": "The response was cut off; try again.",
    "bad_input": None,
}
_RETRYABLE = {code: code in {"rate_limited", "server", "timeout", "incomplete"} for code in _FIX}
_SECRETISH = re.compile(r"(?i)\bbearer\s+[^\s\"',;]+|\bLLM_[A-Za-z0-9_\-.*]+")
_KEY_CHARS = re.compile(r"^[\x21-\x7e]+$")


def _code_for_status(status: int, error_code: Any = None) -> str:
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "billing"
    if status == 404:
        return "bad_input" if error_code == "file_not_found" else "model_not_found"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server"
    if 400 <= status < 500:
        return "bad_input"
    return "server"  # an unexpected redirect or informational answer


def _code_for_error_code(code: Any, default: str) -> str:
    text = str(code or "").lower()
    if "rate_limit" in text:
        return "rate_limited"
    if "content_policy" in text or "refus" in text:
        return "refused"
    if "billing" in text or "quota" in text:
        return "billing"
    if "invalid" in text or "bad_request" in text:
        return "bad_input"
    if "timeout" in text:
        return "timeout"
    return default


@dataclass
class _Failure:
    code: str
    message: str
    retry: bool = False
    retry_after: Optional[float] = None


# --------------------------------------------------------------------------------------------------------------------
# Client


class MetaClient:
    """One Meta Model API key. Methods raise ScreentimeError with the codes in errors.py."""

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        transport: Optional[httpx.BaseTransport] = None,
        *,
        max_tries: int = MAX_TRIES,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._key = _checked_key(api_key)
        self._max_tries = max(1, int(max_tries))
        self._sleep = sleep
        self._rand = rand
        self._http = httpx.Client(
            base_url=base_url,
            transport=transport,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(SPARK_TIMEOUT_SEC, connect=CONNECT_TIMEOUT_SEC),
        )

    def __repr__(self) -> str:
        return f"MetaClient(base_url={str(self._http.base_url)!r})"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MetaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- Files ----------

    def upload_file(self, path: str | Path, *, expires_after_sec: Optional[int] = 3600,
                    timeout: float = UPLOAD_TIMEOUT_SEC) -> str:
        """Uploads a file with purpose=user_data and returns its file id. purpose and expires_after go before the file
        part, as the docs advise. Not billed, so a dropped connection is tried again."""
        p = Path(path)
        data = _read_file(p)
        if len(data) > UPLOAD_MAX_BYTES:
            raise ScreentimeError("too_long", f"{p.name} is larger than the 1 GiB upload limit.")
        fields = {"purpose": "user_data"}
        if expires_after_sec is not None:
            low, high = UPLOAD_EXPIRY_RANGE_SEC
            if not (low <= int(expires_after_sec) <= high):
                raise ValueError(f"expires_after_sec must be between {low} and {high}")
            fields["expires_after[anchor]"] = "created_at"
            fields["expires_after[seconds]"] = str(int(expires_after_sec))
        mime = _MIME_BY_SUFFIX.get(p.suffix.lower(), "application/octet-stream")
        response, _ = self._send(
            "POST", "/files", billed=False, timeout=timeout, headers={"Accept": "application/json"},
            body=lambda: {"data": fields, "files": {"file": (p.name, data, mime)}},
        )
        payload = self._json(response)
        file_id = payload.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ScreentimeError("server", "The Files API answered without a file id.", fix=_FIX["server"])
        return file_id

    def delete_file(self, file_id: str) -> bool:
        """Deletes an uploaded file. Best effort: True when the API confirmed it, False on any failure."""
        if not isinstance(file_id, str) or not file_id:
            return False
        try:
            response, _ = self._send(
                "DELETE", f"/files/{quote(file_id, safe='')}", billed=False, timeout=DELETE_TIMEOUT_SEC,
                headers={"Accept": "application/json"}, body=dict,
            )
        except ScreentimeError:
            return False
        try:
            payload = response.json()
        except ValueError:
            return True
        return not (isinstance(payload, Mapping) and payload.get("deleted") is False)

    # ---------- SAM 3.1 video ----------

    def sam_track(
        self,
        prompt: str,
        *,
        file_id: str,
        include_confidence: bool = True,
        mask_encoding: Optional[str] = "one_bit",
        frame_count: int,
        timeout: Optional[float] = None,
    ) -> SamResult:
        """Tracks every object matching the noun phrase `prompt` in an uploaded video (a streamed sam-3.1 Responses
        call). Error, refusal and incomplete events, and a stream that ends without response.completed, raise."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ScreentimeError("bad_input", "The SAM prompt is empty.")
        if not isinstance(file_id, str) or not file_id:
            raise ScreentimeError("bad_input", "No uploaded file id for SAM.")
        if mask_encoding not in (None, "one_bit", "lossless"):
            raise ValueError("mask_encoding must be 'one_bit', 'lossless' or None")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("frame_count must be a positive int")
        limit = float(timeout) if timeout is not None else sam_timeout_sec(frame_count)
        body = build_sam_body(prompt.strip(), file_id, include_confidence, mask_encoding)
        t0 = time.monotonic()
        response, started = self._send(
            "POST", "/responses", billed=True, timeout=limit, stream=True,
            headers={"Accept": "text/event-stream"}, body=lambda: {"json": body},
        )
        try:
            text, frames = self._read_sam_stream(response, started + limit, limit)
        finally:
            response.close()
        billed = frames if frames is not None else frame_count
        return SamResult(text=text, frames_billed=billed, ms=_ms_since(t0), usd=sam_usd(billed),
                         frames_estimated=frames is None)

    def _read_sam_stream(self, response: httpx.Response, deadline: float, limit: float) -> tuple[str, Optional[int]]:
        parts: list[str] = []
        final_text: Optional[str] = None
        refusal: list[str] = []
        frames: Optional[int] = None
        completed = False
        failure: Optional[_Failure] = None
        with _Watchdog(response, deadline) as dog:
            try:
                for event in iter_sse_events(_lines_until(response, deadline)):
                    kind = event["type"]
                    if kind == "response.output_text.delta" and isinstance(event.get("delta"), str):
                        parts.append(event["delta"])
                    elif kind == "response.output_text.done" and isinstance(event.get("text"), str):
                        final_text = event["text"]
                    elif kind == "response.content_part.done":
                        part = event.get("part")
                        if isinstance(part, Mapping) and part.get("type") == "output_text":
                            final_text = part.get("text") if isinstance(part.get("text"), str) else final_text
                    elif kind == "response.refusal.delta" and isinstance(event.get("delta"), str):
                        refusal.append(event["delta"])
                    elif kind == "response.refusal.done":
                        failure = _Failure("refused", f"SAM refused: {event.get('refusal') or ''}".strip())
                        break
                    elif kind == "error":
                        code = _code_for_error_code(event.get("code"), "server")
                        failure = _Failure(code, f"SAM stream error: {event.get('message') or 'unknown'}")
                        break
                    elif kind == "response.failed":
                        err = _nested(event, "response", "error")
                        code = _code_for_error_code(err.get("code"), "server")
                        failure = _Failure(code, f"SAM response failed: {err.get('message') or 'unknown'}")
                        break
                    elif kind == "response.incomplete":
                        why = _nested(event, "response", "incomplete_details").get("reason") or "unknown"
                        failure = _Failure("incomplete", f"SAM response incomplete: {why}")
                        break
                    elif kind == "response.completed":
                        completed = True
                        usage = _nested(event, "response", "usage")
                        if _is_count(usage.get("video_frames_processed")):
                            frames = usage["video_frames_processed"]
                        break
            except _DeadlinePassed:
                failure = _Failure("timeout", f"SAM did not finish within {limit:.0f} s.")
            except httpx.TimeoutException:
                failure = _Failure("timeout", f"SAM did not finish within {limit:.0f} s.")
            except httpx.HTTPError as err:
                if dog.fired.is_set() or time.monotonic() >= deadline:
                    failure = _Failure("timeout", f"SAM did not finish within {limit:.0f} s.")
                else:
                    failure = _Failure("incomplete", f"The SAM stream was cut off ({type(err).__name__}).")
        if failure is None and refusal:
            failure = _Failure("refused", f"SAM refused: {''.join(refusal)}")
        if failure is None and not completed:
            failure = _Failure("incomplete", "The SAM stream ended without response.completed (truncated).")
        if failure is not None:
            raise self._error(failure)
        return ("".join(parts) if parts else (final_text or "")), frames

    # ---------- Muse Spark structured output ----------

    def spark_json(
        self,
        *,
        model: str = SPARK_MODEL,
        system: Optional[str] = None,
        parts: Sequence[Any],
        json_schema: Mapping[str, Any],
        schema_name: str,
        reasoning_effort: Optional[str] = "low",
        timeout: float = SPARK_TIMEOUT_SEC,
        strict: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
    ) -> SparkResult:
        """One Muse Spark Responses call whose output is constrained to `json_schema`.

        parts: {"type": "text", "text": str} and {"type": "image", "jpeg": bytes} (sent as a data:image/jpeg input_image);
        plain str and bytes work too, and parts already in API form (input_text / input_image) pass through."""
        if "contributor" in str(model).lower():
            raise ScreentimeError("bad_input", f"{model} is a Contributor-mode model (Meta may train on its inputs); "
                                  "Screen-Time uses Standard models only.")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
        if not schema_name:
            raise ValueError("schema_name is required")
        content = [_spark_part(p) for p in parts]
        if not content:
            raise ScreentimeError("bad_input", "Muse Spark needs at least one input part.")
        images = sum(1 for c in content if c.get("type") == "input_image")
        if images > MAX_SPARK_IMAGES:
            raise ScreentimeError("bad_input", f"{images} images in one Muse Spark call; the limit is {MAX_SPARK_IMAGES}.")
        body = build_spark_body(content, json_schema, schema_name, model=model, system=system,
                                reasoning_effort=reasoning_effort, strict=strict, max_output_tokens=max_output_tokens)
        t0 = time.monotonic()
        response, _ = self._send(
            "POST", "/responses", billed=True, timeout=timeout, headers={"Accept": "application/json"},
            body=lambda: {"json": body},
        )
        payload = self._json(response)
        text = extract_output_text(payload)
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = _MALFORMED
        if parsed is _MALFORMED:
            raise self._error(_Failure("incomplete", f"Muse Spark output was not valid JSON: {text[:200]}"))
        if not isinstance(parsed, dict):
            raise self._error(_Failure("server", "Muse Spark returned JSON that is not an object."))
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        usd = spark_usd(usage)
        return SparkResult(parsed=parsed, text=text, ms=_ms_since(t0),
                           usd=usd if usd is not None else SPARK_RESERVE_USD, usage=usage,
                           usd_estimated=usd is None, model=model)

    # ---------- Voice transcribe ----------

    def transcribe(self, wav_path: str | Path, *, keywords: Sequence[str] = (),
                   timeout: Optional[float] = None) -> Transcript:
        """Turn-level transcript of a mono 16-bit PCM WAV (16 or 24 kHz, at most 10 minutes), mode ENDPOINTING.
        `keywords` bias recognition toward words it would otherwise mishear, such as a brand name."""
        p = Path(wav_path)
        audio_sec, code, problem = _wav_check(p)
        if code is not None:
            raise ScreentimeError(code, problem or f"{p.name} cannot be transcribed.")
        audio = _read_file(p)
        if len(audio) > TRANSCRIBE_MAX_BYTES:
            raise ScreentimeError("too_long", "The audio is larger than the 32 MB transcription limit.")
        request: dict[str, Any] = {"model": TRANSCRIBE_MODEL, "audioEncoding": "WAV", "mode": "ENDPOINTING"}
        words: list[str] = []
        for word in keywords:
            if isinstance(word, str) and word.strip() and word.strip() not in words:
                words.append(word.strip())
        if words:
            request["keywords"] = words
        request_part = json.dumps(request).encode("utf-8")
        limit = float(timeout) if timeout is not None else transcribe_timeout_sec(audio_sec)
        t0 = time.monotonic()
        response, _ = self._send(
            "POST", "/asr/transcribe", billed=True, timeout=limit, headers={"Accept": "application/json"},
            body=lambda: {"files": [("request", (None, request_part, "application/json")),
                                    ("audio", ("audio.wav", audio, "audio/wav"))]},
        )
        payload = self._json(response)
        turns = [
            Turn(start_sec=_ms_value(t.get("startMs")) / 1000, end_sec=_ms_value(t.get("endMs")) / 1000,
                 text=str(t.get("transcript") or ""))
            for t in payload.get("turns") or []
            if isinstance(t, Mapping)
        ]
        duration_ms = payload.get("audioDurationMs")
        billed_sec = duration_ms / 1000 if isinstance(duration_ms, (int, float)) and duration_ms >= 0 else audio_sec
        return Transcript(turns, text=str(payload.get("transcript") or ""), audio_sec=billed_sec, ms=_ms_since(t0),
                          usd=transcribe_usd(billed_sec))

    # ---------- Transport ----------

    def _send(
        self,
        method: str,
        path: str,
        *,
        billed: bool,
        timeout: float,
        body: Callable[[], dict[str, Any]],
        headers: Optional[Mapping[str, str]] = None,
        stream: bool = False,
    ) -> tuple[httpx.Response, float]:
        """One API call with retries; returns the 2xx response and when its attempt started (monotonic). `body` is
        called per attempt because multipart bodies cannot be reused."""
        for attempt in range(self._max_tries):
            started = time.monotonic()
            request = self._http.build_request(
                method, path,
                headers={**(headers or {}), "Authorization": f"Bearer {self._key}"},
                timeout=httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT_SEC, timeout)),
                **body(),
            )
            failure: Optional[_Failure] = None
            response: Optional[httpx.Response] = None
            try:
                response = self._http.send(request, stream=stream)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as err:
                # The request never reached the server, so even a billed call is safe to send again.
                failure = _Failure("server", f"Could not connect to the Meta Model API ({type(err).__name__}).",
                                   retry=True)
            except httpx.TimeoutException:
                failure = _Failure("timeout", f"The Meta Model API did not answer within {timeout:.0f} s.",
                                   retry=not billed)
            except httpx.TransportError as err:
                failure = _Failure("server", f"The connection to the Meta Model API failed ({type(err).__name__}).",
                                   retry=not billed)
            if response is not None:
                if response.is_success:
                    return response, started
                failure = self._status_failure(response)
            assert failure is not None
            last = attempt + 1 >= self._max_tries
            too_long = failure.retry_after is not None and failure.retry_after > RETRY_AFTER_CEILING_SEC
            if not failure.retry or last or too_long:
                raise self._error(failure)
            self._sleep(backoff_sec(attempt, failure.retry_after, self._rand))
        raise AssertionError("unreachable")

    def _status_failure(self, response: httpx.Response) -> _Failure:
        status = response.status_code
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        text = ""
        try:
            response.read()
            text = response.content.decode("utf-8", "replace")
        except httpx.HTTPError:
            text = ""
        finally:
            response.close()
        envelope: Mapping[str, Any] = {}
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping) and isinstance(parsed.get("error"), Mapping):
            envelope = parsed["error"]
        message = envelope.get("message") or text.strip() or response.reason_phrase or "request failed"
        code = _code_for_status(status, envelope.get("code"))
        return _Failure(code, f"Meta API {status}: {message}", retry=status == 429 or status >= 500,
                        retry_after=retry_after)

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        finally:
            response.close()
        if not isinstance(payload, dict):
            raise self._error(_Failure("server", "The Meta Model API sent an answer that is not a JSON object."))
        return payload

    def _error(self, failure: _Failure) -> ScreentimeError:
        return ScreentimeError(failure.code, self._scrub(failure.message), fix=_FIX.get(failure.code),
                               retryable=_RETRYABLE.get(failure.code, False))

    def _scrub(self, text: str) -> str:
        text = text.replace(self._key, "[redacted]")
        text = _SECRETISH.sub("[redacted]", text)
        return text if len(text) <= 600 else text[:600] + "..."


# --------------------------------------------------------------------------------------------------------------------
# Helpers

_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class _DeadlinePassed(Exception):
    pass


def _checked_key(api_key: Any) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ScreentimeError("auth", "No Meta Model API key was given.", fix=_FIX["auth"])
    key = api_key.strip()
    if not _KEY_CHARS.match(key):
        # Never echo it: a key with spaces or control characters would also end up in an HTTP library's error text.
        raise ScreentimeError("auth", "The Meta Model API key contains spaces or characters a key cannot have.",
                              fix=_FIX["auth"])
    return key


def _lines_until(response: httpx.Response, deadline: float) -> Iterator[str]:
    for line in response.iter_lines():
        if time.monotonic() >= deadline:
            raise _DeadlinePassed()
        yield line


class _Watchdog:
    """Shuts the socket down at the deadline so a stream that goes silent mid-way fails on time (the read timeout
    alone restarts with every chunk). Mock transports have no socket; the per-line deadline check covers them."""

    def __init__(self, response: httpx.Response, deadline: float) -> None:
        self.fired = threading.Event()
        self._timer: Optional[threading.Timer] = None
        stream = response.extensions.get("network_stream")
        sock = None
        if stream is not None:
            try:
                sock = stream.get_extra_info("socket")
            except Exception:  # noqa: BLE001 - any transport without a socket
                sock = None
        if isinstance(sock, socket.socket):
            def fire() -> None:
                self.fired.set()
                try:
                    socket.socket.shutdown(sock, socket.SHUT_RDWR)  # base method: wakes the blocked read, TLS too
                except (OSError, ValueError):
                    pass

            self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), fire)
            self._timer.daemon = True
            self._timer.start()

    def __enter__(self) -> _Watchdog:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._timer is not None:
            self._timer.cancel()


def _spark_part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "input_text", "text": part}
    if isinstance(part, (bytes, bytearray, memoryview)):
        return _image_part(bytes(part))
    if isinstance(part, Mapping):
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            return {"type": "input_text", "text": part["text"]}
        if kind == "image" and isinstance(part.get("jpeg"), (bytes, bytearray, memoryview)):
            return _image_part(bytes(part["jpeg"]))
        if kind == "image" and isinstance(part.get("url"), str):
            return {"type": "input_image", "image_url": part["url"]}
        if kind == "input_text" and isinstance(part.get("text"), str):
            return {"type": "input_text", "text": part["text"]}
        if kind == "input_image" and (isinstance(part.get("image_url"), str) or isinstance(part.get("file_id"), str)):
            return dict(part)
        raise ScreentimeError("bad_input", f"Unsupported Muse Spark input part of type {kind!r}.")
    raise ScreentimeError("bad_input", f"Unsupported Muse Spark input part: {type(part).__name__}.")


def _image_part(data: bytes) -> dict[str, Any]:
    if data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ScreentimeError("bad_input", "Image bytes for Muse Spark are not a JPEG or PNG.")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}


def _read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as err:
        reason = err.strerror or type(err).__name__
    raise ScreentimeError("bad_input", f"Cannot read {path.name}: {reason}.")


def _wav_check(path: Path) -> tuple[float, Optional[str], Optional[str]]:
    """(duration, error code, problem) for a WAV the transcribe endpoint accepts: mono 16-bit PCM at 16 or 24 kHz, at
    most 10 minutes. The code is None when the file is fine."""
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    except (wave.Error, EOFError, OSError) as err:
        reason = str(err)
    else:
        reason = None
    if reason is not None:
        return 0.0, "bad_input", f"Not a readable WAV file: {path.name} ({reason})"
    if channels != 1 or width != 2 or rate not in (16000, 24000):
        return 0.0, "bad_input", (f"{path.name} must be mono 16-bit PCM at 16 or 24 kHz "
                                  f"(got {channels} ch, {8 * width}-bit, {rate} Hz).")
    duration = frames / float(rate)
    if frames == 0:
        return 0.0, "no_audio", f"{path.name} has no audio samples."
    if duration > TRANSCRIBE_MAX_SEC:
        return duration, "too_long", f"{path.name} is {duration:.0f} s long; transcription takes at most 10 minutes."
    return duration, None, None


def _nested(event: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = event
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _ms_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _ms_since(t0: float) -> int:
    return int(round((time.monotonic() - t0) * 1000))
