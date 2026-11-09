"""Hand-run live checks against the Meta Model API, and the bundled demo clip. Never run by pytest.

    .venv/bin/python dev/live_smoke.py demo          # no network: fixtures/demo/sample.mp4 from the builder's own Reel
    ~/ai/bin/with-secrets.sh .venv/bin/python dev/live_smoke.py live    # SAM + Spark + transcribe + 401 control
    ~/ai/bin/with-secrets.sh .venv/bin/python dev/live_smoke.py product # fixtures/demo/product.jpg from the SAM run

`live` makes at most four billed calls under a hard cap (SPEND_CAP_USD, checked before every call) and writes:
fixtures/demo/sample.sam.txt and sample.sam.json, tests/fixtures/meta/{sam_stream.sse, spark_response.json,
transcribe_response.json, files_upload_response.json, files_delete_response.json, error_401_response.json}.
Response bodies only; no headers are stored. The key is read from META_MODEL_API_KEY and only its length is printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from screentime_core import media, sam  # noqa: E402
from screentime_core.errors import ScreentimeError  # noqa: E402
from screentime_core.meta_client import (  # noqa: E402
    SAM_MODEL,
    SPARK_MODEL,
    TRANSCRIBE_MODEL,
    MetaClient,
    sam_usd,
    transcribe_usd,
)

SOURCE = Path.home() / "Projects/aetrai/screen-time/data/videos/lacroix/lc-12-aldo.mp4"
DEMO = ROOT / "fixtures" / "demo"
META_FIXTURES = ROOT / "tests" / "fixtures" / "meta"
WINDOW = (0.0, 10.0)  # seconds of the source in the demo clip: both cans on screen, "La Croix" said twice
WAV_WINDOW = (2.0, 4.0)  # start, length: "I'm settling a debate, LaCroix or Poppi"
PROMPT = "drink can"
SPEND_CAP_USD = 0.40
SPARK_ESTIMATE_USD = 0.01  # a one-sentence, one-field call is far below this
FAKE_KEY = "LLM_fake_key_for_negative_control_000000"
SPARK_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
SPARK_PROMPT = "Reply with the single word ok."


# --------------------------------------------------------------------------------------------------------------------
# Spend cap


class Budget:
    def __init__(self, cap: float) -> None:
        self.cap = cap
        self.spent = 0.0

    def check(self, estimate: float, what: str) -> None:
        if self.spent + estimate > self.cap:
            raise SystemExit(f"STOP: {what} (~${estimate:.4f}) would pass the ${self.cap:.2f} cap (spent ${self.spent:.4f})")

    def add(self, usd: float, what: str) -> None:
        self.spent += usd
        print(f"  spent ${usd:.6f} on {what}; total ${self.spent:.6f} of ${self.cap:.2f}")
        if self.spent > self.cap:
            raise SystemExit("STOP: billed above the cap")


# --------------------------------------------------------------------------------------------------------------------
# A transport that keeps each response body (never the headers)


class _Tee(httpx.SyncByteStream):
    def __init__(self, inner: httpx.SyncByteStream, sink: bytearray) -> None:
        self.inner = inner
        self.sink = sink

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.inner:
            self.sink.extend(chunk)
            yield chunk

    def close(self) -> None:
        self.inner.close()


class Recorder(httpx.BaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.HTTPTransport()
        self.records: list[dict] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.inner.handle_request(request)
        body = bytearray()
        self.records.append({"method": request.method, "path": request.url.path, "status": response.status_code,
                             "body": body})
        return httpx.Response(response.status_code, headers=response.headers,
                              stream=_Tee(response.stream, body), extensions=response.extensions)

    def close(self) -> None:
        self.inner.close()

    def last(self, method: str, path_suffix: str) -> dict:
        return next(r for r in reversed(self.records) if r["method"] == method and r["path"].endswith(path_suffix))


def scrub(text: str, *secrets: str) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[redacted]")
    return text


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------------------------------
# Commands


def cmd_demo(_: argparse.Namespace) -> None:
    out = DEMO / "sample.mp4"
    start, end = WINDOW
    info = media.normalize(SOURCE, out, fps=15, max_edge=media.MAX_EDGE, start_sec=start, duration_sec=end - start)
    print(f"sample.mp4: {info.width}x{info.height}, {info.frame_count} frames at {info.fps:g} fps, "
          f"{info.duration_sec:.2f} s, {out.stat().st_size / 1e6:.2f} MB, sha256 {media.sha256_file(out)[:16]}")
    if out.stat().st_size > 4_000_000:
        raise SystemExit("sample.mp4 is over 4 MB; lower the CRF")


def cmd_live(args: argparse.Namespace) -> None:
    key = os.environ.get("META_MODEL_API_KEY", "")
    print(f"META_MODEL_API_KEY length: {len(key)}")
    if not key:
        raise SystemExit("META_MODEL_API_KEY is not set; run through ~/ai/bin/with-secrets.sh")
    budget = Budget(SPEND_CAP_USD)
    sample = DEMO / "sample.mp4"
    info = media.probe(sample)
    recorder = Recorder()
    with MetaClient(key, transport=recorder) as client:
        if "sam" in args.only:
            run_sam(client, recorder, budget, sample, info, key)
        if "spark" in args.only:
            run_spark(client, recorder, budget, key)
        if "transcribe" in args.only:
            run_transcribe(client, recorder, budget, key)
    if "negative" in args.only:
        run_negative(key)
    print(f"done: total billed ${budget.spent:.6f}")


def run_sam(client: MetaClient, recorder: Recorder, budget: Budget, sample: Path, info, key: str) -> None:
    budget.check(sam_usd(info.frame_count), "SAM")
    print(f"upload {sample.name} ({info.frame_count} frames, {info.width}x{info.height})")
    file_id = client.upload_file(sample, expires_after_sec=3600)
    upload = json.loads(recorder.last("POST", "/files")["body"])
    created, expires = upload.get("created_at"), upload.get("expires_at")
    print(f"  file id received; expires_at - created_at = {None if not (created and expires) else expires - created} s")
    write_json(META_FIXTURES / "files_upload_response.json", upload)
    try:
        print(f'SAM "{PROMPT}"')
        result = client.sam_track(PROMPT, file_id=file_id, frame_count=info.frame_count)
    finally:
        deleted = client.delete_file(file_id)
        print(f"  delete_file: {deleted}")
        try:
            write_json(META_FIXTURES / "files_delete_response.json",
                       json.loads(recorder.last("DELETE", file_id)["body"] or b"{}"))
        except (StopIteration, json.JSONDecodeError) as err:
            print(f"  (no delete body recorded: {err})")
    budget.add(result.usd, "SAM")
    stream = bytes(recorder.last("POST", "/responses")["body"])
    (META_FIXTURES / "sam_stream.sse").write_bytes(scrub(stream.decode("utf-8"), key).encode("utf-8"))
    print(f"  wrote tests/fixtures/meta/sam_stream.sse ({len(stream)} bytes)")
    (DEMO / "sample.sam.txt").write_text(result.text)
    write_json(DEMO / "sample.sam.json", {
        "prompt": PROMPT, "frames_billed": result.frames_billed, "frames_estimated": result.frames_estimated,
        "ms": result.ms, "usd": result.usd, "recorded_at": now_iso(), "model": SAM_MODEL,
        "mask_encoding": "one_bit", "include_confidence": True,
        "width": info.width, "height": info.height, "frame_count": info.frame_count, "fps": 15,
        "video": "sample.mp4", "video_sha256": media.sha256_file(sample),
        "source_window_sec": list(WINDOW),
    })
    parsed = sam.parse_sam_output(result.text, info.width, info.height)
    print(f"  {len(parsed.detections)} detections, diagnostics: {parsed.diagnostics or 'none'}")
    for oid, track in sam.group_tracks(parsed.detections).items():
        xs = [d.centroid_x for d in track]
        areas = [d.area_fraction for d in track]
        print(f"  object {oid}: {len(track)} frames, first {track[0].frame}, last {track[-1].frame}, "
              f"mean centroid x {sum(xs) / len(xs):.3f}, mean area {sum(areas) / len(areas):.4f}")


def run_spark(client: MetaClient, recorder: Recorder, budget: Budget, key: str) -> None:
    budget.check(SPARK_ESTIMATE_USD, "Spark")
    print("Spark one-field JSON")
    result = client.spark_json(model=SPARK_MODEL, parts=[{"type": "text", "text": SPARK_PROMPT}],
                               json_schema=SPARK_SCHEMA, schema_name="smoke_check", reasoning_effort="low")
    budget.add(result.usd, "Spark")
    print(f"  parsed {result.parsed}, {result.ms} ms, usage {result.usage}, usd_estimated {result.usd_estimated}")
    raw = json.loads(scrub(bytes(recorder.last("POST", "/responses")["body"]).decode("utf-8"), key))
    write_json(META_FIXTURES / "spark_response.json", {
        "recorded_at": now_iso(), "model": SPARK_MODEL, "prompt": SPARK_PROMPT, "schema_name": "smoke_check",
        "schema": SPARK_SCHEMA, "reasoning_effort": "low", "parsed": result.parsed, "usd": result.usd,
        "ms": result.ms, "response": raw,
    })


def run_transcribe(client: MetaClient, recorder: Recorder, budget: Budget, key: str) -> None:
    start, length = WAV_WINDOW
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "cut.wav"
        if not media.extract_audio_wav(SOURCE, wav, start_sec=start, duration_sec=length):
            raise SystemExit("no audio in the source")
        audio_sec = media.wav_duration_sec(wav)
        budget.check(transcribe_usd(audio_sec) + 0.001, "transcribe")
        print(f"transcribe {audio_sec:.2f} s ({start:g}-{start + length:g} s of {SOURCE.name}), keywords LaCroix")
        out = client.transcribe(wav, keywords=["LaCroix"])
    budget.add(out.usd, "transcribe")
    print(f"  {len(out)} turn(s): {[(t.start_sec, t.end_sec, t.text) for t in out]}; text {out.text!r}; {out.ms} ms")
    raw = json.loads(scrub(bytes(recorder.last("POST", "/asr/transcribe")["body"]).decode("utf-8"), key))
    write_json(META_FIXTURES / "transcribe_response.json", {
        "recorded_at": now_iso(), "model": TRANSCRIBE_MODEL, "source": SOURCE.name,
        "source_window_sec": [start, start + length], "audio_sec": audio_sec, "keywords": ["LaCroix"],
        "usd": out.usd, "ms": out.ms, "response": raw,
    })


def run_negative(key: str) -> None:
    print("negative control: the same Spark request with a fake key")
    recorder = Recorder()
    with MetaClient(FAKE_KEY, transport=recorder) as fake:
        try:
            fake.spark_json(parts=[{"type": "text", "text": SPARK_PROMPT}], json_schema=SPARK_SCHEMA,
                            schema_name="smoke_check")
        except ScreentimeError as err:
            print(f"  raised {err.code}: {err.message}")
            if err.code != "auth":
                raise SystemExit("negative control FAILED: expected auth") from None
            if FAKE_KEY in str(err):
                raise SystemExit("negative control FAILED: the key appears in the error") from None
        else:
            raise SystemExit("negative control FAILED: a fake key was accepted")
    record = recorder.last("POST", "/responses")
    body = scrub(bytes(record["body"]).decode("utf-8"), FAKE_KEY, key)
    try:
        response: object = json.loads(body)
    except json.JSONDecodeError:
        response = body
    write_json(META_FIXTURES / "error_401_response.json",
               {"recorded_at": now_iso(), "status": record["status"], "response": response})


def cmd_product(args: argparse.Namespace) -> None:
    """The LaCroix can from the clearest frame of the recorded SAM run, cut from the 1080x1920 source."""
    meta = json.loads((DEMO / "sample.sam.json").read_text())
    dets = sam.parse_sam_text((DEMO / "sample.sam.txt").read_text(), meta["width"], meta["height"])
    track = sam.group_tracks(dets)[args.object_id]
    best = max(track, key=lambda d: d.area_fraction) if args.frame is None else next(
        d for d in track if d.frame == args.frame)
    src = media.probe(SOURCE)
    scale = src.width / meta["width"]
    start_frame = round(WINDOW[0] * src.fps)
    source_frame = start_frame + round(best.frame * src.fps / 15)
    frame = media.frame_jpeg(SOURCE, source_frame, fps=src.fps)
    box = (best.box.x1 * scale, best.box.y1 * scale, (best.box.x2 + 1) * scale - 1, (best.box.y2 + 1) * scale - 1)
    crop = media.crop_jpeg(frame, box, pad=0.08, quality=88)
    out = DEMO / "product.jpg"
    out.write_bytes(crop)
    print(f"product.jpg: object {args.object_id}, demo frame {best.frame} (source frame {source_frame}), "
          f"box {box}, {len(crop) / 1000:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    live = sub.add_parser("live")
    live.add_argument("--only", nargs="+", default=["sam", "spark", "transcribe", "negative"],
                      choices=["sam", "spark", "transcribe", "negative"])
    live.set_defaults(fn=cmd_live)
    product = sub.add_parser("product")
    product.add_argument("--object-id", type=int, required=True)
    product.add_argument("--frame", type=int)
    product.set_defaults(fn=cmd_product)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
