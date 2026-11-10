"""meta_client.py with httpx.MockTransport: request shapes, stream handling, retries, error codes and key hygiene.

Ports the Screen-Time app's src/lib/meta/client.test.ts and http.test.ts. No network.
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import time
import traceback
import wave
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PIL import Image

from screentime_core import meta_client as mc
from screentime_core.errors import ScreentimeError
from screentime_core.meta_client import MetaClient, Transcript, Turn

KEY = "LLM_test0123456789abcdefTESTKEY"
FIXTURES = Path(__file__).parent / "fixtures" / "meta"
DEMO = Path(__file__).parent.parent / "fixtures" / "demo"

# One frame of SAM 3.1 output, from the meta-sam-parser README.
FRAME0 = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>\n"
)
LANE = {"item_id": "msg-1", "output_index": 0, "content_index": 0}


class Api:
    """A MockTransport handler that answers from a script and records every request."""

    def __init__(self, *answers: httpx.Response | Callable[[httpx.Request], httpx.Response] | Exception):
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if len(self.requests) > len(self.answers):
            raise AssertionError(f"unexpected request #{len(self.requests)}")
        answer = self.answers[len(self.requests) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer(request) if callable(answer) else answer


def make(api: Api, sleeps: list[float] | None = None, **kw) -> MetaClient:
    record = sleeps if sleeps is not None else []
    return MetaClient(KEY, transport=httpx.MockTransport(api), sleep=record.append, rand=lambda: 0.5, **kw)


def ok_json(body: object, status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers)


def error_json(status: int, code: str, message: str | None = None, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message or f"{code} happened", "type": "x",
                                                  "param": None, "code": code}}, headers=headers)


def sse(*events: dict, done: bool = False) -> bytes:
    text = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return (text + ("data: [DONE]\n\n" if done else "")).encode()


def sse_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def sam_events(text: str = FRAME0, frames: int | None = 134) -> bytes:
    usage = {} if frames is None else {"usage": {"video_frames_processed": frames}}
    return sse(
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_text.delta", **LANE, "delta": text[:40]},
        {"type": "response.output_text.delta", **LANE, "delta": text[40:]},
        {"type": "response.content_part.done", **LANE, "part": {"type": "output_text", "text": text}},
        {"type": "response.completed", "response": {"id": "r1", "status": "completed", **usage}},
    )


def spark_body(text: str = '{"verdict":"match"}', usage: dict | None = None, **extra) -> dict:
    body = {"status": "completed",
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}
    if usage is not None:
        body["usage"] = usage
    body.update(extra)
    return body


def jpeg(color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def write_wav(path: Path, seconds: float, rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * int(seconds * rate))
    return path


def multipart_parts(request: httpx.Request) -> list[tuple[str, dict[str, str], bytes]]:
    """(name, headers, body) for each part of a multipart request, in order."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts = []
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.strip(b"\r\n").partition(b"\r\n\r\n")
        headers = {}
        for line in head.decode().split("\r\n"):
            k, _, v = line.partition(": ")
            headers[k.lower()] = v
        name = headers["content-disposition"].split('name="')[1].split('"')[0]
        parts.append((name, headers, body))
    return parts


SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"],
          "additionalProperties": False}


# --------------------------------------------------------------------------------------------------------------------
# Pure helpers


def test_prices_and_time_limits():
    assert mc.sam_usd(10) == 0.01  # the 50-frame minimum
    assert mc.sam_usd(150) == 0.03
    assert mc.sam_usd(326) == 0.0652
    assert mc.spark_usd({"input_tokens": 1000, "output_tokens": 500,
                         "input_tokens_details": {"cached_tokens": 200}}) == pytest.approx(0.003155)
    assert mc.spark_usd({"input_tokens": 1_000_000, "output_tokens": 0}) == 1.25
    assert mc.spark_usd({}) is None and mc.spark_usd(None) is None
    assert mc.transcribe_usd(4.9) == pytest.approx(4 * 0.18 / 3600, abs=1e-6)
    assert mc.transcribe_usd(3600) == 0.18
    assert mc.sam_timeout_sec(150) == pytest.approx(52.5)
    assert mc.transcribe_timeout_sec(4) == 32


def test_parse_retry_after():
    assert mc.parse_retry_after("3") == 3
    now = datetime.datetime(2026, 10, 31, 10, 40, tzinfo=datetime.UTC).timestamp()
    assert mc.parse_retry_after("Sat, 31 Oct 2026 10:40:05 GMT", now=now) == pytest.approx(5)
    assert mc.parse_retry_after(None) is None
    assert mc.parse_retry_after("soon") is None


def test_backoff_grows_with_full_jitter_and_honours_retry_after():
    assert mc.backoff_sec(0, None, lambda: 1) == 0.5
    assert mc.backoff_sec(2, None, lambda: 1) == 2.0
    assert mc.backoff_sec(10, None, lambda: 1) == 30.0
    assert mc.backoff_sec(2, None, lambda: 0.25) == 0.5
    assert mc.backoff_sec(0, 9, lambda: 1) == 9
    assert mc.backoff_sec(0, 3600, lambda: 1) == 60


def test_sse_reader():
    lines = [": keepalive", "", "event: a", 'data: {"type":"a",', 'data: "n":1}', "", "data: [DONE]", ""]
    assert list(mc.iter_sse_events(lines)) == [{"type": "a", "n": 1}]
    # An unterminated, cut-off last event is dropped (the missing terminal event fails the call).
    assert list(mc.iter_sse_events(['data: {"type":"a"}', "", 'data: {"type":"b",'])) == [{"type": "a"}]
    with pytest.raises(ScreentimeError):
        list(mc.iter_sse_events(['data: {"type":', "", 'data: {"type":"b"}', ""]))


# --------------------------------------------------------------------------------------------------------------------
# Files


def test_upload_is_multipart_with_purpose_and_expiry_before_the_file(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake mp4")
    api = Api(ok_json({"id": "file-842549258569145", "object": "file"}))
    assert make(api).upload_file(clip) == "file-842549258569145"
    req = api.requests[0]
    assert (req.method, str(req.url)) == ("POST", "https://api.meta.ai/v1/files")
    assert req.headers["authorization"] == f"Bearer {KEY}"
    parts = multipart_parts(req)
    assert [p[0] for p in parts] == ["purpose", "expires_after[anchor]", "expires_after[seconds]", "file"]
    assert [p[2] for p in parts[:3]] == [b"user_data", b"created_at", b"3600"]
    name, headers, body = parts[3]
    assert 'filename="clip.mp4"' in headers["content-disposition"]
    assert headers["content-type"] == "video/mp4"
    assert body == b"fake mp4"


def test_upload_without_expiry_and_bad_expiry(tmp_path):
    clip = tmp_path / "shot.mov"
    clip.write_bytes(b"movie")
    api = Api(ok_json({"id": "file-2"}))
    make(api).upload_file(clip, expires_after_sec=None)
    parts = multipart_parts(api.requests[0])
    assert [p[0] for p in parts] == ["purpose", "file"]
    assert parts[1][1]["content-type"] == "video/quicktime"
    with pytest.raises(ValueError):
        make(Api()).upload_file(clip, expires_after_sec=60)


def test_upload_of_a_missing_file_is_bad_input(tmp_path):
    api = Api()
    with pytest.raises(ScreentimeError) as err:
        make(api).upload_file(tmp_path / "nope.mp4")
    assert err.value.code == "bad_input"
    assert api.requests == []


def test_upload_answer_without_an_id_is_a_server_error(tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    with pytest.raises(ScreentimeError) as err:
        make(Api(ok_json({"object": "file"}))).upload_file(clip)
    assert err.value.code == "server"


@pytest.mark.parametrize(
    "status,code,expected",
    [(400, "invalid_request", "bad_input"), (401, "invalid_api_key", "auth"), (402, "billing_not_configured", "billing"),
     (403, "forbidden", "auth"), (404, "model_not_found", "model_not_found"), (404, "file_not_found", "bad_input"),
     (413, "payload_too_large", "bad_input")],
)
def test_permanent_errors_are_never_retried(tmp_path, status, code, expected):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    api = Api(error_json(status, code), ok_json({"id": "file-x"}))
    with pytest.raises(ScreentimeError) as err:
        make(api).upload_file(clip)
    assert err.value.code == expected
    assert f"Meta API {status}: {code} happened" in err.value.message
    assert len(api.requests) == 1


def test_404_with_an_empty_body(tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    with pytest.raises(ScreentimeError) as err:
        make(Api(httpx.Response(404, content=b""))).upload_file(clip)
    assert err.value.code == "model_not_found"


@pytest.mark.parametrize("status", [429, 500, 503, 504])
def test_retries_rate_limits_and_server_errors_rebuilding_the_body(tmp_path, status):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    api = Api(error_json(status, "busy"), ok_json({"id": "file-ok"}))
    assert make(api).upload_file(clip) == "file-ok"
    assert len(api.requests) == 2
    assert [p[0] for p in multipart_parts(api.requests[1])][-1] == "file"


def test_retry_after_is_a_floor_on_the_backoff(tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    sleeps: list[float] = []
    api = Api(error_json(429, "rate_limit_exceeded", headers={"retry-after": "7"}),
              error_json(503, "service_overloaded"), ok_json({"id": "file-ok"}))
    make(api, sleeps).upload_file(clip)
    assert sleeps == [7.0, 0.5]  # attempt 0: max(7, 0.5*0.5); attempt 1: 1.0*0.5


def test_gives_up_after_three_tries(tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    api = Api(*[error_json(503, "backend_unavailable")] * 4)
    with pytest.raises(ScreentimeError) as err:
        make(api).upload_file(clip)
    assert err.value.code == "server" and err.value.retryable
    assert len(api.requests) == 3


def test_a_retry_after_beyond_the_ceiling_fails_at_once(tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    sleeps: list[float] = []
    api = Api(error_json(429, "rate_limit_exceeded", "come back tomorrow", headers={"retry-after": "86400"}),
              ok_json({"id": "file-ok"}))
    with pytest.raises(ScreentimeError) as err:
        make(api, sleeps).upload_file(clip)
    assert err.value.code == "rate_limited"
    assert sleeps == [] and len(api.requests) == 1


def test_an_unbilled_upload_is_retried_after_a_dropped_connection(tmp_path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    api = Api(httpx.ReadError("connection reset"), httpx.ReadTimeout("slow"), ok_json({"id": "file-ok"}))
    assert make(api).upload_file(clip) == "file-ok"
    assert len(api.requests) == 3


def test_delete_file():
    api = Api(ok_json({"id": "file-1", "object": "file", "deleted": True}))
    assert make(api).delete_file("file-1") is True
    assert (api.requests[0].method, str(api.requests[0].url)) == ("DELETE", "https://api.meta.ai/v1/files/file-1")
    assert make(Api(error_json(404, "file_not_found"))).delete_file("file-1") is False
    assert make(Api(ok_json({"id": "file-1", "deleted": False}))).delete_file("file-1") is False
    assert make(Api(*[httpx.ConnectError("down")] * 3)).delete_file("file-1") is False
    assert make(Api()).delete_file("") is False


# --------------------------------------------------------------------------------------------------------------------
# SAM 3.1


def test_sam_request_shape_and_result():
    api = Api(sse_response(sam_events()))
    result = make(api).sam_track("red can", file_id="file-abc123", frame_count=134)
    req = api.requests[0]
    assert str(req.url) == "https://api.meta.ai/v1/responses"
    assert req.headers["accept"] == "text/event-stream"
    assert req.headers["content-type"] == "application/json"
    assert json.loads(req.content) == {
        "model": "sam-3.1",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "red can"}, {"type": "input_video", "file_id": "file-abc123"}]}],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        "stream": True,
    }
    assert result.text == FRAME0
    assert result.frames_billed == 134 and result.frames_estimated is False
    assert result.usd == pytest.approx(134 * 0.0002)
    assert isinstance(result.ms, int) and result.ms >= 0
    assert req.extensions["timeout"]["read"] == pytest.approx(mc.sam_timeout_sec(134))


def test_sam_lossless_and_no_confidence():
    api = Api(sse_response(sam_events()))
    make(api).sam_track("can", file_id="f", frame_count=10, include_confidence=False, mask_encoding="lossless")
    assert json.loads(api.requests[0].content)["metadata"] == {"include_confidence": "false",
                                                               "mask_encoding": "lossless"}


def test_sam_without_usage_bills_the_frame_count_and_says_so():
    result = make(Api(sse_response(sam_events(frames=None)))).sam_track("can", file_id="f", frame_count=20)
    assert result.frames_billed == 20 and result.frames_estimated is True
    assert result.usd == 0.01  # 50-frame minimum


def test_sam_zero_matches_is_a_normal_result():
    body = sse({"type": "response.completed", "response": {"usage": {"video_frames_processed": 50}}}, done=True)
    result = make(Api(sse_response(body))).sam_track("unicorn", file_id="f", frame_count=50)
    assert result.text == ""


def test_sam_text_from_the_finalizer_when_no_deltas_arrived():
    body = sse({"type": "response.content_part.done", **LANE, "part": {"type": "output_text", "text": FRAME0}},
               {"type": "response.completed", "response": {"usage": {"video_frames_processed": 50}}})
    assert make(Api(sse_response(body))).sam_track("can", file_id="f", frame_count=50).text == FRAME0


@pytest.mark.parametrize(
    "events,code,pattern",
    [
        ([{"type": "error", "code": "server_error", "message": "boom", "param": None}], "server", "boom"),
        ([{"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}], "rate_limited", "slow down"),
        ([{"type": "response.failed", "response": {"error": {"message": "kaput", "code": "server_error"}}}],
         "server", "kaput"),
        ([{"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}],
         "incomplete", "max_output_tokens"),
        ([{"type": "response.refusal.delta", **LANE, "delta": "I can't help"},
          {"type": "response.refusal.done", **LANE, "refusal": "I can't help with that"},
          {"type": "response.completed", "response": {}}], "refused", "I can't help with that"),
        ([{"type": "response.refusal.delta", **LANE, "delta": "no"},
          {"type": "response.completed", "response": {}}], "refused", "no"),
        ([{"type": "response.output_text.delta", **LANE, "delta": FRAME0}], "incomplete", "truncated"),
        ([{"type": "response.output_text.delta", **LANE, "delta": FRAME0},
          {"type": "error", "message": "mid-stream error"},
          {"type": "response.completed", "response": {}}], "server", "mid-stream error"),
    ],
)
def test_sam_stream_failures_raise_and_are_never_resent(events, code, pattern):
    api = Api(sse_response(sse(*events)), sse_response(sam_events()))
    with pytest.raises(ScreentimeError) as err:
        make(api).sam_track("can", file_id="f", frame_count=50)
    assert err.value.code == code
    assert pattern in err.value.message
    assert len(api.requests) == 1


def test_sam_cut_off_final_event_is_truncation_not_a_json_error():
    body = sse({"type": "response.output_text.delta", **LANE, "delta": FRAME0}) + b'data: {"type":"response.compl'
    with pytest.raises(ScreentimeError) as err:
        make(Api(sse_response(body))).sam_track("can", file_id="f", frame_count=50)
    assert err.value.code == "incomplete" and "truncated" in err.value.message


def test_sam_malformed_event_mid_stream():
    body = b'data: {"type":"response.created"\n\n' + sam_events()
    with pytest.raises(ScreentimeError) as err:
        make(Api(sse_response(body))).sam_track("can", file_id="f", frame_count=50)
    assert err.value.code == "server"


def test_sam_is_not_resent_after_the_request_may_have_reached_the_server():
    for failure in (httpx.ReadError("reset"), httpx.ReadTimeout("slow"), httpx.WriteError("broken pipe"),
                    httpx.RemoteProtocolError("bad")):
        api = Api(failure, sse_response(sam_events()))
        with pytest.raises(ScreentimeError) as err:
            make(api).sam_track("can", file_id="f", frame_count=50)
        assert err.value.code in ("server", "timeout")
        assert len(api.requests) == 1, type(failure).__name__


def test_sam_is_resent_when_the_connection_never_opened_or_the_server_said_busy():
    api = Api(httpx.ConnectError("refused"), error_json(503, "service_overloaded"), sse_response(sam_events()))
    assert make(api).sam_track("can", file_id="f", frame_count=50).text == FRAME0
    assert len(api.requests) == 3


def test_sam_stream_cut_mid_way_is_incomplete():
    def body():
        yield sam_events()[:200]
        raise httpx.ReadError("connection reset")

    api = Api(httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"}))
    with pytest.raises(ScreentimeError) as err:
        make(api).sam_track("can", file_id="f", frame_count=50)
    assert err.value.code == "incomplete"
    assert len(api.requests) == 1


def test_sam_stream_that_only_sends_heartbeats_times_out_after_one_request():
    def heartbeats():
        yield sse({"type": "response.created", "response": {"id": "r1"}})
        for _ in range(200):
            time.sleep(0.02)
            yield b": keepalive\n\n"

    api = Api(httpx.Response(200, content=heartbeats(), headers={"content-type": "text/event-stream"}))
    t0 = time.monotonic()
    with pytest.raises(ScreentimeError) as err:
        make(api).sam_track("can", file_id="f", frame_count=50, timeout=0.3)
    assert err.value.code == "timeout"
    assert time.monotonic() - t0 < 2.0
    assert len(api.requests) == 1


def test_sam_rejects_bad_arguments_before_any_request():
    api = Api()
    client = make(api)
    with pytest.raises(ScreentimeError):
        client.sam_track("  ", file_id="f", frame_count=10)
    with pytest.raises(ScreentimeError):
        client.sam_track("can", file_id="", frame_count=10)
    with pytest.raises(ValueError):
        client.sam_track("can", file_id="f", frame_count=0)
    with pytest.raises(ValueError):
        client.sam_track("can", file_id="f", frame_count=10, mask_encoding="png")
    assert api.requests == []


# --------------------------------------------------------------------------------------------------------------------
# Muse Spark


def test_spark_request_shape_with_the_verify_parts():
    usage = {"input_tokens": 1000, "output_tokens": 500, "input_tokens_details": {"cached_tokens": 200}}
    api = Api(ok_json(spark_body(usage=usage)))
    image = jpeg()
    result = make(api).spark_json(
        model="muse-spark-1.3",
        parts=[{"type": "text", "text": "Is this the brand?"}, {"type": "image", "jpeg": image}],
        json_schema=SCHEMA, schema_name="product_check", reasoning_effort="low",
    )
    req = api.requests[0]
    assert str(req.url) == "https://api.meta.ai/v1/responses"
    assert req.headers["accept"] == "application/json"
    assert json.loads(req.content) == {
        "model": "muse-spark-1.3",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Is this the brand?"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}]}],
        "text": {"format": {"type": "json_schema", "name": "product_check", "schema": SCHEMA}},
        "reasoning": {"effort": "low"},
    }
    assert result.parsed == {"verdict": "match"}
    assert result.text == '{"verdict":"match"}'
    assert result.usage == usage
    assert result.usd == pytest.approx(0.003155) and result.usd_estimated is False
    assert isinstance(result.ms, int)
    assert req.extensions["timeout"]["read"] == 60


def test_spark_optional_fields():
    api = Api(ok_json(spark_body()))
    make(api).spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n", system="Be brief.", strict=True,
                         max_output_tokens=2000, reasoning_effort=None, timeout=45)
    body = json.loads(api.requests[0].content)
    assert body["instructions"] == "Be brief."
    assert body["text"]["format"]["strict"] is True
    assert body["max_output_tokens"] == 2000
    assert "reasoning" not in body
    assert api.requests[0].extensions["timeout"]["read"] == 45


def test_spark_without_usage_reports_the_reserve_as_an_estimate():
    result = make(Api(ok_json(spark_body()))).spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n")
    assert result.usd == mc.SPARK_RESERVE_USD and result.usd_estimated is True
    assert result.usage == {}


@pytest.mark.parametrize(
    "body,code,pattern",
    [
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
         "incomplete", "max_output_tokens"),
        ({"error": {"message": "model overloaded"}, "output": []}, "server", "model overloaded"),
        ({"status": "completed", "output": [{"type": "message", "content": [
            {"type": "refusal", "refusal": "I can't"}]}]}, "refused", "I can't"),
        ({"status": "completed", "output": []}, "incomplete", "no output_text"),
        (spark_body(text='{"verdict": "ma'), "incomplete", "not valid JSON"),
        (spark_body(text="[1, 2]"), "server", "not an object"),
    ],
)
def test_spark_bad_answers(body, code, pattern):
    api = Api(ok_json(body), ok_json(spark_body()))
    with pytest.raises(ScreentimeError) as err:
        make(api).spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n")
    assert err.value.code == code
    assert pattern in err.value.message
    assert len(api.requests) == 1  # a 200 is never sent again


def test_spark_is_resent_on_503_but_not_after_a_dropped_connection():
    api = Api(error_json(503, "busy"), ok_json(spark_body()))
    assert make(api).spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n").parsed == {"verdict": "match"}
    assert len(api.requests) == 2
    for failure, code in ((httpx.ReadError("reset"), "server"), (httpx.ReadTimeout("slow"), "timeout")):
        api = Api(failure, ok_json(spark_body()))
        with pytest.raises(ScreentimeError) as err:
            make(api).spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n")
        assert err.value.code == code
        assert len(api.requests) == 1


def test_spark_refuses_contributor_models_and_bad_parts_before_any_request():
    api = Api()
    client = make(api)
    with pytest.raises(ScreentimeError) as err:
        client.spark_json(model="muse-spark-1.3-contributor", parts=["hi"], json_schema=SCHEMA, schema_name="n")
    assert err.value.code == "bad_input" and "Contributor" in err.value.message
    with pytest.raises(ScreentimeError):
        client.spark_json(parts=[{"type": "image", "jpeg": b"GIF89a"}], json_schema=SCHEMA, schema_name="n")
    with pytest.raises(ScreentimeError):
        client.spark_json(parts=[{"type": "image", "jpeg": jpeg()}] * 51, json_schema=SCHEMA, schema_name="n")
    with pytest.raises(ScreentimeError):
        client.spark_json(parts=[], json_schema=SCHEMA, schema_name="n")
    with pytest.raises(ValueError):
        client.spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n", reasoning_effort="none")
    assert api.requests == []


def test_spark_accepts_png_bytes_and_api_shaped_parts():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    api = Api(ok_json(spark_body()))
    make(api).spark_json(parts=[buf.getvalue(), {"type": "input_image", "image_url": "https://example.com/p.jpg"}],
                         json_schema=SCHEMA, schema_name="n")
    content = json.loads(api.requests[0].content)["input"][0]["content"]
    assert content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "input_image", "image_url": "https://example.com/p.jpg"}


# --------------------------------------------------------------------------------------------------------------------
# Voice transcribe


def test_transcribe_request_and_turns(tmp_path):
    wav = write_wav(tmp_path / "a.wav", 4.0)
    api = Api(ok_json({"sessionId": "s1", "transcript": "hello there. try this drink", "audioDurationMs": 4000,
                       "turns": [{"turnId": 0, "startMs": 1520, "endMs": 3100, "transcript": "hello there."},
                                 {"turnId": 1, "startMs": 3200, "endMs": 3900, "transcript": "try this drink"}]}))
    out = make(api).transcribe(wav)
    req = api.requests[0]
    assert str(req.url) == "https://api.meta.ai/v1/asr/transcribe"
    assert req.headers["accept"] == "application/json"
    parts = multipart_parts(req)
    assert [p[0] for p in parts] == ["request", "audio"]
    assert parts[0][1]["content-type"] == "application/json"
    assert json.loads(parts[0][2]) == {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV",
                                       "mode": "ENDPOINTING"}
    assert 'filename="audio.wav"' in parts[1][1]["content-disposition"]
    assert parts[1][1]["content-type"] == "audio/wav"
    assert parts[1][2] == wav.read_bytes()
    assert isinstance(out, Transcript) and isinstance(out, list)
    assert out == [Turn(1.52, 3.1, "hello there."), Turn(3.2, 3.9, "try this drink")]
    assert out.text == "hello there. try this drink"
    assert out.audio_sec == 4.0 and out.usd == mc.transcribe_usd(4.0)
    assert req.extensions["timeout"]["read"] == pytest.approx(mc.transcribe_timeout_sec(4.0))


def test_transcribe_sends_keywords_once_each(tmp_path):
    wav = write_wav(tmp_path / "a.wav", 1.0)
    api = Api(ok_json({"transcript": "it's LaCroix", "turns": []}))
    make(api).transcribe(wav, keywords=["LaCroix", " LaCroix ", "", "La Croix"])
    request = json.loads(multipart_parts(api.requests[0])[0][2])
    assert request["keywords"] == ["LaCroix", "La Croix"]


def test_transcribe_checks_the_wav_before_sending(tmp_path, monkeypatch):
    api = Api()
    client = make(api)
    not_wav = tmp_path / "x.wav"
    not_wav.write_bytes(b"RIFF....nope")
    cases = [(not_wav, "bad_input"),
             (write_wav(tmp_path / "stereo.wav", 1.0, channels=2), "bad_input"),
             (write_wav(tmp_path / "44k.wav", 1.0, rate=44100), "bad_input"),
             (write_wav(tmp_path / "empty.wav", 0.0), "no_audio")]
    monkeypatch.setattr(mc, "TRANSCRIBE_MAX_SEC", 1.0)
    cases.append((write_wav(tmp_path / "long.wav", 2.0), "too_long"))
    for path, code in cases:
        with pytest.raises(ScreentimeError) as err:
            client.transcribe(path)
        assert err.value.code == code, path.name
    assert api.requests == []


def test_transcribe_is_not_resent_after_a_dropped_connection_but_is_on_503(tmp_path):
    wav = write_wav(tmp_path / "a.wav", 1.0)
    api = Api(httpx.ReadError("reset"), ok_json({"transcript": "", "turns": []}))
    with pytest.raises(ScreentimeError):
        make(api).transcribe(wav)
    assert len(api.requests) == 1
    api = Api(error_json(503, "busy"), ok_json({"transcript": "hi", "turns": []}))
    assert make(api).transcribe(wav).text == "hi"
    assert len(api.requests) == 2


# --------------------------------------------------------------------------------------------------------------------
# The key never leaks


def _assert_no_key(err: BaseException) -> None:
    rendered = "".join(traceback.format_exception(err))
    for text in (str(err), repr(err), rendered, getattr(err, "message", ""), getattr(err, "fix", "") or "",
                 json.dumps(getattr(err, "to_dict", dict)())):
        assert KEY not in text
        assert KEY[4:] not in text  # not even without its prefix
    assert err.__cause__ is None and err.__context__ is None  # no chained transport error holding the request


def _leaky(request: httpx.Request) -> httpx.Response:
    # A server that echoes the credentials it received in its error message.
    return error_json(401, "invalid_api_key", f"Incorrect API key provided: {request.headers['authorization']}")


ERROR_PATHS = {
    "401 echoing the key": lambda c, w, f: c.upload_file(f),
    "402": lambda c, w, f: c.upload_file(f),
    "404": lambda c, w, f: c.spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n"),
    "429": lambda c, w, f: c.upload_file(f),
    "500": lambda c, w, f: c.transcribe(w),
    "timeout": lambda c, w, f: c.spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n"),
    "connect": lambda c, w, f: c.sam_track("can", file_id="f", frame_count=10),
    "read error": lambda c, w, f: c.transcribe(w),
    "sam refusal": lambda c, w, f: c.sam_track("can", file_id="f", frame_count=10),
    "sam stream error echoing the key": lambda c, w, f: c.sam_track("can", file_id="f", frame_count=10),
    "sam truncated": lambda c, w, f: c.sam_track("can", file_id="f", frame_count=10),
    "spark refusal": lambda c, w, f: c.spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n"),
    "spark bad json": lambda c, w, f: c.spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n"),
    "not json": lambda c, w, f: c.transcribe(w),
}

ANSWERS = {
    "401 echoing the key": [_leaky],
    "402": [error_json(402, "billing_not_configured")],
    "404": [error_json(404, "model_not_found")],
    "429": [error_json(429, "rate_limit_exceeded", f"Too many requests for key {KEY}")] * 3,
    "500": [error_json(500, "server_error", f"Bearer {KEY} failed")] * 3,
    "timeout": [httpx.ReadTimeout("slow")],
    "connect": [httpx.ConnectError("refused")] * 3,
    "read error": [httpx.ReadError("reset")],
    "sam refusal": [sse_response(sse({"type": "response.refusal.done", **LANE, "refusal": "no"}))],
    "sam stream error echoing the key": [sse_response(sse({"type": "error", "message": f"bad key {KEY}"}))],
    "sam truncated": [sse_response(sse({"type": "response.output_text.delta", **LANE, "delta": FRAME0}))],
    "spark refusal": [ok_json({"status": "completed", "output": [{"type": "message", "content": [
        {"type": "refusal", "refusal": "no"}]}]})],
    "spark bad json": [ok_json(spark_body(text=f'{{"key": "{KEY}'))],
    "not json": [httpx.Response(200, content=b"<html>oops</html>")],
}


@pytest.mark.parametrize("name", list(ERROR_PATHS))
def test_the_key_never_appears_in_an_error(tmp_path, name):
    wav = write_wav(tmp_path / "a.wav", 1.0)
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    api = Api(*ANSWERS[name])
    with pytest.raises(ScreentimeError) as err:
        ERROR_PATHS[name](make(api), wav, clip)
    _assert_no_key(err.value)
    # Positive control: the key really was sent, so its absence above means something.
    assert api.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_a_malformed_key_is_refused_without_echoing_it():
    for bad in ("", "   ", "LLM_has space", "LLM_new\nline", None):
        with pytest.raises(ScreentimeError) as err:
            MetaClient(bad, transport=httpx.MockTransport(Api()))  # type: ignore[arg-type]
        assert err.value.code == "auth"
        if bad and bad.strip():
            assert bad.strip() not in str(err.value)
            assert "space" not in str(err.value).split("contains")[0]  # the key's own text is not quoted
        _assert_no_key(err.value)


def test_a_pasted_key_with_a_trailing_newline_works():
    api = Api(ok_json(spark_body()))
    MetaClient(KEY + "\n", transport=httpx.MockTransport(api)).spark_json(parts=["hi"], json_schema=SCHEMA,
                                                                            schema_name="n")
    assert api.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_key_is_not_in_the_client_repr_or_any_result(tmp_path):
    api = Api(sse_response(sam_events()), ok_json(spark_body(usage={"input_tokens": 1, "output_tokens": 1})),
              ok_json({"transcript": "hi", "turns": [{"startMs": 0, "endMs": 10, "transcript": "hi"}]}))
    client = make(api)
    assert KEY not in repr(client)
    results = [client.sam_track("can", file_id="f", frame_count=10),
               client.spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n"),
               client.transcribe(write_wav(tmp_path / "a.wav", 1.0))]
    for r in results:
        assert KEY not in repr(r) and KEY not in repr(vars(r))


# --------------------------------------------------------------------------------------------------------------------
# Recorded live responses (dev/live_smoke.py), replayed


def _fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not recorded")
    return path


def test_recorded_spark_response_parses():
    recorded = json.loads(_fixture("spark_response.json").read_text())
    api = Api(ok_json(recorded["response"]))
    result = make(api).spark_json(parts=[{"type": "text", "text": recorded["prompt"]}],
                                  json_schema=recorded["schema"], schema_name=recorded["schema_name"])
    assert result.parsed == recorded["parsed"]
    assert result.usd_estimated is False and result.usd == pytest.approx(recorded["usd"])


def test_recorded_transcribe_response_parses(tmp_path):
    recorded = json.loads(_fixture("transcribe_response.json").read_text())
    wav = write_wav(tmp_path / "a.wav", recorded["audio_sec"])
    out = make(Api(ok_json(recorded["response"]))).transcribe(wav, keywords=recorded["keywords"])
    assert [t.text for t in out] == [t["transcript"] for t in recorded["response"]["turns"]]
    assert out.usd == pytest.approx(recorded["usd"])


def test_recorded_upload_and_401_answers():
    upload = json.loads(_fixture("files_upload_response.json").read_text())
    assert upload["id"].startswith("file-")
    denied = json.loads(_fixture("error_401_response.json").read_text())
    api = Api(ok_json(denied["response"], status=401))
    with pytest.raises(ScreentimeError) as err:
        make(api).spark_json(parts=["hi"], json_schema=SCHEMA, schema_name="n")
    assert err.value.code == "auth"


def test_recorded_sam_stream_replays_to_the_recorded_text():
    """The live stream (gzipped, byte for byte): 150 deltas, then output_text.done, content_part.done,
    output_item.done and response.completed, each repeating the full text."""
    stream = _fixture("sam_stream.sse").read_bytes()
    meta = json.loads((DEMO / "sample.sam.json").read_text())
    result = make(Api(sse_response(stream))).sam_track(meta["prompt"], file_id="file-x",
                                                       frame_count=meta["frame_count"])
    assert result.text == (DEMO / "sample.sam.txt").read_text()
    assert result.frames_billed == meta["frames_billed"] and result.frames_estimated is False
    assert result.usd == meta["usd"]
    # Cut the same stream before response.completed: the client must refuse the partial result.
    cut = stream[: stream.index(b"event: response.completed")]
    with pytest.raises(ScreentimeError) as err:
        make(Api(sse_response(cut))).sam_track(meta["prompt"], file_id="file-x", frame_count=meta["frame_count"])
    assert err.value.code == "incomplete"
