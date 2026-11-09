"""Tracking with the user's own vision model, and listening with Hermes speech-to-text (no network: fakes)."""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import pytest

from screentime_core import media
from screentime_core.errors import ScreentimeError
from screentime_core.models import VideoInfo
from screentime_core.pipeline import Engines, MeasureRequest, dedupe_mentions, listen_in_windows, run_measure
from screentime_core.verify import ReplayVerifier
from screentime_core.vision_track import DETECT_SCHEMA, VisionTracker, _Box, link, parse_boxes, to_detections

DEMO = Path(__file__).resolve().parents[1] / "fixtures" / "demo"


def test_parse_boxes_scales_sorts_and_drops_bad_entries():
    parsed = {"frames": [
        {"frame": 1, "objects": [{"x1": 500, "y1": 600, "x2": 100, "y2": 200}]},  # corners swapped
        {"frame": 2, "objects": [{"x1": 0, "y1": 0, "x2": 1000, "y2": 1000}]},  # whole frame: dropped
        {"frame": 3, "objects": [{"x1": "a"}, "junk", {"x1": 10, "y1": 10, "x2": 10, "y2": 10}]},
        {"frame": 9, "objects": [{"x1": 1, "y1": 1, "x2": 100, "y2": 100}]},  # out of range
        {"frame": True, "objects": []},
    ]}
    out = parse_boxes(parsed, 3)
    assert len(out) == 3
    b = out[0][0]
    assert (b.x1, b.y1, b.x2, b.y2) == (0.1, 0.2, 0.5, 0.6)
    assert out[1] == [] and out[2] == []
    assert parse_boxes(None, 2) == [[], []]


def test_link_keeps_identity_across_short_gaps_and_splits_far_boxes():
    a = _Box(0.1, 0.1, 0.3, 0.3)
    a2 = _Box(0.12, 0.1, 0.32, 0.3)
    far = _Box(0.7, 0.7, 0.9, 0.9)
    tracks = link([[a], [a2, far], [], [], [a2], [], [], [], [], [a]])
    first = [s for s, _ in tracks[0]]
    assert first == [0, 1, 4]  # gap of 2 samples keeps the identity
    assert any([s for s, _ in t] == [1] for t in tracks)  # the far box is its own object
    assert any([s for s, _ in t] == [9] for t in tracks)  # a gap of 4 starts a new object


def test_to_detections_covers_the_frames_around_each_sample():
    info = VideoInfo(100, 200, 12, 0.8)
    dets = to_detections([[(0, _Box(0.1, 0.1, 0.5, 0.5)), (2, _Box(0.1, 0.1, 0.5, 0.5))]], [2, 7, 12], 5, 2, info)
    assert [d.frame for d in dets] == [0, 1, 2, 3, 4, 10, 11]  # clipped at the last frame
    d = dets[0]
    assert d.object_id == 0 and abs(d.area_fraction - 0.16) < 1e-9
    assert (d.box.x1, d.box.y1) == (10.0, 20.0) and d.box.x2 == 49.0


class FakeLlm:
    """Answers every frame with one box in the middle; counts calls (thread-safe)."""

    def __init__(self, fail_first: int = 0, raises: bool = False):
        self.calls = 0
        self.fail_first = fail_first
        self.raises = raises
        self.lock = threading.Lock()

    def __call__(self, instructions, images, schema, name):
        assert schema is DETECT_SCHEMA and name == "frame_boxes"
        assert all(img[:2] == b"\xff\xd8" and mime == "image/jpeg" for img, mime in images)
        with self.lock:
            self.calls += 1
            n = self.calls
        if self.raises:
            raise RuntimeError("no vision")
        if n <= self.fail_first:
            return {"parsed": None}
        frames = [{"frame": i + 1, "objects": [{"x1": 300, "y1": 300, "x2": 600, "y2": 700}]} for i in range(len(images))]
        return {"parsed": {"frames": frames}, "provider": "fake", "model": "eyes-1", "usd": 0.001}


def test_vision_tracker_on_the_demo_clip(tmp_path):
    info = media.probe(str(DEMO / "sample.mp4"))
    llm = FakeLlm()
    res = VisionTracker(llm, samples_per_sec=3).track(str(DEMO / "sample.mp4"), info, "drink can", tmp_path)
    assert llm.calls == 8  # 30 frames sampled, 4 per call
    assert {d.object_id for d in res.detections} == {0}
    assert len(res.detections) == info.frame_count  # every frame covered by one sample
    assert res.label == "hermes:fake/eyes-1"
    assert res.call.usd == pytest.approx(0.008)
    assert "3 frames a second" in res.notes[0]


def test_vision_tracker_reports_unread_frames_and_total_failure(tmp_path):
    info = media.probe(str(DEMO / "sample.mp4"))
    res = VisionTracker(FakeLlm(fail_first=2), 3).track(str(DEMO / "sample.mp4"), info, "can", tmp_path)
    assert any("got no readable answer" in n for n in res.notes)
    with pytest.raises(ScreentimeError) as err:
        VisionTracker(FakeLlm(raises=True), 3).track(str(DEMO / "sample.mp4"), info, "can", tmp_path / "b")
    assert err.value.code == "engine_unavailable"


def _wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def test_listen_in_windows_overlaps_and_times_each_window(tmp_path):
    seen = []

    def stt(path):
        with wave.open(path, "rb") as w:
            seen.append(round(w.getnframes() / w.getframerate(), 2))
        return {"success": True, "transcript": "this is an ad for La Croix" if len(seen) <= 2 else ""}

    turns, why = listen_in_windows(stt, _wav(tmp_path / "a.wav", 13.0), tmp_path / "w")
    assert why == ""
    assert seen == [5.0, 5.0, 5.0]  # windows at 0, 4 and 8 s; the last one reaches the end
    assert [(t["start_sec"], t["end_sec"]) for t in turns] == [(0.0, 5.0), (4.0, 9.0)]


def test_listen_in_windows_reports_a_dead_stt(tmp_path):
    turns, why = listen_in_windows(lambda p: {"success": False, "error": "STT is disabled"}, _wav(tmp_path / "a.wav", 6),
                                   tmp_path / "w")
    assert turns is None and "STT is disabled" in why


def test_dedupe_mentions_drops_the_overlap_echo():
    from screentime_core.models import Mention

    ms = [Mention(4.5, 4.9, "La Croix"), Mention(4.4, 4.8, "La Croix"), Mention(8.0, 8.4, "La Croix")]
    assert [m.start_sec for m in dedupe_mentions(ms)] == [4.4, 8.0]


def test_measure_end_to_end_with_hermes_models_only(tmp_path):
    """No Meta client anywhere: the user's vision model tracks, a recorded verdict checks, Hermes STT listens."""
    src = DEMO / "sample.mp4"

    def stt(path):
        return {"success": True, "transcript": "La Croix, this is an ad"}

    eng = Engines("vision", ReplayVerifier({0: {"verdict": "match", "visibility": "clear", "reason": "pink can"}},
                                           source_model="fake-check"),
                  vision=VisionTracker(FakeLlm(), 3), stt=stt, stt_label="hermes-stt:fake")
    from screentime_core.models import ContractTerm

    req = MeasureRequest(str(src), "pink LaCroix can", "drink can", brand_names=["LaCroix"],
                         terms=[ContractTerm("3 s", {"kind": "min_seconds_total", "seconds": 3}),
                                ContractTerm("say it", {"kind": "mentioned", "min_times": 1})])
    # The bundled sample is video-only; give it audio so the listening path runs.
    with_audio = tmp_path / "with_audio.mp4"
    import subprocess

    subprocess.run([media.ffmpeg_bin(), "-v", "error", "-y", "-i", str(src), "-i", str(DEMO / "sample_audio.m4a"),
                    "-map", "0:v", "-map", "1:a", "-c", "copy", "-shortest", str(with_audio)], check=True)
    req.video_path = str(with_audio)
    rep = run_measure(req, eng, workdir=tmp_path / "w", outdir=tmp_path / "o", budget=None, render_proof=False)
    assert rep.engine == "vision"
    assert rep.receipt.models["sam"] == "hermes:fake/eyes-1"
    assert rep.receipt.models["transcribe"] == "hermes-stt:fake"
    assert rep.metrics.seconds_on_screen == pytest.approx(10.0, abs=0.1)
    assert [t.passed for t in rep.terms] == [True, True]
    assert len(rep.metrics.mentions) == 3  # one per window, windows at 0, 4 and 8 s (dedupe keeps hits 4 s apart)
    assert any("not by SAM 3.1" in n for n in rep.notes)


def test_link_follows_a_fast_mover_by_nearness_but_not_a_far_or_odd_sized_box():
    a = _Box(0.10, 0.10, 0.20, 0.30)
    moved = _Box(0.22, 0.12, 0.32, 0.32)  # no overlap, centre about one box away, same size
    tiny = _Box(0.21, 0.11, 0.22, 0.12)  # near but a fraction of the size
    far = _Box(0.60, 0.60, 0.70, 0.80)
    assert [len(t) for t in link([[a], [moved]])] == [2]
    assert [len(t) for t in link([[a], [tiny]])] == [1, 1]
    assert [len(t) for t in link([[a], [far]])] == [1, 1]
