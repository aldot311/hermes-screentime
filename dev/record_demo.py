"""Record the keyless demo's remaining fixtures with live Meta calls (hand-run, about $0.01).

    ~/ai/bin/with-secrets.sh .venv/bin/python dev/record_demo.py /path/to/lc-12-aldo.mp4

Writes fixtures/demo/sample_audio.m4a (the clip's own audio, first 10 s), sample.turns.json (Voice Transcribe) and
sample.verdicts.json (Muse Spark's product check of every tracked object, with the product photo).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screentime_core import media, sam  # noqa: E402
from screentime_core.meta_client import MetaClient  # noqa: E402
from screentime_core.models import FPS  # noqa: E402
from screentime_core.verify import MetaSparkVerifier, pick_frames  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "fixtures" / "demo"
PRODUCT = (
    "a pink LaCroix Strawberry Peach sparkling water can: blue 'LaCroix' script over a yellow peach, red and yellow "
    "hearts, strawberries and green stripes near the base"
)
BRANDS = ["LaCroix", "La Croix"]


def main(source: str) -> None:
    key = os.environ.get("META_MODEL_API_KEY", "")
    if not key:
        sys.exit("META_MODEL_API_KEY missing (run through ~/ai/bin/with-secrets.sh)")
    meta = json.loads((DEMO / "sample.sam.json").read_text())
    start, end = meta["source_window_sec"]
    client = MetaClient(key)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    # 1. The clip's own audio for the proof reel (the normalized sample has no audio track by design).
    audio = DEMO / "sample_audio.m4a"
    subprocess.run([media.ffmpeg_bin(), "-v", "error", "-y", "-protocol_whitelist", "file,pipe", "-ss", str(start),
                    "-t", str(end - start), "-i", source, "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "96k", str(audio)],
                   check=True, timeout=60)

    # 2. Transcript of the same window.
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        media.extract_audio_wav(source, str(wav), start_sec=start, duration_sec=end - start)
        tr = client.transcribe(str(wav), keywords=BRANDS)
    turns = [{"start_sec": t.start_sec, "end_sec": t.end_sec, "text": t.text} for t in tr]
    (DEMO / "sample.turns.json").write_text(json.dumps(
        {"recorded_at": now, "model": "muse-voice-transcribe-1.0", "keywords": BRANDS, "usd": getattr(tr, "usd", None),
         "turns": turns}, indent=2))

    # 3. Muse Spark's check of every tracked object, exactly as the live pipeline asks it.
    detections = sam.parse_sam_text((DEMO / "sample.sam.txt").read_text(), meta["width"], meta["height"])
    tracks = sam.group_tracks(detections)
    photo = (DEMO / "product.jpg").read_bytes()
    verifier = MetaSparkVerifier(client)
    verdicts, usd = {}, 0.0
    for oid, dets in sorted(tracks.items()):
        picks = pick_frames(dets, meta["frame_count"])
        crops = [media.crop_jpeg(media.frame_jpeg(str(DEMO / "sample.mp4"), d.frame, fps=FPS), d.box, pad=0.15)
                 for d in picks]
        tv, call = verifier.check(object_id=oid, product=PRODUCT, sam_prompt=meta["prompt"], photo_jpeg=photo,
                                  crops=crops)
        usd += call.usd
        verdicts[str(oid)] = {"verdict": tv.verdict, "visibility": tv.visibility, "reason": tv.reason,
                              "checks": tv.checks}
        print(f"object {oid}: {tv.verdict} ({tv.visibility}) {tv.reason}")
    (DEMO / "sample.verdicts.json").write_text(json.dumps(
        {"recorded_at": now, "model": "muse-spark-1.3", "product": PRODUCT, "brand_names": BRANDS,
         "usd": round(usd, 5), "verdicts": verdicts}, indent=2))
    print("turns:", turns)
    print(f"spark usd {usd:.5f}, transcribe usd {getattr(tr, 'usd', None)}")


if __name__ == "__main__":
    main(sys.argv[1])
