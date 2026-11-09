# Module contract (build-time, for the coding agents)

The plugin ports the Screen-Time web app (TypeScript, ~/Projects/aetrai/screen-time, READ-ONLY for this build) to a
Hermes Agent plugin in Python. `screentime_core/` is a plain Python package with **no Hermes imports**, so pytest runs
without Hermes. The Hermes glue (plugin.yaml, __init__.py, schemas.py, tools.py, commands.py, cli.py, skills/) sits at
the repository root and is owned by the lead (Claude, main session).

Shared types live in `screentime_core/models.py`. Do not change existing fields there; if you need a new field, add it
with a default and say so in your final report.

## Rules for every module
- Python 3.11+ (Hermes runs 3.14). Standard library plus: httpx, Pillow, meta-sam-parser. No numpy, OpenCV, torch,
  scipy or anything else without asking the lead.
- ffmpeg/ffprobe are external executables found on PATH (or the path in `media.ffmpeg_bin()`); call them only with argv
  lists via `subprocess.run(..., check=False)`; never `shell=True`; never interpolate user text into a shell.
- Never read API keys from os.environ, .env or files inside screentime_core. The Meta key is passed in as an argument.
- Never print, log or store an API key or an Authorization header. Receipts store model, purpose, short input/output
  summaries, ms and usd only.
- Functions return values; they do not print. Raise `screentime_core.errors.ScreentimeError(code, message, fix=...)`
  for expected failures (codes below).
- Every number that reaches a user must be measured or computed, never invented. Unknown is `None`, never 0.
- Tests: pytest, no network, no API key. Live calls only in scripts under `dev/` run by hand.

## Error codes (screentime_core/errors.py)
ffmpeg_missing, bad_input, too_long, no_audio, engine_unavailable, auth (401/403), billing (402), model_not_found (404),
rate_limited (429), server (5xx), timeout, refused, incomplete, budget, verifier_unavailable.

## Modules and owners
| Module | Owner | Public functions |
|---|---|---|
| models.py, errors.py | lead | dataclasses, to_dict, from_dict, ScreentimeError |
| media.py | agent MEDIA+META | ffmpeg_bin(), ffprobe_bin(), probe(path) -> VideoInfo, normalize(src, dst, fps=15, max_height=None) -> VideoInfo, extract_audio_wav(src, dst, rate=16000) -> bool, frame_jpeg(video, frame, fps=15, max_side=None) -> bytes, crop_jpeg(jpeg_bytes, box, pad=0.15) -> bytes, sha256_file(path) -> str, encoders() -> dict |
| meta_client.py | agent MEDIA+META | class MetaClient(api_key, base_url="https://api.meta.ai/v1", transport=None): upload_file, delete_file, sam_track(prompt, file_id) -> SamResult, spark_json(...) -> SparkResult, transcribe(wav_path, keywords) -> list[Turn]; prices; error mapping |
| sam.py | agent MEDIA+META | parse_sam_text(text, width, height) -> list[Detection]; group_tracks(detections) -> dict[int, list[Detection]] |
| metrics.py, rules.py, mentions.py | agent PORT | ports of src/lib/metrics/{compute,rules,mentions}.ts with parity tests |
| teardown.py, card.py | agent TEARDOWN | analyze(video, workdir, ...) -> TeardownReport; compare(yours, theirs) -> CompareReport; render cards (Pillow) |
| proof.py | agent PROOF | render_proof(video, detections, verdict_by_object, metrics, info, out_mp4, ...) -> str |
| verify.py, budget.py, store.py, pipeline.py | lead | look-alike check, spend caps, run index, Measure orchestration |
| vision_track.py | lead | VisionTracker(llm, samples_per_sec=3).track(video, info, noun, workdir) -> VisionTrackResult: any vision model boxes the noun in sampled frames (no Meta key); media.sample_frames; pipeline.listen_in_windows for Hermes speech-to-text |

## Test data on this machine (read-only, never committed except where a task says so)
- 11 LaCroix test Reels: ~/Projects/aetrai/screen-time/data/videos/lacroix/*.mp4
- Human-checked frame labels: ~/Projects/aetrai/screen-time/data/labels/lacroix/*.json and objects.json
- Stored app runs (reports with detections): ~/Projects/aetrai/screen-time/data/runs/, data/out/
- Builder's own clips (OK to commit cut-down copies under fixtures/demo/ only when a task says so):
  lc-12-aldo.mp4 (Poppi look-alike + LaCroix product), lc-10-aldo.mp4 (Pamplemousse look-alike only).
