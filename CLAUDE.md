# hermes-screentime

Screen-Time for Hermes Agent: a native Hermes plugin (id `screentime`) entered in the Nous Research Hermes Agent Plugin
Hackathon (due 2026-11-11 17:00 EST). Measure: a product's seconds on screen in a video, look-alikes rejected, plain-
English terms checked with proving moments, a proof reel. Compare: a draft against a reference, measured shot by shot.
Port of the Screen-Time web app (~/Projects/aetrai/screen-time, read-only here). Plan: ~/Projects/aetrai/hermes-hackathon-2026/plan/PLAN.md.

## Stack
- Python 3.11+ (Hermes runs 3.14). Deps: httpx, Pillow, meta-sam-parser (SAM License). ffmpeg/ffprobe external.
- Measure engine: Meta Model API (sam-3.1, muse-spark-1.3, muse-voice-transcribe-1.0), key `META_MODEL_API_KEY`
  through Hermes (`agent.secret_scope.get_secret`), never os.environ. Keyless: `replay` engine on fixtures/demo.
- Look-alike verifier: Muse Spark (validated) or the user's Hermes model via `ctx.llm.complete_structured` + aux task.

## Layout
- Root = the plugin: plugin.yaml, __init__.py (register only), schemas.py, tools.py, commands.py, cli.py, skills/.
- screentime_core/ = pure package, no Hermes imports (contract: docs/CONTRACT.md, types: models.py).
- tests/ (pytest, no network), dev/ (hand-run live scripts), fixtures/demo/ (bundled sample, builder's own footage).

## Commands
- Tests: `.venv/bin/python -m pytest` (venv: `uv venv --python 3.14 .venv && uv pip install --python .venv -r requirements-dev.txt`)
- Lint: `.venv/bin/ruff check .`
- Plugin checks (judge-path Hermes v0.23.0): `HERMES_HOME=~/hermes-judge/home ~/hermes-judge/hermes-agent/.venv/bin/hermes plugins doctor . --ci`
  and `... hermes plugins validate . --install-deps`
- Live Meta calls only through `~/ai/bin/with-secrets.sh` (key never printed); spend caps in code.

## Verification and security notes
- Gate test runs unpiped and check `$?` before committing.
- Public repo later: run `git status --porcelain` and read every untracked file before `git add`; never commit friends'
  Reels, labels with names, .env, or anything from screen-time/data except cut-down builder-owned clips in fixtures/demo.
- No shell=True, no eval/exec/pickle, no network at import, no os.environ key reads, no em dashes in shipped text.
- Deliverables (proof reels, cards) go under `$HERMES_HOME/cache/generated/screentime/<run>/`; runtime state under
  `plugin_data_dir("screentime")`.
