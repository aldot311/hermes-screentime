# TypeScript parity dump

`dump.mts` runs Screen-Time's own TypeScript measurement code (`src/lib/metrics/compute.ts`, `rules.ts`, `mentions.ts`
in the screen-time checkout) on a fixed, seeded set of inputs and writes the inputs together with the TypeScript's
outputs to `tests/parity/*.json`. The Python tests (`tests/test_metrics.py`, `test_rules.py`, `test_mentions.py`) replay
those files against `screentime_core` and need no Node, no network and no screen-time checkout.

## Run it

From the screen-time checkout, so tsx resolves its `@/...` import alias:

```
cd /Users/aldotile/Projects/aetrai/screen-time && ./node_modules/.bin/tsx --tsconfig tsconfig.json /Users/aldotile/Projects/aetrai/hermes-screentime/dev/ts-parity/dump.mts
```

`SCREEN_TIME_DIR` overrides where the checkout is. `pnpm exec tsx ...` works too; calling the tsx shim directly keeps
pnpm out of it. Then run `.venv/bin/python -m pytest` here.

Regenerate when the TypeScript changes. The output is deterministic (seed 20261108, no timestamps, non-ASCII escaped
as `\uXXXX`): two runs on the same TypeScript give byte-identical files, so a diff in `tests/parity/` is a behaviour
change. Each file records the screen-time commit, the sha256 of every source file and whether it matches that commit.

## Read-only on screen-time

The script imports the TypeScript, reads three stored reports and runs `git rev-parse` and `git hash-object` (without
`-w`). It writes only into this repo's `tests/parity/`. To check after a run:

```
touch /tmp/marker && (cd /Users/aldotile/Projects/aetrai/screen-time && ./node_modules/.bin/tsx --tsconfig tsconfig.json /Users/aldotile/Projects/aetrai/hermes-screentime/dev/ts-parity/dump.mts) && find /Users/aldotile/Projects/aetrai/screen-time -newer /tmp/marker
```

(prints nothing when nothing there changed).

## What is in the files

| File | TypeScript functions | Inputs |
|---|---|---|
| `compute.json` | computeMetrics, frameSamples, framesToSegments (also with maxGap 0, the strict runs) | the cases from compute.test.ts, gaps of exactly 1, 2 and 3 frames, ties, overlapping accepted objects, look-alikes, out-of-range and non-integer frames, empty masks, odd durations, 30 seeded random videos |
| `rules.json` | checkTerm, clock, clipSegments | the cases from rules.test.ts, every disclosure combination, thresholds that are exact rounding halves, and for each generated video every rule kind at, just inside and just past the 1e-6 tolerance and one frame either side of what it measured |
| `mentions.json` | tokenize, distinctNames, findMentions, findDisclosures | the cases from mentions.test.ts, aliases, case, punctuation, Unicode (decomposed accents, ligatures, full-width and math letters, emoji, Turkish and German case), repeated names, denials and window edges, 40 + 60 seeded random transcripts |
| `real_reports.json` | all of the above | three stored reports measured again by the current TypeScript |

The three reports are `data/out/lacroix-can/lacroix/lc-12-aldo` and `data/out/stanley-tumbler/clip11` (clips the builder
filmed himself) and `data/out/sneakers/stock/stock01` (Pexels stock). Only frame numbers, object ids, mask areas,
verdicts, typed terms, mention times with their words ("La Croix") and the spoken disclosure ("this is an ad") are
copied: no boxes, transcripts, prompts or video ids.

## Deliberate differences

The tests compare every field except these, which each have their own tests:

- `longest_run_sec` is strict (consecutive frames only); the TypeScript's spanned bridged gaps. `rules.json` also holds
  the TypeScript's min_continuous result computed on the strict run (`tsStrict`), which the Python must equal.
- Per-frame area is the largest accepted instance, not the sum; the tests compare it with a reference computed from the
  raw detections and show where the TypeScript's sum differed.
- "disclosed" reads only the transcript in v0.1: when nothing was said, the Python result is None (a person checks the
  screen) where the TypeScript, which also read frames, could fail it.
- No prominence score or centre weighting.
