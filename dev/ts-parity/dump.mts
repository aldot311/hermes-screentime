/**
 * Differential-test dump for the Python port (screentime_core/metrics.py, rules.py, mentions.py).
 *
 * Runs Screen-Time's TypeScript (src/lib/metrics/compute.ts, rules.ts, mentions.ts) on a fixed, seeded set of inputs
 * and writes the inputs with the TypeScript outputs to tests/parity/*.json in this repo; tests/test_metrics.py,
 * test_rules.py and test_mentions.py replay them against the Python. Read-only on the screen-time repo: it imports the
 * source, reads three stored reports and runs read-only git plumbing (rev-parse, hash-object without -w).
 *
 * Run from the screen-time checkout so tsx resolves its "@/..." import alias (see README.md here):
 *   cd /Users/aldotile/Projects/aetrai/screen-time && ./node_modules/.bin/tsx --tsconfig tsconfig.json \
 *     /Users/aldotile/Projects/aetrai/hermes-screentime/dev/ts-parity/dump.mts
 * SCREEN_TIME_DIR overrides where the screen-time checkout is.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ST = resolve(process.env.SCREEN_TIME_DIR ?? "/Users/aldotile/Projects/aetrai/screen-time");
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), "../../tests/parity");
const SOURCES = ["src/lib/metrics/compute.ts", "src/lib/metrics/rules.ts", "src/lib/metrics/mentions.ts", "src/lib/types.ts"];
const SEED = 20261108;

// Three stored reports, chosen for what they exercise and because their footage may be published: two clips the
// builder filmed himself and one Pexels stock clip (screen-time docs/ACCURACY.md). Only frame numbers, object ids, mask
// areas, verdicts, terms, mention times and disclosure phrases are copied; no boxes, transcripts, prompts or names.
const REPORTS = [
  { path: "data/out/lacroix-can/lacroix/lc-12-aldo/report.json", why: "builder's own clip: the product and a look-alike in every frame, brand said twice, spoken disclosure" },
  { path: "data/out/sneakers/stock/stock01/report.json", why: "Pexels stock clip: up to five accepted objects in one frame, gaps of 1 and 14 frames" },
  { path: "data/out/stanley-tumbler/clip11/report.json", why: "builder's own clip: three separate appearances in 30 s, product first seen late" },
];

const load = (rel: string) => import(pathToFileURL(join(ST, rel)).href);
const C = await load("src/lib/metrics/compute.ts");
const R = await load("src/lib/metrics/rules.ts");
const M = await load("src/lib/metrics/mentions.ts");
const T = await load("src/lib/types.ts");
const FPS: number = T.FPS;

// --------------------------------------------------------------------------------------------------------------------
// Shapes (the TypeScript's own, kept loose: tsx does not type-check)

type Row = [frame: number, objectId: number, areaFraction: number];
interface Info { width: number; height: number; frameCount: number; durationSec: number }
interface Segment { startSec: number; endSec: number }
interface Mention { startSec: number; endSec: number; text: string }
interface Sample { frame: number; areaFraction: number; centroid: { x: number; y: number } }
type Rule = Record<string, unknown> & { kind: string };
interface Disclosure { spoken: Mention[]; onScreen: { startSec: number; text: string }[]; onScreenChecked?: boolean; spokenChecked?: boolean }
interface Turn { startMs: number; endMs: number; transcript: string }

// --------------------------------------------------------------------------------------------------------------------
// Seeded randomness (mulberry32): the same draws on every machine and every run.

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
const int = (lo: number, hi: number) => lo + Math.floor(rand() * (hi - lo + 1));
const pick = <X,>(xs: readonly X[]): X => xs[Math.floor(rand() * xs.length)];
const round6 = (x: number) => Math.round(x * 1e6) / 1e6;
const range = (from: number, to: number) => Array.from({ length: Math.max(0, to - from) }, (_, i) => from + i);
const uniq = (xs: number[]) => [...new Set(xs)];
function shuffle<X>(xs: X[]): X[] {
  const out = [...xs];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// --------------------------------------------------------------------------------------------------------------------
// compute.ts

const info = (frameCount: number, durationSec = frameCount / FPS): Info => ({ width: 1080, height: 1920, frameCount, durationSec });
const toDetection = ([frame, objectId, areaFraction]: Row) => ({
  frame,
  objectId,
  box: { x1: 0, y1: 0, x2: 10, y2: 10 },
  areaFraction,
  centroid: { x: 0.5, y: 0.5 },
});
const rows = (frames: number[], objectId = 1, area: number | ((f: number) => number) = 0.1): Row[] =>
  frames.map((f) => [f, objectId, typeof area === "number" ? area : area(f)]);

/** The longest strict run (gaps not bridged) by the TypeScript's own framesToSegments(frames, 0); first one on a tie. */
function strictRun(frames: number[]) {
  let best: Segment | null = null;
  let bestFrames = 0;
  for (const s of C.framesToSegments(frames, 0) as Segment[]) {
    const n = Math.round((s.endSec - s.startSec) * FPS);
    if (n > bestFrames) [best, bestFrames] = [s, n];
  }
  return { frames: bestFrames, longestRunSec: bestFrames / FPS, longestRunStartSec: best ? best.startSec : null, segment: best };
}

function measure(detections: Row[], counted: number[], inf: Info, mentions: Mention[] = []) {
  const keep = new Set(counted);
  // As the TypeScript pipeline does (src/lib/pipeline/run.ts): only verified matches reach computeMetrics.
  const matched = detections.map(toDetection).filter((d) => keep.has(d.objectId));
  const metrics = C.computeMetrics(matched, inf, mentions);
  return { metrics, samples: C.frameSamples(matched, inf), strict: strictRun(metrics.visibleFrames) };
}

function computeCase(name: string, inf: Info, detections: Row[], counted: number[], mentions: Mention[] = []) {
  const { metrics, samples, strict } = measure(detections, counted, inf, mentions);
  return {
    name,
    info: inf,
    detections,
    counted,
    mentions,
    ts: {
      metrics,
      samples: samples.map((s) => [s.frame, s.areaFraction]),
      strictSegments: C.framesToSegments(metrics.visibleFrames, 0),
      strict: { longestRunSec: strict.longestRunSec, longestRunStartSec: strict.longestRunStartSec },
    },
  };
}

/** Objects that appear in runs separated by gaps of 1, 2, 3 and longer, some overlapping others. */
function randomVideo(): { inf: Info; detections: Row[]; counted: number[]; mentions: Mention[] } {
  const frameCount = int(20, 300);
  const durationSec = pick([frameCount / FPS, frameCount / FPS, frameCount / FPS + 0.05, frameCount / FPS - 0.2]);
  const objects = int(1, 4);
  const detections: Row[] = [];
  for (let o = 0; o < objects; o++) {
    let f = int(0, Math.floor(frameCount / 2));
    const base = 0.002 + rand() * 0.4;
    while (f < frameCount) {
      const run = int(1, 35);
      for (let k = 0; k < run && f + k < frameCount; k++) detections.push([f + k, o, round6(Math.max(0.0001, base + (rand() - 0.5) * 0.1))]);
      f += run + pick([1, 1, 2, 2, 3, 3, 4, 6, 15, 40]);
    }
  }
  // Now and then something the metrics must ignore: an empty mask, a negative area, an area above 1, a frame outside
  // the video or between frames, and the same box twice.
  if (rand() < 0.5) detections.push([int(0, frameCount - 1), 0, 0]);
  if (rand() < 0.3) detections.push([int(0, frameCount - 1), 0, -0.05]);
  if (rand() < 0.3) detections.push([int(0, frameCount - 1), 0, 1.4]);
  if (rand() < 0.3) detections.push([frameCount, 0, 0.2], [-1, 0, 0.2]);
  if (rand() < 0.3) detections.push([int(0, frameCount - 2) + 0.5, 0, 0.2]);
  if (rand() < 0.3 && detections.length) detections.push([...detections[0]] as Row);
  const ids = range(0, objects);
  const counted = rand() < 0.15 ? [] : ids.filter(() => rand() < 0.7);
  if (!counted.length && rand() < 0.8) counted.push(0);
  const mentions: Mention[] = [];
  for (let k = int(0, 3); k > 0; k--) {
    const at = round6(rand() * durationSec);
    mentions.push({ startSec: at, endSec: round6(at + 0.4), text: pick(["LaCroix", "la croix", "LACROIX"]) });
  }
  return { inf: info(frameCount, durationSec), detections: rand() < 0.4 ? shuffle(detections) : detections, counted, mentions };
}

const computeCases: ReturnType<typeof computeCase>[] = [];
// From compute.test.ts
computeCases.push(computeCase("ts test: empty video", { width: 0, height: 0, frameCount: 0, durationSec: 0 }, [], [1]));
computeCases.push(computeCase("ts test: no detections", info(150), [], [1]));
computeCases.push(computeCase("ts test: covers every frame", info(150), rows(range(0, 150), 1, 0.2), [1]));
computeCases.push(computeCase("ts test: blur gaps bridged", info(150), rows([...range(0, 30).filter((f) => f !== 15 && f !== 16), ...range(60, 75)]), [1]));
computeCases.push(computeCase("ts test: first seen at 3 s", info(150), rows(range(45, 60)), [1]));
computeCases.push(computeCase("ts test: two objects in one frame count once", info(15), [[0, 1, 0.1], [0, 2, 0.1]], [1, 2]));
computeCases.push(computeCase("ts test: mentions sorted", info(150), [], [1], [{ startSec: 5, endSec: 6, text: "b" }, { startSec: 1, endSec: 2, text: "a" }]));
computeCases.push(computeCase("ts test: summed area of two objects (python: largest)", info(10), [[3, 1, 0.1], [3, 2, 0.3]], [1, 2]));
computeCases.push(computeCase("ts test: summed area capped at 1, out-of-range dropped (python: largest)", info(10), [[0, 1, 0.7], [0, 2, 0.7], [10, 1, 0.1], [-1, 1, 0.1]], [1, 2]));
computeCases.push(computeCase("ts test: zero and negative area not on screen", info(30), [[0, 1, 0], [1, 1, -0.1], [2, 1, 0.1]], [1]));
computeCases.push(computeCase("ts test: big early product", info(150), rows(range(0, 30), 1, 0.2), [1]));
computeCases.push(computeCase("ts test: small late product", info(150), rows(range(120, 150), 1, 0.02), [1]));
// Gap bridging and strict runs
computeCases.push(computeCase("gap of 1 frame", info(60), rows([...range(0, 10), ...range(11, 21)]), [1]));
computeCases.push(computeCase("gap of 2 frames", info(60), rows([...range(0, 10), ...range(12, 21)]), [1]));
computeCases.push(computeCase("gap of 3 frames", info(60), rows([...range(0, 10), ...range(13, 21)]), [1]));
computeCases.push(computeCase("gaps of 1, 2 and 3 with the longest strict run last", info(90), rows([...range(0, 5), ...range(6, 9), ...range(11, 31), ...range(34, 41)]), [1]));
computeCases.push(computeCase("two equal strict runs (first wins)", info(60), rows([...range(0, 10), ...range(20, 30)]), [1]));
computeCases.push(computeCase("bridged run longer than any strict run", info(60), rows([...range(0, 8), ...range(10, 18), ...range(20, 28)]), [1]));
computeCases.push(computeCase("single frame at 0", info(30), rows([0]), [1]));
computeCases.push(computeCase("single frame at the last frame", info(30), rows([29]), [1]));
computeCases.push(computeCase("frame equal to frameCount is outside", info(30), rows([30]), [1]));
computeCases.push(computeCase("non-integer frames are not frames", info(30), [[2.5, 1, 0.2], [3, 1, 0.2]], [1]));
// Counting only the accepted objects
computeCases.push(computeCase("look-alike only: nothing counted", info(60), [...rows(range(0, 30), 1), ...rows(range(10, 40), 2)], []));
computeCases.push(computeCase("product and look-alike interleaved", info(60), [...rows(range(0, 20), 1, 0.05), ...rows(range(15, 45), 2, 0.3)], [1]));
computeCases.push(computeCase("overlapping accepted objects, areas summing past 1 (python: largest)", info(45), [...rows(range(0, 30), 1, 0.6), ...rows(range(10, 40), 2, 0.55)], [1, 2]));
computeCases.push(computeCase("one instance above 1 is clamped", info(15), [[0, 1, 1.7], [1, 1, 0.4]], [1]));
computeCases.push(computeCase("same box twice", info(15), [[4, 1, 0.2], [4, 1, 0.2]], [1]));
computeCases.push(computeCase("detections out of order", info(40), shuffle(rows([...range(0, 12), ...range(14, 20), ...range(30, 33)], 1, (f) => round6(0.01 + f / 100))), [1]));
// Duration and share
computeCases.push(computeCase("audio longer than the frames", info(150, 10.05), rows(range(0, 75)), [1]));
computeCases.push(computeCase("duration shorter than the counted frames (share capped at 1)", info(150, 4), rows(range(0, 150)), [1]));
computeCases.push(computeCase("zero duration with frames (share 0)", info(30, 0), rows(range(0, 10)), [1]));
computeCases.push(computeCase("mentions with equal start times keep their order", info(30), rows([1]), [1], [
  { startSec: 2, endSec: 2.5, text: "second" },
  { startSec: 1, endSec: 1.5, text: "first" },
  { startSec: 2, endSec: 2.2, text: "third" },
]));
// Seeded random videos
const randomVideos = range(0, 30).map(() => randomVideo());
randomVideos.forEach((v, i) => computeCases.push(computeCase(`random ${i}`, v.inf, v.detections, v.counted, v.mentions)));

const framesToSegmentsCases: { frames: number[]; maxGap: number; ts: Segment[] }[] = [];
for (const frames of [[], [0], [15], [0, 1, 3, 4, 7, 8], [0, 1, 5, 6], [2, 3, 4, 8, 12, 13, 17, 21, 22, 23], range(0, 40).filter((f) => f % 3 !== 0), [5, 9, 10, 11, 20, 24, 28, 29]]) {
  for (const maxGap of [0, 1, 2, 3]) framesToSegmentsCases.push({ frames, maxGap, ts: C.framesToSegments(frames, maxGap) });
}

// --------------------------------------------------------------------------------------------------------------------
// rules.ts

const DELTAS = [0, 5e-7, -5e-7, 2e-6, -2e-6, 1 / 15, -1 / 15]; // at, inside EPS either side, past EPS either side, a frame
const near = (base: number, deltas = DELTAS) => deltas.map((d) => base + d).filter((x) => x > 0);

/** The TypeScript's detected seconds in [0, limit) (rules.ts secondsWithin), only to place thresholds around it. */
function secondsWithin(frames: Sample[], limit: number): number {
  let total = 0;
  for (const f of frames) {
    const start = f.frame / FPS;
    total += Math.max(0, Math.min(start + 1 / FPS, limit) - start);
  }
  return total;
}

/** Every rule kind at, just below and just above the thresholds this video meets or misses. */
function battery(m: any, frames: Sample[], strict: { longestRunSec: number }, inf: Info): Rule[] {
  const out: Rule[] = [];
  for (const s of near(m.secondsOnScreen)) out.push({ kind: "min_seconds_total", seconds: s });
  if (m.secondsOnScreen === 0) out.push({ kind: "min_seconds_total", seconds: 1 });
  for (const base of uniq([strict.longestRunSec, m.longestRunSec])) for (const s of near(base)) out.push({ kind: "min_continuous", seconds: s });
  if (m.longestRunSec === 0) out.push({ kind: "min_continuous", seconds: 1 });
  if (m.firstSeenSec === null) out.push({ kind: "first_seen_before", seconds: 2 });
  else for (const s of near(m.firstSeenSec)) out.push({ kind: "first_seen_before", seconds: s });
  const windows = [1, 3, inf.durationSec];
  if (frames.length) {
    const f = frames[Math.floor(frames.length / 2)].frame;
    windows.push(f / FPS, (f + 0.5) / FPS);
  }
  for (const w of uniq(windows)) {
    if (!(w > 0)) continue;
    const within = secondsWithin(frames, w);
    for (const s of near(within, [0, 5e-7, 2e-6, -2e-6, 1 / 15])) out.push({ kind: "min_seconds_within", seconds: s, withinFirstSec: w });
    if (within === 0) out.push({ kind: "min_seconds_within", seconds: 0.5, withinFirstSec: w });
  }
  const areas = frames.map((f) => f.areaFraction).sort((a, b) => a - b);
  if (areas.length) {
    for (const a of uniq([areas[0], areas[Math.floor(areas.length / 2)], areas[areas.length - 1]])) {
      for (const fraction of [a, a + 5e-7, a + 2e-6, a - 2e-6]) {
        if (!(fraction > 0 && fraction <= 1)) continue;
        const n = frames.filter((f) => f.areaFraction + 1e-6 >= fraction).length;
        for (const s of near(n / FPS, [0, 2e-6, -2e-6, 1 / 15])) out.push({ kind: "min_area", fraction, forSeconds: s });
      }
    }
  } else out.push({ kind: "min_area", fraction: 0.1, forSeconds: 1 });
  const n = m.mentions.length;
  for (const t of [n - 1, n, n + 1]) if (t >= 1) out.push({ kind: "mentioned", minTimes: t });
  for (const x of m.mentions.slice(0, 2)) for (const w of [x.startSec, x.startSec + 1e-9, x.startSec + 0.5]) if (w > 0) out.push({ kind: "mentioned", minTimes: 1, withinFirstSec: w });
  if (!n) out.push({ kind: "mentioned", minTimes: 1, withinFirstSec: 5 });
  out.push({ kind: "disclosed" });
  return out;
}

function outcome(r: { pass: boolean | null; evidence: Segment[]; detail: string }) {
  return { pass: r.pass, evidence: r.evidence, detail: r.detail };
}

/**
 * One metrics object checked against many rules, as the pipeline calls checkTerms (frames = frameSamples of the
 * matched detections). frames null calls it without per-frame data (older reports), and the case's metrics then carry
 * no visibleFrames. min_continuous is also checked on the strict run (tsStrict), the Python's deliberate difference,
 * by handing checkTerm that run as both the longest run and the only segment.
 */
function rulesCase(
  name: string,
  metrics: any,
  frames: Sample[] | null,
  disclosure: Disclosure | null,
  rules: (Rule | null)[],
  strict: { longestRunSec: number; longestRunStartSec: number | null; segment: Segment | null },
) {
  const m = frames === null ? { ...metrics, visibleFrames: [] } : metrics;
  const checks = rules.map((rule) => {
    const term = { text: "contract wording", rule };
    const check: Record<string, unknown> = { rule, ts: outcome(R.checkTerm(term, m, frames ?? undefined, disclosure ?? undefined)) };
    if (rule?.kind === "min_continuous") {
      const onStrict = { ...m, longestRunSec: strict.longestRunSec, segments: strict.segment ? [strict.segment] : [] };
      check.tsStrict = outcome(R.checkTerm(term, onStrict, frames ?? undefined, disclosure ?? undefined));
    }
    return check;
  });
  return {
    name,
    fps: FPS,
    metrics: m,
    strict: { longestRunSec: strict.longestRunSec, longestRunStartSec: strict.longestRunStartSec },
    frames: frames === null ? null : frames.map((f) => [f.frame, f.areaFraction]),
    disclosure,
    checks,
  };
}

/** Hand-made metrics (as in rules.test.ts) have no frames: their segments stand in for strict runs. */
function strictFromSegments(segments: Segment[]) {
  let best: Segment | null = null;
  for (const s of segments) if (!best || s.endSec - s.startSec > best.endSec - best.startSec) best = s;
  return { longestRunSec: best ? best.endSec - best.startSec : 0, longestRunStartSec: best ? best.startSec : null, segment: best };
}

const handMetrics = (over: Record<string, unknown> = {}) => ({
  secondsOnScreen: 0, shareOfVideo: 0, firstSeenSec: null, longestRunSec: 0, avgAreaFraction: 0, centerWeightedSec: 0,
  prominenceScore: 0, segments: [] as Segment[], mentions: [] as Mention[], ...over,
});
const sample = (frame: number, areaFraction = 0.1): Sample => ({ frame, areaFraction, centroid: { x: 0.5, y: 0.5 } });
const seen = handMetrics({
  secondsOnScreen: 3.4, firstSeenSec: 1, longestRunSec: 2,
  segments: [{ startSec: 1, endSec: 3 }, { startSec: 8, endSec: 9.4 }],
  mentions: [{ startSec: 2, endSec: 2.5, text: "Nike" }, { startSec: 12, endSec: 12.4, text: "nike" }],
});
const bridged = handMetrics({ secondsOnScreen: 2 / 15, firstSeenSec: 0, longestRunSec: 4 / 15, segments: [{ startSec: 0, endSec: 4 / 15 }] });
const areaFrames = [...range(0, 20).map((f) => sample(f, 0.12)), ...range(40, 70).map((f) => sample(f, 0.05))];

const rulesCases: ReturnType<typeof rulesCase>[] = [];
const hand = (name: string, m: any, frames: Sample[] | null, disclosure: Disclosure | null, rules: (Rule | null)[]) =>
  rulesCases.push(rulesCase(name, m, frames, disclosure, rules, strictFromSegments(m.segments)));

// From rules.test.ts, called as the tests call checkTerm (frames given or not).
hand("ts test: seen, no frames", seen, null, null, [
  null,
  { kind: "min_seconds_total", seconds: 3.4 },
  { kind: "min_seconds_total", seconds: 3.5 },
  { kind: "min_seconds_within", seconds: 3, withinFirstSec: 8.5 },
  { kind: "min_seconds_within", seconds: 2, withinFirstSec: 3 },
  { kind: "min_seconds_within", seconds: 0.1, withinFirstSec: 1 },
  { kind: "min_continuous", seconds: 2 },
  { kind: "min_continuous", seconds: 2.5 },
  { kind: "min_continuous", seconds: 1 },
  { kind: "min_area", fraction: 0.1, forSeconds: 1 },
  { kind: "mentioned", minTimes: 2 },
  { kind: "mentioned", minTimes: 2, withinFirstSec: 12 },
  { kind: "first_seen_before", seconds: 2 },
  { kind: "first_seen_before", seconds: 1 },
]);
hand("ts test: seen, frames 15..44", seen, range(15, 45).map((f) => sample(f)), null, [{ kind: "min_seconds_within", seconds: 2, withinFirstSec: 3 }]);
hand("ts test: seen, area frames", seen, areaFrames, null, [
  { kind: "min_area", fraction: 0.12, forSeconds: 1 },
  { kind: "min_area", fraction: 0.5, forSeconds: 1 },
  { kind: "min_area", fraction: 0.05, forSeconds: 3 },
  { kind: "min_area", fraction: 0.1, forSeconds: 1.4 },
]);
hand("ts test: 45 frames' worth of seconds", handMetrics({ secondsOnScreen: 45 * (1 / 15) }), null, null, [{ kind: "min_seconds_total", seconds: 3 }]);
hand("ts test: near miss", handMetrics({ secondsOnScreen: 2.96 }), null, null, [{ kind: "min_seconds_total", seconds: 3 }]);
hand("ts test: empty metrics, empty frames", handMetrics(), [], null, [
  { kind: "min_seconds_total", seconds: 1 },
  { kind: "min_seconds_within", seconds: 1, withinFirstSec: 5 },
  { kind: "first_seen_before", seconds: 3 },
  { kind: "min_area", fraction: 0.1, forSeconds: 1 },
  { kind: "min_continuous", seconds: 1 },
  { kind: "mentioned", minTimes: 1 },
]);
hand("ts test: a later mention is the closest miss", handMetrics({ mentions: [{ startSec: 20, endSec: 21, text: "Nike" }] }), [], null, [{ kind: "mentioned", minTimes: 1, withinFirstSec: 10 }]);
hand("ts test: audio not transcribed", handMetrics({ spokenChecked: false }), [], null, [{ kind: "mentioned", minTimes: 1 }]);
hand("ts test: bridged, no frames", bridged, null, null, [
  { kind: "min_seconds_within", seconds: 0.25, withinFirstSec: 10 },
  { kind: "min_seconds_total", seconds: 0.25 },
]);
hand("ts test: bridged, frames 0 and 3", bridged, [sample(0), sample(3)], null, [
  { kind: "min_seconds_within", seconds: 0.1, withinFirstSec: 0.1 },
  { kind: "min_seconds_within", seconds: 2 / 15, withinFirstSec: 1 },
]);
hand("ts test: frame straddling the window end", handMetrics({ secondsOnScreen: 2 / 15, firstSeenSec: 0, segments: [{ startSec: 0, endSec: 2 / 15 }] }), [sample(0), sample(1)], null, [
  { kind: "min_seconds_within", seconds: 0.11, withinFirstSec: 0.1 },
  { kind: "min_seconds_within", seconds: 0.1, withinFirstSec: 0.1 },
]);
const twoRuns = handMetrics({ secondsOnScreen: 6 / 15, firstSeenSec: 0, longestRunSec: 4 / 15, segments: [{ startSec: 0, endSec: 4 / 15 }, { startSec: 2, endSec: 34 / 15 }] });
hand("ts test: bridged time not used for the window, no frames", twoRuns, null, null, [{ kind: "min_seconds_within", seconds: 0.2, withinFirstSec: 1 }]);
hand("ts test: bridged time not used for the window, frames", twoRuns, [0, 3, 30, 31, 32, 33].map((f) => sample(f)), null, [{ kind: "min_seconds_within", seconds: 0.2, withinFirstSec: 1 }]);
// Thresholds that are exact halves in binary at the rounding digit (0.125 * 100 = 12.5, 0.0625 * 1000 = 62.5): JavaScript's
// Math.round sends them up, Python's round() to the even neighbour, so the port must not use round().
hand(
  "rounding halves",
  handMetrics({
    secondsOnScreen: 3, firstSeenSec: 0, longestRunSec: 3, segments: [{ startSec: 0, endSec: 3 }],
    mentions: [{ startSec: 2, endSec: 2.5, text: "LaCroix" }, { startSec: 12.25, endSec: 12.75, text: "LaCroix" }],
  }),
  [...range(0, 30).map((f) => sample(f, 0.0625)), ...range(30, 45).map((f) => sample(f, 0.1875))],
  null,
  [
    { kind: "min_seconds_total", seconds: 0.125 },
    { kind: "min_seconds_total", seconds: 2.125 },
    { kind: "min_seconds_total", seconds: 10.875 },
    { kind: "min_seconds_within", seconds: 0.375, withinFirstSec: 0.625 },
    { kind: "min_seconds_within", seconds: 2.625, withinFirstSec: 2.375 },
    { kind: "min_continuous", seconds: 1.875 },
    { kind: "min_area", fraction: 0.0625, forSeconds: 0.125 },
    { kind: "min_area", fraction: 0.1875, forSeconds: 1.625 },
    { kind: "min_area", fraction: 0.3125, forSeconds: 0.375 },
    { kind: "min_area", fraction: 0.4375, forSeconds: 0.875 },
    { kind: "first_seen_before", seconds: 0.625 },
    { kind: "mentioned", minTimes: 1, withinFirstSec: 2.125 },
    { kind: "mentioned", minTimes: 2, withinFirstSec: 12.375 },
  ],
);

// Disclosure: every combination of spoken, on screen and failed checks (rules.test.ts and more).
const said = (startSec: number, text: string, endSec = startSec + 0.35): Mention => ({ startSec, endSec, text });
const disclosures: [string, Disclosure | null][] = [
  ["none given (older report)", null],
  ["spoken", { spoken: [said(1.25, "ad", 1.6)], onScreen: [] }],
  ["on screen only", { spoken: [], onScreen: [{ startSec: 0.5, text: "#sponsored" }] }],
  ["spoken late, two on screen out of order", { spoken: [said(75.5, "sponsored", 76)], onScreen: [{ startSec: 9, text: "#ad" }, { startSec: 2, text: "#ad" }] }],
  ["two spoken out of order", { spoken: [said(9, "this is an ad"), said(3, "#ad")], onScreen: [] }],
  ["nothing said or shown", { spoken: [], onScreen: [] }],
  ["frame check failed, nothing said", { spoken: [], onScreen: [], onScreenChecked: false }],
  ["frame check failed, said", { spoken: [said(1, "this is an ad", 1.2)], onScreen: [], onScreenChecked: false }],
  ["frame check ran, nothing", { spoken: [], onScreen: [], onScreenChecked: true }],
  ["not transcribed, nothing shown", { spoken: [], onScreen: [], onScreenChecked: true, spokenChecked: false }],
  ["not transcribed, frame check failed", { spoken: [], onScreen: [], onScreenChecked: false, spokenChecked: false }],
  ["not transcribed, shown", { spoken: [], onScreen: [{ startSec: 1, text: "#ad" }], onScreenChecked: true, spokenChecked: false }],
  ["said at 0:00", { spoken: [said(0, "Ad")], onScreen: [] }],
  ["said at 9.99 s", { spoken: [said(9.99, "sponsored by")], onScreen: [] }],
];
for (const [label, d] of disclosures) hand(`disclosed: ${label}`, seen, null, d, [{ kind: "disclosed" }, { kind: "min_continuous", seconds: 1 }]);

// Generated: metrics from computeMetrics, frames from frameSamples, as the pipeline passes them.
const disclosureCycle = disclosures.map(([, d]) => d);
randomVideos.slice(0, 14).forEach((v, i) => {
  const { metrics, samples, strict } = measure(v.detections, v.counted, v.inf, v.mentions);
  const m = i % 5 === 3 ? { ...metrics, spokenChecked: false } : metrics;
  rulesCases.push(rulesCase(`random ${i}`, m, samples, disclosureCycle[i % disclosureCycle.length], battery(m, samples, strict, v.inf), strict));
  if (i < 4 && metrics.secondsOnScreen > 0) {
    // The same metrics without per-frame data: min_seconds_within falls back to its upper bound, min_area to "needs data".
    const rules = battery(m, samples, strict, v.inf).filter((r) => r.kind === "min_seconds_within" || r.kind === "min_area");
    rulesCases.push(rulesCase(`random ${i}, no frames`, m, null, null, rules.slice(0, 24), strict));
  }
});
for (const name of ["gaps of 1, 2 and 3 with the longest strict run last", "bridged run longer than any strict run", "two equal strict runs (first wins)", "overlapping accepted objects, areas summing past 1 (python: largest)"]) {
  const c = computeCases.find((x) => x.name === name)!;
  const { metrics, samples, strict } = measure(c.detections, c.counted, c.info, c.mentions);
  rulesCases.push(rulesCase(name, metrics, samples, { spoken: [], onScreen: [] }, battery(metrics, samples, strict, c.info), strict));
}

const clockCases = [0, 1.25, 9.99, 60, 75.5, 600.04, 0.05, 0.1, 0.15, 0.3, 0.7, 1.15, 59.95, 59.99, 599.99, 3599.9, 1e-7, -1, 2 / 15, 12.918181818181818, ...range(0, 20).map(() => round6(rand() * 700))].map((s) => [s, R.clock(s)]);
const clipCases = [
  { segments: [{ startSec: 0, endSec: 2 }, { startSec: 4, endSec: 6 }], limit: 5 },
  { segments: [{ startSec: 0, endSec: 2 }, { startSec: 4, endSec: 6 }], limit: 4 },
  { segments: [{ startSec: 0, endSec: 2 }, { startSec: 4, endSec: 6 }], limit: 2 },
  { segments: [{ startSec: 1, endSec: 3 }, { startSec: 8, endSec: 9.4 }], limit: 8.5 },
  { segments: [{ startSec: 1, endSec: 3 }], limit: 1 },
  { segments: [], limit: 3 },
  { segments: [{ startSec: 1 / 15, endSec: 7 / 15 }, { startSec: 11 / 15, endSec: 2 }], limit: 0.5 },
].map((c) => ({ ...c, ts: R.clipSegments(c.segments, c.limit) }));

// --------------------------------------------------------------------------------------------------------------------
// mentions.ts

/** JavaScript offsets count UTF-16 units; Python's count code points. Both slice the same text. */
const codePoints = (text: string, utf16: number) => [...text.slice(0, utf16)].length;

const tokenizeTexts = [
  "Nike's Café-Bar, OK?", "Nike’s", "Pe\u0301rez", "Pérez", "ﬁne ﬂavour", "ＬａＣｒｏｉｘ", "İstanbul", "Straße", "ΣΑΣ ΟΔΟΣ",
  "rock'n'roll", "rock'n9", "90's", "it's", "IT'S", "don't", "isn't", "o'clock", "'quoted'", "‘single’", "x’", "a'", "'s",
  "I ❤\ufe0f LaCroix 🥤!", "𝐋𝐚𝐂𝐫𝐨𝐢𝐱 can", "日本語のテキスト", "7UP 7-Up", "\u0301", "a\u0301\u0302b", "e\u200dmoji", "½ cup",
  "tab\tseparated\nlines", "H&M", "Ben & Jerry's", "co\u00adop", "La Croix LaCroix.", "", "   ", "!!", "AT&T's", "naïve café",
  "Ǆemal", "ﬀ", "Ⅻ", "x²", "l’été", "SÃO PAULO", "the end.", "#ad #sponsored", "ad-free", "can't won't",
];
const tokenizeCases = tokenizeTexts.map((text) => ({
  text,
  ts: (M.tokenize(text) as { norm: string; start: number; end: number }[]).map((t) => ({ norm: t.norm, start: codePoints(text, t.start), end: codePoints(text, t.end) })),
}));

const distinctNamesCases = [
  ["LaCroix", "La Croix", "la-croix", "Pamplemousse", "  ", "H&M", "H and M"],
  [],
  ["Nestlé", "Nestle", "NESTLE\u0301"],
  ["Coca-Cola", "coca cola", "CocaCola", "Coke"],
  ["!!", "", "Red Bull", "RedBull", "Red Bull Racing"],
].map((names) => ({ names, ts: M.distinctNames(names) }));

const turn = (transcript: string, startMs = 0, endMs = 10_000): Turn => ({ startMs, endMs, transcript });
const mentionCases: { name: string; turns: Turn[]; terms: string[]; ts: Mention[] }[] = [];
const mentionCase = (name: string, turns: Turn[], terms: string[]) => mentionCases.push({ name, turns, terms, ts: M.findMentions(turns, terms) });
// From mentions.test.ts
mentionCase("ts test: no turns", [], ["Nike"]);
mentionCase("ts test: no terms", [turn("Nike")], []);
mentionCase("ts test: blank terms", [turn("Nike")], ["  ", "!!"]);
mentionCase("ts test: case, punctuation, possessive", [turn("I love NIKE. Nike's new shoes, nike!")], ["Nike"]);
mentionCase("ts test: ampersand said as and", [turn("got this at H and M, and some Ben & Jerry's")], ["H&M", "Ben & Jerry's"]);
mentionCase("ts test: whole words only", [turn("Nikesh and nikes went to Nikeland")], ["Nike"]);
mentionCase("ts test: spaced, hyphenated, joined", [turn("Coca-Cola, coca cola and CocaCola")], ["Coca-Cola"]);
mentionCase("ts test: red bull", [turn("a red bull please")], ["RedBull"]);
mentionCase("ts test: partial multi-word brand", [turn("I drank coca water")], ["Coca-Cola"]);
mentionCase("ts test: longest term wins", [turn("Red Bull Racing rocks")], ["Red Bull", "Red Bull Racing"]);
mentionCase("ts test: accents either way", [turn("I bought Nestle")], ["Nestlé"]);
mentionCase("ts test: time by word position", [turn("we wear nike daily", 2000, 6000)], ["nike"]);
mentionCase("ts test: sorted across turns", [turn("nike", 5000, 6000), turn("Nike", 1000, 2000)], ["Nike"]);
mentionCase("ts test: zero-length turn", [turn("Nike", 3000, 3000)], ["nike"]);
mentionCase("ts test: La Croix LaCroix is one mention", [turn("than a La Croix LaCroix. This is an ad", 0, 9000)], ["LaCroix", "La Croix"]);
mentionCase("ts test: two different names side by side", [turn("a LaCroix Pamplemousse can", 0, 4000)], ["LaCroix", "Pamplemousse"]);
mentionCase("ts test: Nike, Nike is two", [turn("Nike, Nike!")], ["Nike"]);
mentionCase("ts test: LaCroix and LaCroix is two", [turn("LaCroix and LaCroix")], ["LaCroix"]);
mentionCase("ts test: decomposed accent kept whole", [turn("I love Nestle\u0301 bars", 0, 4000)], ["Nestlé"]);
// More: aliases, case, punctuation, Unicode, repeats
mentionCase("three spellings in a row", [turn("La Croix LaCroix la croix, then LACROIX", 1000, 7000)], ["LaCroix", "La Croix"]);
mentionCase("repeat across a comma and a newline", [turn("LaCroix,\nLaCroix\nLaCroix", 0, 3000)], ["LaCroix"]);
mentionCase("repeat across a no-break space", [turn("LaCroix\u00a0LaCroix and Nike\u2003Nike", 0, 3000)], ["LaCroix", "Nike"]);
mentionCase("hyphen between repeats is not only a space", [turn("LaCroix-LaCroix", 0, 2000)], ["LaCroix"]);
mentionCase("full-width and math-bold letters", [turn("ＬａＣｒｏｉｘ and 𝐋𝐚𝐂𝐫𝐨𝐢𝐱 and Lacroix’s", 0, 6000)], ["LaCroix"]);
mentionCase("emoji between words", [turn("I ❤\ufe0f LaCroix 🥤 and nike👟 too", 0, 5000)], ["LaCroix", "Nike"]);
mentionCase("curly possessive and quotes", [turn("“LaCroix’s” best? ‘LaCroix’!", 0, 3000)], ["LaCroix"]);
mentionCase("accented alias, plain speech", [turn("cafe, CAFÉ, Café's and Cafe\u0301", 0, 4000)], ["Café"]);
mentionCase("numbers in names", [turn("7UP, 7 up and 7-Up but not 7upx", 500, 4500)], ["7UP"]);
mentionCase("ampersand alias heard both ways", [turn("AT&T, AT and T, ATT and at t", 0, 8000)], ["AT&T"]);
mentionCase("name inside a longer token", [turn("lacroixes LaCroixLaCroix LaCroix", 0, 3000)], ["LaCroix"]);
mentionCase("overlapping aliases, longest first", [turn("red bull racing, red bull, bull racing", 0, 6000)], ["Bull Racing", "Red Bull", "Red Bull Racing"]);
mentionCase("duplicate aliases", [turn("Nike nike NIKE", 0, 3000)], ["Nike", "nike", "NIKE", "N-I-K-E"]);
mentionCase("turn with only punctuation", [turn("... !!! ???", 0, 3000), turn("LaCroix", 3000, 4000)], ["LaCroix"]);
mentionCase("end before start", [turn("LaCroix please", 5000, 4000)], ["LaCroix"]);
mentionCase("odd millisecond times", [turn("one two LaCroix three four five", 1234, 98765), turn("La Croix", 98765, 99001)], ["LaCroix"]);
mentionCase("Turkish dotted I", [turn("İstanbul and ISTANBUL", 0, 2000)], ["istanbul"]);
mentionCase("German sharp s", [turn("Straße and STRASSE", 0, 2000)], ["Straße"]);
const vocab = ["LaCroix", "La Croix", "la-croix", "Lacroix's", "LACROIX", "lacroix", "Pamplemousse", "pamplemousse", "Nike", "nike", "H&M", "H and M", "Nestlé", "Nestle", "I", "love", "this", "drink", "so", "much", "it's", "really", "good", "a", "the", "can", "of", "and", "la", "croix", "honestly", "today", "guys", "😋", "Croix", "LaCroixx"];
const seps = [" ", " ", " ", " ", ", ", ". ", "! ", "? ", "-", " - ", "\n", "  ", "... "];
const aliasSets = [["LaCroix", "La Croix"], ["LaCroix"], ["Pamplemousse", "LaCroix"], ["Nike", "H&M"], ["Nestlé"], ["LaCroix", "La Croix", "Pamplemousse", "Nike"]];
for (let k = 0; k < 40; k++) {
  const turns: Turn[] = [];
  let t = int(0, 3000);
  for (let n = int(1, 3); n > 0; n--) {
    const words = range(0, int(1, 18)).map(() => pick(vocab));
    let text = "";
    words.forEach((w, i) => (text += (i ? pick(seps) : "") + w));
    const end = t + int(0, 6000);
    turns.push(turn(text, t, end));
    t = end + int(0, 1500);
  }
  mentionCase(`random ${k}`, rand() < 0.3 ? shuffle(turns) : turns, pick(aliasSets));
}

const disclosureCases: { name: string; turns: Turn[]; ts: Mention[] }[] = [];
const disclosureCase = (name: string, turns: Turn[]) => disclosureCases.push({ name, turns, ts: M.findDisclosures(turns) });
const disclosureTexts = [
  // mentions.test.ts: disclosures
  "#ad", "Loving this drink #ad #sponsored", "hashtag ad, hashtag sponsored", "This is an ad.", "this video is an ad for Alani",
  "Just so you know, it's a paid ad", "Ad. Okay so today", "Sponsored by Alani. Ad!", "This is sponsored, guys",
  "this post is paid sponsored content", "a paid partnership with Alani", "I partnered with Alani", "made in partnership with Alani",
  "thanks to Alani Nu for sponsoring this video", "Thank you so much Alani for sponsoring", "This is a sponsored video",
  "this post is a paid sponsored one",
  // mentions.test.ts: denials, passing uses, look-alike words
  "This is not sponsored", "this isn't an ad", "no ads", "I saw an ad", "this was never sponsored by anyone", "not a paid partnership",
  "I'm not partnered with them", "without an ad, this is great", "I skipped the ad", "that sponsored post was bad", "an ad came on",
  "the ad was loud", "ads everywhere", "I saw a paid ad", "add it to the cart", "I had one", "I made this", "advert", "advertising",
  "adidas", "a d", "A.D. 2020", "ad-free plan", "This is an ad-free video", "it's a sponsored-content-free zone", "my dad", "bad",
  "sponsor", "sponsorship", "partner", "partnership", "paid for it", "thanks for watching",
  "No way. This is an ad.", "that's actually not bad this is an ad but I'd still buy it", "not gonna lie this is an ad",
  "No, it's sponsored by Alani", "this is not an ad", "honestly I would drink them even if it wasn't an ad",
  "I don't think this is an ad", "I'm not saying it's sponsored",
  // More
  "THIS IS AN AD", "This’s an ad", "It’s a paid ad!", "that one is an ad", "this reel is sponsored", "this clip is a paid sponsored",
  "Ad: today we try LaCroix", "ad", "Ad!", "Ad?", "ad.", "So. Ad. Done", "it's an ad-", "#ad-free", "-ad", "# ad",
  "Thanks so much to LaCroix and everyone at the team for sponsoring", "thanks to a b c d e f g h for sponsoring",
  "thanks to a b c d e f g h i for sponsoring", "Thanks. LaCroix for sponsoring", "thank you for sponsoring",
  "this is not, I repeat, an ad", "I never said this is an ad", "I am sure this is an ad", "don't worry this is an ad",
  "it's not sponsored by them", "not sponsored", "No. Sponsored by LaCroix", "never in partnership with them",
  "this isnt an ad", "this is NOT a paid ad", "dont think it's sponsored", "sponsored sponsored by", "paid paid partnership",
  "in partnership with partnered with", "hashtag hashtag ad", "#sponsored by LaCroix",
  "this video\u00a0is an ad", "this is an ad\u2014really", "it’s sponsored", "it's\u00a0sponsored", "IT'S AN AD",
  "this one's sponsored", "that's an ad", "that video's an ad", "Ad\u3000okay", "\ufeffAd", "Ad\u2028next",
  // Window edges: a negation exactly three words before a phrase (inside) and four (outside); "for sponsoring" nine
  // tokens after "thanks" (inside) and ten (outside); a negation in the sentence before.
  "not really a paid partnership", "not really just a paid partnership", "never ever been sponsored by them",
  "no, I was never really sponsored by them", "Not. Paid partnership", "thanks to a b c d e f for sponsoring",
  "thanks to a b c d e f g for sponsoring",
];
disclosureTexts.forEach((text, i) => disclosureCase(`text ${i}: ${text.slice(0, 40)}`, [turn(text)]));
disclosureCase("ts test: timed by word position", [turn("Ad. this is great", 0, 4000)]);
disclosureCase("several turns, sorted", [turn("and this is an ad", 9000, 11000), turn("#ad first", 1000, 2000), turn("no ads here", 2000, 5000)]);
disclosureCase("zero-length turn", [turn("this is sponsored", 4000, 4000)]);
const dvocab = ["this", "is", "an", "ad", "it's", "not", "sponsored", "by", "paid", "partnership", "partnered", "with", "in", "#ad", "thanks", "for", "sponsoring", "no", "never", "think", "I", "don't", "video", "a", "LaCroix", "really", "isn't", "hashtag", "add", "had", "Ad."];
const dseps = [" ", " ", " ", ", ", ". ", "! ", "? ", "-", " "];
for (let k = 0; k < 60; k++) {
  const words = range(0, int(2, 14)).map(() => pick(dvocab));
  let text = "";
  words.forEach((w, i) => (text += (i ? pick(dseps) : "") + w));
  const start = int(0, 20000);
  disclosureCase(`random ${k}`, [turn(text, start, start + int(0, 8000))]);
}

// --------------------------------------------------------------------------------------------------------------------
// Three stored reports

function reportCase(path: string, why: string) {
  const r = JSON.parse(readFileSync(join(ST, path), "utf8"));
  const inf: Info = { width: r.info.width, height: r.info.height, frameCount: r.info.frameCount, durationSec: r.info.durationSec };
  const detections: Row[] = (r.detections ?? []).map((d: any) => [d.frame, d.objectId, d.areaFraction]);
  const counted: number[] = (r.tracks ?? []).filter((t: any) => t.verdict === "match").map((t: any) => t.objectId);
  const mentions: Mention[] = (r.metrics?.mentions ?? []).map((x: any) => ({ startSec: x.startSec, endSec: x.endSec, text: x.text }));
  const disclosure: Disclosure | null = r.disclosure
    ? { spoken: r.disclosure.spoken.map((x: any) => ({ startSec: x.startSec, endSec: x.endSec, text: x.text })), onScreen: r.disclosure.onScreen ?? [], ...(r.disclosure.onScreenChecked === undefined ? {} : { onScreenChecked: r.disclosure.onScreenChecked }) }
    : null;
  const stored: Rule[] = (r.terms ?? []).map((t: any) => t.term.rule).filter(Boolean);
  const compute = computeCase(path, inf, detections, counted, mentions);
  const { metrics, samples, strict } = measure(detections, counted, inf, mentions);
  const rules = rulesCase(path, metrics, samples, disclosure, [...stored, ...battery(metrics, samples, strict, inf)], strict);
  return { source: path, why, compute, rules };
}
const reports = REPORTS.map((x) => reportCase(x.path, x.why));

// --------------------------------------------------------------------------------------------------------------------
// Write

function git(args: string[]): string | null {
  try {
    return execFileSync("git", ["-C", ST, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}
const provenance = {
  generator: "dev/ts-parity/dump.mts",
  screenTimeCommit: git(["rev-parse", "HEAD"]),
  sources: Object.fromEntries(
    SOURCES.map((rel) => {
      const committed = git(["rev-parse", `HEAD:${rel}`]);
      const working = git(["hash-object", "--", rel]); // without -w: hashes, writes nothing
      return [rel, { sha256: createHash("sha256").update(readFileSync(join(ST, rel))).digest("hex"), sameAsCommit: committed !== null && committed === working }];
    }),
  ),
  node: process.version,
  seed: SEED,
  fps: FPS,
};

/** One JSON value per line for the case lists, everything outside ASCII escaped (\uXXXX), no timestamps: re-running
 *  on the same TypeScript gives byte-identical files. */
function write(file: string, header: Record<string, unknown>, lists: Record<string, unknown[]>) {
  const ascii = (s: string) => s.replace(/[\u007f-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
  const parts = Object.entries(header).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v, null, 1).replace(/\n\s*/g, " ")}`);
  for (const [k, items] of Object.entries(lists)) parts.push(`${JSON.stringify(k)}: [\n${items.map((x) => JSON.stringify(x)).join(",\n")}\n]`);
  writeFileSync(join(OUT, file), ascii(`{\n${parts.join(",\n")}\n}\n`));
  const count = Object.values(lists).reduce((n, xs) => n + xs.length, 0);
  console.log(`wrote tests/parity/${file}: ${Object.entries(lists).map(([k, xs]) => `${xs.length} ${k}`).join(", ")} (${count} items)`);
}

mkdirSync(OUT, { recursive: true });
const about = (what: string) => ({ about: `${what} Inputs and outputs of Screen-Time's TypeScript, written by ${provenance.generator}; do not edit by hand.`, provenance });
write("compute.json", about("compute.ts: computeMetrics, frameSamples and framesToSegments."), { cases: computeCases, framesToSegments: framesToSegmentsCases });
write("rules.json", about("rules.ts: checkTerm, clock and clipSegments."), { cases: rulesCases, clock: clockCases, clipSegments: clipCases });
write("mentions.json", about("mentions.ts: tokenize, distinctNames, findMentions and findDisclosures."), {
  tokenize: tokenizeCases,
  distinctNames: distinctNamesCases,
  findMentions: mentionCases,
  findDisclosures: disclosureCases,
});
write("real_reports.json", about("Three stored Screen-Time reports, measured again by the current TypeScript."), { reports });
