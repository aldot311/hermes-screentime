// Dumps the Screen-Time app's own parse of a recorded SAM stream (src/lib/sam/parse.ts parseSamVideo), so
// tests/test_sam.py can check that screentime_core.sam gives the same detections. Hand-run; reads the app read-only:
//   ~/Projects/aetrai/screen-time/node_modules/.bin/tsx tests/fixtures/sam/ts_dump.mts \
//     tests/fixtures/meta/sam_stream.sse.gz 540 960 > tests/fixtures/sam/demo_ts_detections.json
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

const app = process.env.SCREEN_TIME_DIR ?? `${process.env.HOME}/Projects/aetrai/screen-time`;
const { parseSamVideo } = await import(`${app}/src/lib/sam/parse.ts`);

const [file, width, height] = process.argv.slice(2);
const raw = readFileSync(file);
const sse = (file.endsWith(".gz") ? gunzipSync(raw) : raw).toString("utf8");
const result = await parseSamVideo(sse, { width: Number(width), height: Number(height) });
if (result.diagnostics.length) console.error(result.diagnostics.join("\n"));
process.stdout.write(`${JSON.stringify(result.detections)}\n`);
