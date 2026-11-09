# Accuracy

Measured on 2026-11-09 with the plugin itself (`dev/accuracy.py`, raw results in `accuracy.json`), on the 11 LaCroix
test Reels from the Screen-Time app: 9 people, 2,800 frames at 15 fps, 1,931 of them with the product. The frame labels
were drafted by two independent Claude vision passes and then checked frame by frame by a person (all 11 clips). Two of
the Reels were filmed by the builder: one with the product and a look-alike (a Poppi can), one with only a look-alike (a
LaCroix Pamplemousse can, a different flavour). Product: "a pink LaCroix Strawberry Peach sparkling water can"; the
product photo is `fixtures/demo/product.jpg`; the tracker prompt is "drink can".

A frame counts as a true positive when the label says the product is on screen and the plugin counted it.

| | Precision | Recall | F1 |
|---|---|---|---|
| SAM 3.1 alone (every frame with a tracked box) | 95.0% | 99.6% | 97.2% |
| With the product check (what Screen-Time reports) | **98.1%** | **81.7%** | **89.1%** |

- Look-alike-only Reel: SAM alone counts 4.3 s; with the check, **0.0 s**.
- Contract term "on screen at least 3 seconds in total", against the labels: right on 8 Reels, sent to a person on 2,
  wrong on 1.
- Cost of the whole run: $0.63 of Meta API for 11 Reels (about $0.06 per Reel of 12 to 34 seconds).

## Where it is wrong

- Two Reels count 0.0 s although the product is on screen for 11.2 s and 12.1 s. In both, the can is small and blurred
  in the crops, and the product check answers "unclear" ("too small and blurry to read LaCroix branding"). Unclear
  objects are never counted, so the seconds are a floor, not a guess. A term that fails only because of them is not
  marked FAIL: it reads NEEDS A PERSON, with a note that counting the unconfirmed object would meet it. This costs recall
  (98.1% precision, 81.7% recall) in exchange for never paying for the wrong product.
- One Reel counts 3.1 s where the labels say 2.6 s, so the 3-second term passes where the labels say it should not.
  The extra half second is the can entering and leaving the frame.
- The product check is a model judgement. On borderline crops it can answer differently from run to run: the
  Screen-Time web app's own run of one of the two "unclear" Reels said "match, blurred".

## Compared with the Screen-Time app (Nov 3, same Reels, same labels)

| | Precision | Recall | F1 |
|---|---|---|---|
| App, SAM 3.1 alone | 95.0% | 99.3% | 97.1% |
| App, with Muse Spark | 98.4% | 91.1% | 94.6% |
| This plugin, with Muse Spark | 98.1% | 81.7% | 89.1% |

The tracker numbers match. The difference is the product check on the two blurred cans: the app's run called one of
them a match. The plugin also scores continuity strictly (no gap bridging) and uses the largest instance per frame for
size, so a contract term can only be met by frames that were actually seen.

## Without a Meta key: the user's own models

Measured on 2026-11-09 with the same 11 Reels, labels, product photo, noun and terms. The tracker is
`vision_track.py` (3 frames a second, 4 frames per call, boxes linked into objects by overlap); the same model then
checks each object. The models were called directly with the request Hermes's `ctx.llm.complete_structured` sends
(`dev/accuracy.py --model`, low reasoning). Raw results: `accuracy-gemini-3.8-flash.json`,
`accuracy-gpt-5.4-mini.json`. Speech was not scored here.

| Model | Tracker alone P / R | With the check P / R / F1 | Look-alike Reel | 3-second term |
|---|---|---|---|---|
| Gemini 3.8 Flash | 93.5% / 99.3% | **99.4% / 93.1% / 96.1%** | 5.3 s -> **0.0 s** | right 11 |
| GPT-5.4 mini | 91.1% / 99.2% | 94.9% / 74.9% / 83.7% | 4.3 s -> 4.3 s | right 9, to a person 1, wrong 1 |
| SAM 3.1 + Muse Spark (above) | 95.0% / 99.6% | 98.1% / 81.7% / 89.1% | 4.3 s -> 0.0 s | right 8, to a person 2, wrong 1 |

- Every model found the cans (tracker recall 99%). The difference is the product check.
- The two Reels where Muse Spark called the small, blurred can unclear (0.0 s counted): Gemini confirmed it on both
  (9.0 of 11.2 s and 12.0 of 12.1 s labelled); GPT-5.4 mini on neither fully (0.0 and 5.3 s).
- GPT-5.4 mini counted the different-flavour can on the look-alike-only Reel and rejected real cans on two others
  (5.3 and 7.7 s of about 12 to 14 s). Use a stronger model for `screentime_verify` if the check matters.
- Timing comes from samples, so each object's start and end are accurate to about a third of a second; sizes come from
  boxes and read larger than SAM's masks. A real Hermes run (Gemini 3.8 Flash for both tasks, OpenAI speech-to-text, no
  Meta key) took 37 s for a 21.7 s Reel.
- 11 Reels are a small test set, and the frame labels were drafted with Claude vision before a person checked them.
