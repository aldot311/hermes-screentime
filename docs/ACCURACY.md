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
