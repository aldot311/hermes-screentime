---
name: screentime
description: How to use the Screen-Time tools well - measuring a product's seconds on screen against a brief, turning contract terms into typed rules, and comparing a draft with a reference reel.
---

# Screen-Time

Three tools. All numbers come from the tools; never estimate on-screen time, cut timing or loudness yourself.

## screentime_measure: "how long is my product on screen?"
- `product`: as specific as the user gave it (brand, flavour, colour, packaging). The check rejects other flavours and
  other brands, so specificity matters.
- `object_noun`: one to three plain words a tracker can find, never a brand: "drink can", "bottle", "cup", "box",
  "shoe", "tube", "bag". If the run reports that nothing was found, call it once more with a plainer noun for the
  object's shape (a tumbler held like a cup is a "cup").
- `photo`: pass the product photo whenever the user attached one.
- `brand_names`: the spoken forms to listen for ("LaCroix", "La Croix").
- `terms`: one entry per term the user wrote, each turned into exactly one rule:
  - "at least 3 seconds on screen" -> min_seconds_total, seconds 3
  - "4 seconds of product time in the opening 8 seconds" -> min_seconds_within, seconds 4, within_first_sec 8
  - "show the bottle in the first 2 seconds" -> first_seen_before, seconds 2
  - "one continuous shot of at least 2 seconds" -> min_continuous, seconds 2
  - "fills a quarter of the frame for 2 seconds" -> min_area, fraction 0.25, for_seconds 2
  - "say the brand name in the first 4 seconds" -> mentioned, min_times 1, within_first_sec 4
  - "say it's an ad" -> disclosed
  - two requirements in one term, captions or hashtags, a phrase other than the brand name, lighting, tone, or when a
    disclosure must appear -> uncheckable (it is reported for a person)
- Reply with the numbers exactly as returned, each term's result with its time, the rejected look-alikes and why, and
  the `media_tag` line on its own line so the proof reel is attached.
- Cost: about $0.003 per second of video plus about $0.02 per checked object on the user's Meta key; caps are
  enforced by the plugin. Without a key, offer `/screentime demo`.

## screentime_compare: "compare my draft with this reel"
- `video` is the user's draft; `reference` the take they are comparing against. Free, local, ffmpeg only.
- Describe differences neutrally ("your first cut lands 1.5 s later", "3 fewer cuts in the first 3 seconds"). Never
  call one version better and never predict views or retention.
- If the same file was measured before in this chat, product and brand timing are included at no cost.

## screentime_runs
- Answers follow-ups about earlier results ("what was the longest run?", "which object was rejected?") without
  measuring again.
