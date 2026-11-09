"""Tool schemas: what the model reads to decide when and how to call each Screen-Time tool."""

RULE_KINDS = [
    "min_seconds_total",
    "min_seconds_within",
    "min_continuous",
    "min_area",
    "mentioned",
    "first_seen_before",
    "disclosed",
    "uncheckable",
]

TERM_ITEM = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The term exactly as the user wrote it."},
        "kind": {
            "type": "string",
            "enum": RULE_KINDS,
            "description": (
                "min_seconds_total: product on screen at least `seconds` in total. "
                "min_seconds_within: at least `seconds` on screen within the first `within_first_sec` seconds. "
                "min_continuous: one continuous appearance of at least `seconds`. "
                "min_area: product fills at least `fraction` (0-1) of the frame for at least `for_seconds`. "
                "mentioned: brand said aloud at least `min_times` times (optionally within the first `within_first_sec`). "
                "first_seen_before: product first appears before `seconds` (\"show it in the first 2 s\"). "
                "disclosed: the creator says it is an ad / sponsored. "
                "uncheckable: two requirements in one term, captions or hashtags, a phrase other than the brand name, "
                "lighting or tone, or anything else these kinds cannot express."
            ),
        },
        "seconds": {"type": "number"},
        "within_first_sec": {"type": "number"},
        "fraction": {"type": "number"},
        "for_seconds": {"type": "number"},
        "min_times": {"type": "integer"},
    },
    "required": ["text", "kind"],
}

MEASURE = {
    "name": "screentime_measure",
    "description": (
        "Measure how long a specific product is on screen in a video file, frame by frame (15 fps), and prove it. "
        "Tracks every object of the given kind (Meta SAM 3.1 with a Meta key, otherwise the user's own Hermes vision "
        "model on sampled frames), checks each one against the product description "
        "and photo so look-alikes (another flavour, another brand) are NOT counted, optionally listens for the brand "
        "name, checks contract terms as typed rules, and renders a proof reel. Use it for questions like 'how long is "
        "my product on screen', 'did the sponsor get 3 seconds', 'check this reel against the brief'. It returns "
        "measured numbers; never estimate on-screen time yourself. With a Meta key it costs about $0.003 per second of "
        "video plus about $0.02 per checked object (spend caps are enforced); without one it runs on the user's own "
        "Hermes models and says which. Takes 30-120 s for a short reel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video": {"type": "string", "description": "Path of the video file (an attachment's local path is fine)."},
            "product": {
                "type": "string",
                "description": "The exact product, as specific as the user gave it: brand, variant, colour, packaging "
                "(e.g. 'LaCroix Strawberry Peach can, pink and orange').",
            },
            "object_noun": {
                "type": "string",
                "description": "One to three plain words for the object's category that a tracker can find, never a "
                "brand name: 'drink can', 'bottle', 'cup', 'shoe', 'box'. If a run finds nothing, call again with a "
                "plainer noun.",
            },
            "photo": {"type": "string", "description": "Optional path of a photo of the product (improves the check)."},
            "brand_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names to listen for in the audio, e.g. ['LaCroix', 'La Croix'].",
            },
            "terms": {
                "type": "array",
                "items": TERM_ITEM,
                "description": "Contract terms to check, one entry per term, each turned into one typed rule. Leave "
                "unused number fields out.",
            },
        },
        "required": ["video", "product", "object_noun"],
    },
}

COMPARE = {
    "name": "screentime_compare",
    "description": (
        "Teardown: measure a short video shot by shot (cuts, first cut, shot lengths, pixel change, loudness, first "
        "sound, silences) and, when a reference is given, compare the two side by side with neutral differences "
        "('your first cut is 1.5 s later'). Uses ffmpeg only: no model, no key, no cost. Returns a card image. Use it "
        "for 'compare my draft with this reel', 'tear down this video', 'how fast does this cut'. It measures; it "
        "never predicts performance or scores a video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video": {"type": "string", "description": "Path of the user's video (their draft)."},
            "reference": {"type": "string", "description": "Optional path of the reference video to compare against."},
        },
        "required": ["video"],
    },
}

RUNS = {
    "name": "screentime_runs",
    "description": (
        "List this chat's earlier Screen-Time results, or fetch one by run id with every number, term, object verdict "
        "and the receipt. Free: answers follow-up questions about a past run without measuring again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "A run id such as st_1a2b3c4d. Omit to list recent runs."},
            "limit": {"type": "integer", "description": "How many recent runs to list (default 10)."},
        },
        "required": [],
    },
}
