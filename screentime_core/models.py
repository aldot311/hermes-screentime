"""Shared data contracts for every screentime_core module.

Times are seconds from the start of the normalized video. Every video Measure analyses is normalized to FPS frames per
second first, so a frame index divided by FPS is an exact time. Nothing in this package imports Hermes: the plugin glue
at the repository root adapts these types to tool results.

All dataclasses serialize with ``to_dict`` (plain JSON types) and load with ``from_dict``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, Optional, TypeVar, get_type_hints

FPS = 15
"""Analysis frame rate for Measure. frame / FPS = seconds."""

Verdict = Literal["match", "lookalike", "unclear"]
Visibility = Literal["clear", "partial", "blurred"]

T = TypeVar("T")


def to_dict(obj: Any) -> Any:
    """Dataclass (or list/dict of them) to plain JSON types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def from_dict(cls: type[T], data: Any) -> T:
    """Plain JSON types back to a dataclass, recursing into nested dataclass fields and lists of them."""
    if data is None or not is_dataclass(cls):
        return data  # type: ignore[return-value]
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name), data[f.name])
    return cls(**kwargs)  # type: ignore[arg-type]


def _coerce(tp: Any, value: Any) -> Any:
    if value is None or tp is None:
        return value
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ()) or ()
    if is_dataclass(tp):
        return from_dict(tp, value)
    if origin is list and args:
        return [_coerce(args[0], v) for v in value]
    if origin is tuple and args:
        return tuple(_coerce(a, v) for a, v in zip(args, value, strict=False))
    # Optional[X] / X | None: try the non-None member
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _coerce(inner[0], value)
    return value


# --------------------------------------------------------------------------------------------------------------------
# Video and detections


@dataclass
class VideoInfo:
    width: int
    height: int
    frame_count: int
    duration_sec: float
    fps: float = FPS
    has_audio: bool = True


@dataclass
class Box:
    """Pixel box in the normalized video. x2 and y2 are inclusive (SAM 3.1 convention)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection:
    """One object in one frame, reduced to what metrics and the proof reel need."""

    frame: int
    object_id: int
    box: Box
    area_fraction: float  # mask pixels / frame pixels, 0..1
    centroid_x: float  # mask centroid, normalized 0..1
    centroid_y: float
    confidence: Optional[float] = None


@dataclass
class TrackVerification:
    """The look-alike check's call on one tracked object (one SAM object id)."""

    object_id: int
    verdict: Verdict
    visibility: Visibility = "clear"
    reason: str = ""
    checks: list[dict[str, str]] = field(default_factory=list)  # [{"feature": ..., "seen": "yes|no|unclear"}]
    checked_by: str = ""  # e.g. "muse-spark-1.3", "hermes:<provider>/<model>", "replay:muse-spark-1.3", "unchecked"


@dataclass
class Segment:
    start_sec: float
    end_sec: float  # exclusive


@dataclass
class Mention:
    start_sec: float
    end_sec: float
    text: str
    approximate: bool = True  # placed inside a transcript turn, not a word timestamp


# --------------------------------------------------------------------------------------------------------------------
# Measure


@dataclass
class VideoMetrics:
    seconds_on_screen: float
    share_of_video: float  # seconds_on_screen / duration_sec
    first_seen_sec: Optional[float]
    longest_run_sec: float  # STRICT consecutive accepted frames (no gap bridging)
    longest_run_start_sec: Optional[float]
    avg_area_fraction: float  # mean, over frames where visible, of the largest accepted instance's mask fraction
    peak_area_fraction: float
    peak_area_sec: Optional[float]
    segments: list[Segment]  # navigation aid: accepted runs, gaps of up to 2 frames bridged (labelled as such)
    visible_frames: list[int]  # frame indices actually counted
    mentions: list[Mention] = field(default_factory=list)
    spoken_checked: Optional[bool] = None  # False = audio not transcribed (mentions unknown, not none)
    visible_areas: list[float] = field(default_factory=list)  # largest counted instance per visible frame (min_area)


@dataclass
class ContractTerm:
    text: str  # as the user wrote it
    rule: Optional[dict[str, Any]]  # typed rule (see rules.RULE_KINDS) or None when it cannot be checked


@dataclass
class TermResult:
    term: ContractTerm
    passed: Optional[bool]  # None = needs a person (not checkable, or evidence unknown)
    evidence: list[Segment]
    detail: str


@dataclass
class Disclosure:
    spoken: list[Mention] = field(default_factory=list)
    spoken_checked: Optional[bool] = None


@dataclass
class ReceiptCall:
    model: str
    purpose: str
    input: str
    output: str
    ms: int
    usd: float
    replayed: bool = False


@dataclass
class RunReceipt:
    run_id: str
    started_at: str  # ISO 8601 UTC
    fps: float
    models: dict[str, str]
    calls: list[ReceiptCall] = field(default_factory=list)

    @property
    def usd(self) -> float:
        return round(sum(c.usd for c in self.calls), 4)


@dataclass
class PromptAttempt:
    prompt: str
    objects: int
    matched: int
    sam_frames: int


@dataclass
class MeasureReport:
    run_id: str
    mode: Literal["measure"]
    video_path: str
    video_sha256: str
    info: VideoInfo
    product: str  # the user's product description
    prompt: str  # SAM noun phrase actually used
    engine: str  # "meta" | "replay"
    verifier: str  # "muse-spark-1.3" | "hermes:<model>" | "replay" | "none"
    verifier_validated: bool
    tracks: list[TrackVerification]
    metrics: VideoMetrics
    terms: list[TermResult]
    detections: list[Detection]  # every tracked box (all verdicts), for the proof reel
    verdict_by_object: dict[str, Verdict]  # str(object_id) -> verdict
    disclosure: Optional[Disclosure] = None
    prompt_attempts: list[PromptAttempt] = field(default_factory=list)
    receipt: Optional[RunReceipt] = None
    proof_path: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------------------------
# Teardown / Compare


@dataclass
class Shot:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    pixel_change: Optional[float]  # mean scdet mafd over the shot's frames, cut frames excluded


@dataclass
class TeardownReport:
    run_id: str
    mode: Literal["teardown"]
    video_path: str
    video_sha256: str
    info: VideoInfo
    cuts_sec: list[float]
    shots: list[Shot]
    first_cut_sec: Optional[float]  # None = single shot
    cuts_first_3s: int
    cuts_per_10s: list[int]  # windows [0,10), [10,20), ...
    median_shot_sec: Optional[float]
    longest_shot_sec: Optional[float]
    longest_shot_start_sec: Optional[float]
    integrated_lufs: Optional[float]
    loudness_range_lu: Optional[float]
    loudness_curve: list[tuple[float, float]]  # (time_sec, short-term LUFS) every 0.5 s
    first_sound_sec: Optional[float]
    silences: list[Segment]
    thresholds: dict[str, Any]  # every threshold used, printed with the results
    product_first_seen_sec: Optional[float] = None  # from a Measure run of the same file, when one exists
    brand_first_said_sec: Optional[float] = None
    card_path: Optional[str] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CompareRow:
    metric: str
    yours: Any
    theirs: Any
    diff: str  # neutral wording: "1.5 s later", "3 fewer"; never "better" or "worse"


@dataclass
class CompareReport:
    run_id: str
    mode: Literal["compare"]
    yours: TeardownReport
    theirs: TeardownReport
    rows: list[CompareRow]
    card_path: Optional[str] = None
