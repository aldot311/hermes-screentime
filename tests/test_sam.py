"""sam.py against the Screen-Time app's own SAM fixtures and expected values (src/lib/sam/sam.test.ts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from screentime_core.errors import ScreentimeError
from screentime_core.models import Box
from screentime_core.sam import (
    box_footprint,
    group_tracks,
    mask_footprint,
    parse_sam_output,
    parse_sam_text,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sam"
DEMO = Path(__file__).parent.parent / "fixtures" / "demo"

# Real one_bit encodings from the app's fixtures (sse-fixtures.ts), each checked there to round-trip.
TOP_HALF_10 = "data=10,10,!!!!!*y:@2@r4Zkg"  # 10x10, top five rows set: 50 pixels
FULL_4x6 = "data=4,6,!!!!!'y:/tAxx"  # 4 rows x 6 columns, all set: 24 pixels
DIAGONAL_8 = "data=8,8,!!!!!(O1TG:){7"  # 8x8 diagonal: 8 pixels
EMPTY_3 = "data=3,3,!!!!!&!!!!!"  # 3x3, nothing set

# Frame 0 of the parser README example: a 320x334 clip, prompt "button", two objects, lossless masks.
README_LINE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)

W, H = 100, 50


def record(oid: int, box: tuple[int, int, int, int], mask: str, frame=(100, 50), c: float | None = None) -> str:
    conf = "" if c is None else f";c={c}"
    mask_conf = "" if c is None else f"c={c};"
    return (
        f"{oid}<|box;x1={box[0]};y1={box[1]};x2={box[2]};y2={box[3]};w={frame[0]};h={frame[1]}{conf}|>"
        f"<|mask;x=0;y=0;{mask_conf}{mask}|>"
    )


def line(frame: int, *records: str) -> str:
    return f"<{frame}f>" + ",".join(records) + "\n"


# Frames 0, 1 and 4: frames 2 and 3 had nothing visible, so the API emitted no line for them.
MULTI = "".join(
    [
        line(0, record(0, (10, 20, 19, 29), TOP_HALF_10, c=0.9), record(2, (60, 10, 65, 13), FULL_4x6, c=0.8)),
        line(1, record(0, (10, 20, 19, 29), TOP_HALF_10, c=0.85)),
        line(4, record(2, (60, 10, 67, 17), DIAGONAL_8)),
    ]
)


def test_several_objects_over_several_frames():
    parsed = parse_sam_output(MULTI, W, H)
    assert parsed.diagnostics == []
    dets = parsed.detections
    assert [(d.frame, d.object_id) for d in dets] == [(0, 0), (0, 2), (1, 0), (4, 2)]
    assert dets[0].box == Box(x1=10, y1=20, x2=19, y2=29)  # inclusive, as on the wire
    assert dets[0].confidence == 0.9
    assert dets[3].confidence is None  # no c field is unknown, not zero
    tracks = group_tracks(dets)
    assert list(tracks) == [0, 2]
    assert [d.frame for d in tracks[0]] == [0, 1]
    assert [d.frame for d in tracks[2]] == [0, 4]


def test_skipped_frames_keep_their_index():
    frames = sorted({d.frame for d in parse_sam_text(MULTI, W, H)})
    assert frames == [0, 1, 4]


def test_area_and_centroid_from_known_masks():
    dets = parse_sam_text(MULTI, W, H)
    # Box 10..19 x 20..29, mask rows 0-4 set: 50 px of 5000, centre at (15, 22.5).
    assert dets[0].area_fraction == pytest.approx(50 / 5000)
    assert dets[0].centroid_x == pytest.approx(0.15)
    assert dets[0].centroid_y == pytest.approx(0.45)
    # Full 6x4 block at 60..65 x 10..13: 24 px, centre at (63, 12).
    assert dets[1].area_fraction == pytest.approx(24 / 5000)
    assert (dets[1].centroid_x, dets[1].centroid_y) == pytest.approx((0.63, 0.24))
    # 8x8 diagonal at 60..67 x 10..17: 8 px, mean col/row 3.5, centre at (64, 14).
    assert dets[3].area_fraction == pytest.approx(8 / 5000)
    assert (dets[3].centroid_x, dets[3].centroid_y) == pytest.approx((0.64, 0.28))


def test_zero_matches_is_an_empty_result():
    assert parse_sam_output("", W, H).detections == []
    assert parse_sam_output("", W, H).diagnostics == []


def test_falls_back_to_the_box_when_a_mask_cannot_be_decoded():
    parsed = parse_sam_output(line(3, record(5, (0, 0, 9, 4), "data=5,10,!zzzz")), W, H)
    assert len(parsed.detections) == 1
    d = parsed.detections[0]
    assert d.area_fraction == pytest.approx(50 / 5000)
    assert (d.centroid_x, d.centroid_y) == pytest.approx((0.05, 0.05))
    assert "invalid_mask_payload" in parsed.diagnostics[0]


def test_an_empty_mask_has_zero_area_and_the_box_centre():
    d = parse_sam_text(line(0, record(1, (30, 30, 32, 32), EMPTY_3)), W, H)[0]
    assert d.area_fraction == 0
    assert (d.centroid_x, d.centroid_y) == pytest.approx((0.315, 0.63))


def test_readme_lossless_example():
    parsed = parse_sam_output(README_LINE, 320, 334)
    assert parsed.diagnostics == []  # masks decoded, not the box fallback
    assert [d.object_id for d in parsed.detections] == [0, 1]
    assert parsed.detections[0].box == Box(x1=211, y1=228, x2=270, y2=254)
    for d in parsed.detections:
        assert 0 < d.area_fraction <= (60 * 27) / (320 * 334)
        assert 228 / 334 < d.centroid_y < 255 / 334


def test_flags_boxes_reported_in_another_frame_size():
    parsed = parse_sam_output(MULTI, 200, 100)
    assert len(parsed.detections) == 4
    assert parsed.diagnostics == [
        "warning frame_size_mismatch: boxes are in a 100x50 frame but info is 200x100"
    ]


def test_rejects_a_missing_frame_size():
    with pytest.raises(ScreentimeError) as err:
        parse_sam_text("", 0, 50)
    assert err.value.code == "bad_input"


def test_a_later_record_for_the_same_frame_and_object_replaces_the_earlier_one():
    text = line(0, record(7, (0, 0, 9, 9), TOP_HALF_10)) + line(0, record(7, (10, 20, 19, 29), TOP_HALF_10, c=0.5))
    dets = parse_sam_text(text, W, H)
    assert len(dets) == 1
    assert dets[0].box == Box(x1=10, y1=20, x2=19, y2=29)
    assert dets[0].confidence == 0.5


def test_backward_frames_come_out_sorted():
    # With propagation in both directions SAM emits later frames first, then earlier ones in descending order.
    text = line(5, record(1, (0, 0, 9, 9), TOP_HALF_10)) + line(2, record(1, (0, 0, 9, 9), TOP_HALF_10))
    dets = parse_sam_text(text, W, H)
    assert [d.frame for d in dets] == [2, 5]
    assert [d.frame for d in group_tracks(reversed(dets))[1]] == [2, 5]


def test_plain_text_lines_are_reported_not_parsed():
    parsed = parse_sam_output("hello\n" + line(0, record(1, (0, 0, 9, 9), TOP_HALF_10)), W, H)
    assert len(parsed.detections) == 1
    assert any("ignored_text" in d for d in parsed.diagnostics)


def test_mask_footprint_scales_a_raster_smaller_than_its_box():
    # 2x2 raster over a 4x4 box: one set pixel stands for a 2x2 block.
    area, cx, cy = mask_footprint(bytes([0, 0, 0, 1]), 2, 2, (0, 0, 4, 4), 10, 10)
    assert area == pytest.approx(4 / 100)
    assert (cx, cy) == pytest.approx((0.3, 0.3))


def test_box_footprint_is_zero_not_an_error_for_a_zero_frame():
    assert box_footprint((0, 0, 0, 0), 0, 0) == (0.0, 0.0, 0.0)


def test_matches_the_app_parser_on_a_recorded_stream():
    """The TS app's parseSamVideo, run on the recorded demo stream (tests/fixtures/sam/ts_dump.mts), gave these
    detections; the Python port must give the same ones from the same recording."""
    expected_path = FIXTURES / "demo_ts_detections.json"
    sam_text = DEMO / "sample.sam.txt"
    meta = DEMO / "sample.sam.json"
    if not (expected_path.exists() and sam_text.exists() and meta.exists()):
        pytest.skip("recorded demo fixtures not present")
    info = json.loads(meta.read_text())
    expected = json.loads(expected_path.read_text())
    got = parse_sam_text(sam_text.read_text(), info["width"], info["height"])
    assert len(got) == len(expected) > 0
    for d, e in zip(got, expected):
        assert (d.frame, d.object_id) == (e["frame"], e["objectId"])
        assert (d.box.x1, d.box.y1, d.box.x2, d.box.y2) == (e["box"]["x1"], e["box"]["y1"], e["box"]["x2"], e["box"]["y2"])
        # Same arithmetic in the same order: equal to the last bit, allow 1e-12 for safety.
        assert d.area_fraction == pytest.approx(e["areaFraction"], abs=1e-12)
        assert d.centroid_x == pytest.approx(e["centroid"]["x"], abs=1e-12)
        assert d.centroid_y == pytest.approx(e["centroid"]["y"], abs=1e-12)
        assert d.confidence == e.get("confidence")
