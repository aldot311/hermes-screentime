from screentime_core.models import Box, Detection, Segment, VideoMetrics, from_dict, to_dict


def test_roundtrip_nested():
    d = Detection(frame=3, object_id=1, box=Box(1, 2, 30, 40), area_fraction=0.1, centroid_x=0.5, centroid_y=0.4)
    m = VideoMetrics(
        seconds_on_screen=1.0, share_of_video=0.1, first_seen_sec=0.2, longest_run_sec=0.8, longest_run_start_sec=0.2,
        avg_area_fraction=0.1, peak_area_fraction=0.2, peak_area_sec=0.5, segments=[Segment(0.2, 1.2)],
        visible_frames=[3, 4],
    )
    assert from_dict(Detection, to_dict(d)) == d
    back = from_dict(VideoMetrics, to_dict(m))
    assert back == m and isinstance(back.segments[0], Segment)
