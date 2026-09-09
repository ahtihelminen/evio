import numpy as np

from evio.core.generate_events import (
    get_event_count,
    get_events_from_frames,
    get_mean_change,
)

# Synthetic 2x2 frames, hand-computed expected values:
#   change = curr - prev = [[10, -10], [0, 20]]
#   mean_change = (10 + -10 + 0 + 20) / 4 = 5.0
#   normalized_change = change - mean_change = [[5, -15], [-5, 15]]
#   threshold = 8 -> only (row=0, col=1)=-15 and (row=1, col=1)=15 cross it
PREV_FRAME = np.array([[0, 0], [0, 0]], dtype=np.int16)
CURR_FRAME = np.array([[10, -10], [0, 20]], dtype=np.int16)
THRESHOLD = 8.0
TIMESTAMP_US = 1000


def test_get_mean_change_computes_average_pixel_change() -> None:
    assert get_mean_change(PREV_FRAME, CURR_FRAME) == 5.0


def test_get_event_count_counts_pixels_past_threshold() -> None:
    mean_change = get_mean_change(PREV_FRAME, CURR_FRAME)
    assert (
        get_event_count(PREV_FRAME, CURR_FRAME, mean_change, THRESHOLD) == 2
    )


def test_get_events_from_frames_returns_expected_events() -> None:
    batch = get_events_from_frames(
        PREV_FRAME, CURR_FRAME, TIMESTAMP_US, THRESHOLD
    )

    assert len(batch.timestamps) == 2
    assert list(batch.x_coords) == [1, 1]
    assert list(batch.y_coords) == [0, 1]
    assert list(batch.polarities) == [0, 1]
    assert list(batch.timestamps) == [TIMESTAMP_US, TIMESTAMP_US]
