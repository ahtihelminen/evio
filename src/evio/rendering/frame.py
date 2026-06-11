import time

import cv2
import numpy as np

from evio.core.events import BatchRange
from evio.core.pacer import Pacer
from evio.core.streaming_pacer import StreamingPacer


def get_window_dt8(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_window_evt3(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identical to play_dat, except unpacking EVT3-packed words."""
    # Index into time-sorted events
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)

    # EVT3 packing used in RawFileSource:
    #   bits 0..10  : x (11 bits)
    #   bit 11      : polarity
    #   bits 12..22 : y (11 bits)
    x_coords = (words & 0x7FF).astype(np.int32, copy=False)
    y_coords = ((words >> 12) & 0x7FF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 11) & 0x1).astype(bool, copy=False)

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (0, 0, 0),  # black — stationary
    on_color: tuple[int, int, int] = (0, 255, 0),  # green — brighter (BGR)
    off_color: tuple[int, int, int] = (0, 0, 255),  # red — dimmer (BGR)
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window

    valid = (
        (x_coords >= 0)
        & (x_coords < width)
        & (y_coords >= 0)
        & (y_coords < height)
    )

    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    polarities_on = polarities_on[valid]

    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (255, 255, 255),  # white on black base
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    # Effective speed factor based on total elapsed times
    actual_speed = rec_time_s / wall_time_s if wall_time_s > 0 else 0.0

    if pacer.force_speed:
        # Forced-speed mode: show target speed, instantaneous speed, and drop
        # statistics
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        # Free-running mode: show target vs actual effective speed plus
        # instantaneous speed
        first_row_str = (
            f"target speed={pacer.speed:.2f}x  actual speed={actual_speed:.2f}x"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_hud_stream(
    frame: np.ndarray,
    pacer: StreamingPacer,
    end_ts_us: int,
    *,
    color: tuple[int, int, int] = (255, 255, 255),  # white on black base
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    # If HUD clocks are not initialized yet, skip drawing
    try:
        wall_time_s = pacer.wall_elapsed()
        rec_time_s = pacer.recording_elapsed(end_ts_us)
        actual_speed = pacer.actual_playback_speed(end_ts_us)
    except ValueError:
        return

    inst_speed = pacer.instantaneous_speed
    inst_drop = pacer.instantaneous_drop_rate
    avg_drop = pacer.average_drop_rate

    first_row_str = (
        f"target={pacer.config.speed:.2f}x"
        f"  inst={inst_speed:.2f}x"
        f"  actual={actual_speed:4.2f}x"
        f"  wall={wall_time_s:7.3f}s"
        f"  rec={rec_time_s:7.3f}s"
    )

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    if pacer.config.force_speed:
        # Forced-speed mode: show target/instant speeds and drop statistics
        second_row_str = (
            f"  drops/ms={inst_drop:.2f}  avg drops/ms={avg_drop:.2f}"
        )
        # second row
        cv2.putText(
            frame,
            second_row_str,
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
