import numpy as np
from numba import njit, prange

from evio.core.events import EventsBatch


@njit(cache=True, fastmath=True, parallel=True)
def get_mean_change(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    frame_height, frame_width = prev_frame.shape
    total_change = 0.0
    for row in prange(frame_height):
        row_total_change = 0.0
        for col in range(frame_width):
            row_total_change += float(curr_frame[row, col]) - float(
                prev_frame[row, col]
            )
        total_change += row_total_change
    return total_change / (frame_height * frame_width)


@njit(cache=True, fastmath=True, parallel=True)
def get_event_count(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    mean_change: float,
    threshold: float,
) -> int:
    frame_height, frame_weight = prev_frame.shape
    event_count = 0
    for row in prange(frame_height):
        row_count = 0
        for col in range(frame_weight):
            normalized_change = (
                float(curr_frame[row, col]) - float(prev_frame[row, col])
            ) - mean_change
            if (
                normalized_change >= threshold
                or normalized_change <= -threshold
            ):
                row_count += 1
        event_count += row_count
    return event_count


@njit(cache=True, fastmath=True)
def fill_event_arrays(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    mean_change: float,
    threshold: float,
    timestamp_us: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps: np.ndarray,
    polarities: np.ndarray,
) -> None:
    frame_height, frame_width = prev_frame.shape
    idx = 0
    for row in range(frame_height):
        for col in range(frame_width):
            normalized_change = (
                float(curr_frame[row, col]) - float(prev_frame[row, col])
            ) - mean_change
            if (
                normalized_change >= threshold
                or normalized_change <= -threshold
            ):
                x_coords[idx] = col
                y_coords[idx] = row
                polarities[idx] = 1 if normalized_change > 0 else 0
                timestamps[idx] = timestamp_us
                idx += 1


def get_events_from_frames(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    timestamp_us: int,
    threshold: float,
) -> EventsBatch:
    mean_change = get_mean_change(prev_frame, curr_frame)
    event_count = get_event_count(
        prev_frame, curr_frame, mean_change, threshold
    )

    x_coords = np.empty(event_count, dtype=np.int16)
    y_coords = np.empty(event_count, dtype=np.int16)
    polarities = np.empty(event_count, dtype=np.int8)
    timestamps = np.empty(event_count, dtype=np.int64)

    fill_event_arrays(
        prev_frame,
        curr_frame,
        mean_change,
        threshold,
        timestamp_us,
        x_coords,
        y_coords,
        timestamps,
        polarities,
    )
    return EventsBatch(
        timestamps,
        x_coords,
        y_coords,
        polarities,
    )
