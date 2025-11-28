from collections.abc import Iterable

import numpy as np
from numba import njit, types
from numba.typed import List
from numpy.typing import NDArray

from evio.core.events import EventsBatch


@njit
def window_events(
    timestamps: NDArray[np.int64],
    x_coords: NDArray[np.int16],
    y_coords: NDArray[np.int16],
    polarities: NDArray[np.int8],
    window_us: int,
    # Pylance cries about calling a module with numba lists, hence the ignores
    timestamps_buffer_jit: List[int],  # type: ignore
    x_coords_buffer_jit: List[int],  # type: ignore
    y_coords_buffer_jit: List[int],  # type: ignore
    polarities_buffer_jit: List[int],  # type: ignore
    window_start_ts: int,
    window_start_idx: int,
    global_idx: int,
) -> tuple[
    # Pylance cries about calling a module with numba lists, hence the ignores
    List[NDArray[np.int64]],  # type: ignore
    List[NDArray[np.int16]],  # type: ignore
    List[NDArray[np.int16]],  # type: ignore
    List[NDArray[np.int8]],  # type: ignore
    int,
    int,
    int,
]:
    # Pylance cries about calling a module with numba lists, hence the ignores
    timestamps_jit = List()  # type: ignore
    x_coords_jit = List()  # type: ignore
    y_coords_jit = List()  # type: ignore
    polarities_jit = List()  # type: ignore

    n_timestamps = timestamps.shape[0]

    for i in range(n_timestamps):
        timestamp = int(timestamps[i])

        # initialize window start
        if window_start_ts < 0:
            window_start_ts = timestamp
            window_start_idx = global_idx

        # complete window
        if (
            timestamp - window_start_ts >= window_us
            and len(timestamps_buffer_jit) > 0
        ):
            n_timestamps_buffer = len(timestamps_buffer_jit)
            timestamps_buffer_np = np.empty(n_timestamps_buffer, np.int64)
            x_coords_buffer_np = np.empty(n_timestamps_buffer, np.int16)
            y_coords_buffer_np = np.empty(n_timestamps_buffer, np.int16)
            polarities_buffer_np = np.empty(n_timestamps_buffer, np.int8)

            # copy buffers from Numba Lists to np arrays
            for j in range(n_timestamps_buffer):
                timestamps_buffer_np[j] = timestamps_buffer_jit[j]
                x_coords_buffer_np[j] = x_coords_buffer_jit[j]
                y_coords_buffer_np[j] = y_coords_buffer_jit[j]
                polarities_buffer_np[j] = polarities_buffer_jit[j]

            timestamps_jit.append(timestamps_buffer_np)
            x_coords_jit.append(x_coords_buffer_np)
            y_coords_jit.append(y_coords_buffer_np)
            polarities_jit.append(polarities_buffer_np)

            timestamps_buffer_jit.clear()
            x_coords_buffer_jit.clear()
            y_coords_buffer_jit.clear()
            polarities_buffer_jit.clear()

            window_start_ts = timestamp
            window_start_idx = global_idx

        # append event
        timestamps_buffer_jit.append(timestamp)
        x_coords_buffer_jit.append(x_coords[i])
        y_coords_buffer_jit.append(y_coords[i])
        polarities_buffer_jit.append(polarities[i])
        global_idx += 1

    return (
        timestamps_jit,
        x_coords_jit,
        y_coords_jit,
        polarities_jit,
        window_start_ts,
        window_start_idx,
        global_idx,
    )


@njit
def flush_tail(
    # Pylance cries about calling a module with numba lists, hence the ignores
    timestamps_buffer_jit: List[int],  # type: ignore
    x_coords_buffer_jit: List[int],  # type: ignore
    y_coords_buffer_jit: List[int],  # type: ignore
    polarities_buffer_jit: List[int],  # type: ignore
    start_idx: int,
) -> tuple[
    # Pylance cries about calling a module with numba lists, hence the ignores
    List[NDArray[np.int64]],  # type: ignore
    List[NDArray[np.int16]],  # type: ignore
    List[NDArray[np.int16]],  # type: ignore
    List[NDArray[np.int8]],  # type: ignore
]:
    # Pylance cries about calling a module with numba lists, hence the ignores
    timestampts_jit = List()  # type: ignore
    timestamps_jit = List()  # type: ignore
    y_coords_jit = List()  # type: ignore
    polarities_jit = List()  # type: ignore

    if len(timestamps_buffer_jit) > 0:
        n_timestamps_buffer = len(timestamps_buffer_jit)

        timestamps_buffer_np = np.empty(n_timestamps_buffer, np.int64)
        x_coords_buffer_np = np.empty(n_timestamps_buffer, np.int16)
        y_coords_buffer_np = np.empty(n_timestamps_buffer, np.int16)
        polarities_buffer_np = np.empty(n_timestamps_buffer, np.int8)

        # copy buffers from Numba Lists to np arrays
        for j in range(n_timestamps_buffer):
            timestamps_buffer_np[j] = timestamps_buffer_jit[j]
            x_coords_buffer_np[j] = x_coords_buffer_jit[j]
            y_coords_buffer_np[j] = y_coords_buffer_jit[j]
            polarities_buffer_np[j] = polarities_buffer_jit[j]

        timestampts_jit.append(timestamps_buffer_np)
        timestamps_jit.append(x_coords_buffer_np)
        y_coords_jit.append(y_coords_buffer_np)
        polarities_jit.append(polarities_buffer_np)

        timestamps_buffer_jit.clear()
        x_coords_buffer_jit.clear()
        y_coords_buffer_jit.clear()
        polarities_buffer_jit.clear()

    return (
        timestampts_jit,
        timestamps_jit,
        y_coords_jit,
        polarities_jit,
    )


class WindowBatchProcessor:
    """Stateful time-windowing for decoded EVT3 batches."""

    def __init__(self, window_length: int) -> None:
        self.window_length = window_length

        # numba.typed.List buffers
        self.timestamps_buffer_jit = List.empty_list(types.int64)
        self.x_coords_buffer_jit = List.empty_list(types.int16)
        self.y_coords_buffer_jit = List.empty_list(types.int16)
        self.polarities_buffer_jit = List.empty_list(types.int8)

        self.window_start_ts = -1
        self.window_start_idx = 0
        self.global_idx = 0

    def process(
        self, batches: Iterable[EventsBatch], *, final: bool = False
    ) -> Iterable[EventsBatch]:
        for batch in batches:
            (
                timestamps_jit,
                x_coords_jit,
                y_coords_jit,
                polarities_jit,
                self.window_start_ts,
                self.window_start_idx,
                self.global_idx,
            ) = window_events(
                batch.timestamps,
                batch.x_coords,
                batch.y_coords,
                batch.polarities,
                self.window_length,
                self.timestamps_buffer_jit,
                self.x_coords_buffer_jit,
                self.y_coords_buffer_jit,
                self.polarities_buffer_jit,
                self.window_start_ts,
                self.window_start_idx,
                self.global_idx,
            )

            for i in range(len(timestamps_jit)):
                yield EventsBatch(
                    timestamps_jit[i],
                    x_coords_jit[i],
                    y_coords_jit[i],
                    polarities_jit[i],
                )

        if final:
            (
                timestamps_jit,
                x_coords_jit,
                y_coords_jit,
                polarities_jit,
            ) = flush_tail(
                self.timestamps_buffer_jit,
                self.x_coords_buffer_jit,
                self.y_coords_buffer_jit,
                self.polarities_buffer_jit,
                self.window_start_idx,
            )

            for i in range(len(timestamps_jit)):
                yield EventsBatch(
                    timestamps_jit[i],
                    x_coords_jit[i],
                    y_coords_jit[i],
                    polarities_jit[i],
                )
