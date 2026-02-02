from collections import deque
from collections.abc import Iterable

import numpy as np

from evio.core.events import EventsBatch


class RollingWindowBatchProcessor:
    """Stateful rolling time-windowing for decoded batches.

    Windows are left-aligned: [t, t + window_length), advancing by step_us.
    Empty windows are skipped (no EventsBatch emitted).
    """

    def __init__(self, window_length: int, step_us: int) -> None:
        self.window_length = window_length
        self.step_us = step_us
        self._chunks: deque[EventsBatch] = deque()
        self._head_offset = 0
        self._next_start_ts: int | None = None

    def _drop_before(self, timestamp_us: int) -> None:
        while self._chunks:
            head = self._chunks[0]
            timestamps = head.timestamps
            if timestamps.size == 0:
                self._chunks.popleft()
                self._head_offset = 0
                continue

            if timestamps[-1] < timestamp_us:
                self._chunks.popleft()
                self._head_offset = 0
                continue

            if self._head_offset:
                if timestamps[self._head_offset] < timestamp_us:
                    local = np.searchsorted(
                        timestamps[self._head_offset :],
                        timestamp_us,
                        side="left",
                    )
                    self._head_offset += int(local)
            elif timestamps[0] < timestamp_us:
                self._head_offset = np.searchsorted(
                    timestamps, timestamp_us, side="left"
                )

            if self._head_offset >= timestamps.shape[0]:
                self._chunks.popleft()
                self._head_offset = 0
                continue
            break

    def _collect_window(self, start_ts: int, end_ts: int) -> EventsBatch | None:
        if not self._chunks:
            return None

        timestamps_parts: list[np.ndarray] = []
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        p_parts: list[np.ndarray] = []

        for idx, chunk in enumerate(self._chunks):
            timestamps = chunk.timestamps
            if timestamps.size == 0:
                continue

            base_offset = self._head_offset if idx == 0 else 0
            if timestamps[-1] < start_ts:
                continue

            start_idx = base_offset + np.searchsorted(
                timestamps[base_offset:], start_ts, side="left"
            )

            if start_idx >= timestamps.shape[0]:
                continue

            end_idx = np.searchsorted(timestamps, end_ts, side="left")
            if end_idx <= start_idx:
                if timestamps[0] >= end_ts:
                    break
                continue

            timestamps_parts.append(timestamps[start_idx:end_idx])
            x_parts.append(chunk.x_coords[start_idx:end_idx])
            y_parts.append(chunk.y_coords[start_idx:end_idx])
            p_parts.append(chunk.polarities[start_idx:end_idx])

            if end_idx < timestamps.shape[0] or timestamps[-1] >= end_ts:
                break

        if not timestamps_parts:
            return None

        return EventsBatch(
            np.concatenate(timestamps_parts),
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            np.concatenate(p_parts),
        )

    def _emit_ready(self, *, final: bool) -> Iterable[EventsBatch]:
        if self._next_start_ts is None:
            return

        while self._chunks:
            last_ts = int(self._chunks[-1].timestamps[-1])
            if final:
                if self._next_start_ts > last_ts:
                    return
            else:
                if last_ts < self._next_start_ts + self.window_length:
                    return

            start_ts = self._next_start_ts
            end_ts = start_ts + self.window_length

            self._drop_before(start_ts)
            if not self._chunks:
                return

            window = self._collect_window(start_ts, end_ts)
            if window is not None:
                yield window

            self._next_start_ts += self.step_us

    def process(
        self, batches: Iterable[EventsBatch], *, final: bool = False
    ) -> Iterable[EventsBatch]:
        for batch in batches:
            if batch.timestamps.size == 0:
                continue

            if self._next_start_ts is None:
                self._next_start_ts = int(batch.timestamps[0])

            self._chunks.append(batch)
            yield from self._emit_ready(final=False)

        if final:
            yield from self._emit_ready(final=True)
            self._chunks.clear()
            self._head_offset = 0
            self._next_start_ts = None
