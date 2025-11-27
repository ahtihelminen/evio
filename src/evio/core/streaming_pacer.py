import time
from collections.abc import Iterable, Iterator

from evio.core.events import EventsBatch


class StreamingPacer:
    __slots__ = (
        "_avg_drop_rate",
        "_dropped_batches",
        "_dropped_pending",
        "_e_start",  # pacing event anchor (mutable)
        "_emitted_batches",
        "_hud_e_start",  # HUD event anchor (immutable)
        "_hud_t_start",  # HUD wall anchor (immutable)
        "_inst_drop_rate",
        "_inst_speed",
        "_last_emit_ts_us",
        "_last_wall_emit_s",
        "_t_start",  # pacing wall anchor (mutable)
        "drop_tolerance_s",
        "force_speed",
        "speed",
    )

    def __init__(
        self,
        speed: float,
        force_speed: bool,
        drop_tolerance_s: float = 0.0,
    ) -> None:
        self.speed = max(float(speed), 1e-9)
        self.force_speed = force_speed
        self.drop_tolerance_s = max(0.0, float(drop_tolerance_s))

        # pacing anchors (can be reset)
        self._t_start: float | None = None
        self._e_start: int | None = None

        # HUD anchors (set once, never changed)
        self._hud_t_start: float | None = None
        self._hud_e_start: int | None = None

        # stats
        self._dropped_batches = 0
        self._emitted_batches = 0
        self._dropped_pending = 0
        self._last_emit_ts_us: int | None = None
        self._last_wall_emit_s: float | None = None
        self._inst_drop_rate = 0.0
        self._avg_drop_rate = 0.0
        self._inst_speed = 0.0

    @property
    def hud_wall_start_s(self) -> float | None:
        """Wall-clock anchor for HUD (perf_counter domain)."""
        return self._hud_t_start

    @property
    def hud_event_start_us(self) -> int | None:
        """Event-time anchor for HUD (µs)."""
        return self._hud_e_start

    # HUD helpers that never re-anchor (useful for long-term drift display)
    def hud_wall_elapsed_s(self) -> float | None:
        """Elapsed wall-clock time since first batch was seen."""
        if self._hud_t_start is None:
            return None
        return time.perf_counter() - self._hud_t_start

    def hud_recording_elapsed_s(
        self, end_ts_us: int | None = None
    ) -> float | None:
        """Elapsed recording time (event domain) since first batch was seen."""
        if self._hud_e_start is None:
            return None

        ts_us = end_ts_us if end_ts_us is not None else self._last_emit_ts_us
        if ts_us is None:
            return None

        return max(0.0, (ts_us - self._hud_e_start) / 1e6)

    def hud_actual_speed(self, end_ts_us: int | None = None) -> float | None:
        """Long-term effective speed (recording elapsed / wall elapsed)."""
        wall = self.hud_wall_elapsed_s()
        rec = self.hud_recording_elapsed_s(end_ts_us)
        if wall is None or wall <= 0 or rec is None:
            return None
        return rec / wall

    # --- small helpers for StreamingBatch timestamps ---

    @staticmethod
    def _start_ts(batch: EventsBatch) -> int:
        return batch.timestamps[0]

    @staticmethod
    def _end_ts(batch: EventsBatch) -> int:
        return batch.timestamps[-1]

    # --- timing internals ---

    def _target_wall_s(self, end_ts_us: int) -> float:
        # event elapsed (s) scaled by speed factor
        return max(0.0, (end_ts_us - (self._e_start or 0)) / 1e6) / self.speed

    def _sleep_until(self, target_rel_s: float) -> None:
        # Sleep in chunks so UI can be pumped and OS timers stay accurate
        while True:
            now_s = time.perf_counter() - (self._t_start or time.perf_counter())
            dt = target_rel_s - now_s
            if dt <= 0:
                break
            time.sleep(min(0.05, dt))

    def _snapshot_and_yield(self, batch: EventsBatch) -> Iterator[EventsBatch]:
        """Emit batch and compute instantaneous + average drop rates."""
        self._emitted_batches += 1

        # Compute drop rate (drops per second of event time)
        if self._last_emit_ts_us is not None:
            dt_ms = max(
                (self._end_ts(batch) - self._last_emit_ts_us) / 1e3, 1e-9
            )
            self._inst_drop_rate = self._dropped_pending / dt_ms
        else:
            self._inst_drop_rate = 0.0

        # Compute instantaneous playback speed (event-time / wall-time)
        now_wall = time.perf_counter()
        last_ts = (
            self._last_emit_ts_us
            if self._last_emit_ts_us is not None
            else self._start_ts(batch)
        )
        dt_event_s = max((self._end_ts(batch) - last_ts) / 1e6, 0.0)
        dt_wall_s = max(
            now_wall
            - (
                self._last_wall_emit_s
                if self._last_wall_emit_s is not None
                else self._t_start or now_wall
            ),
            1e-9,
        )
        self._inst_speed = dt_event_s / dt_wall_s

        # Update running average
        n = self._emitted_batches
        self._avg_drop_rate += (self._inst_drop_rate - self._avg_drop_rate) / n

        # Reset per-window accumulator and update last timestamp
        self._dropped_pending = 0
        self._last_emit_ts_us = self._end_ts(batch)
        self._last_wall_emit_s = now_wall

        yield batch

    # --- public pacing API ---

    def pace(self, batches: Iterable[EventsBatch]) -> Iterator[EventsBatch]:
        iterator = iter(batches)
        try:
            batch = next(iterator)
        except StopIteration:
            return

        if self._t_start is None:
            now = time.perf_counter()
            first_ts = self._start_ts(batch)

            # pacing anchors
            self._t_start = now
            self._e_start = first_ts
            self._last_emit_ts_us = first_ts

            # HUD anchors: set once, never touched again
            if self._hud_t_start is None:
                self._hud_t_start = now
            if self._hud_e_start is None:
                self._hud_e_start = first_ts

        while True:
            target_s = self._target_wall_s(self._end_ts(batch))
            now_s = time.perf_counter() - self._t_start
            lag_s = now_s - target_s

            if self.force_speed and lag_s > 0:
                if self._emitted_batches == 0:
                    # One-time resync at startup so we don't drop everything
                    # that arrived before playback began.
                    now = time.perf_counter()
                    start_ts = self._start_ts(batch)
                    self._t_start = now
                    self._e_start = start_ts
                    self._last_emit_ts_us = start_ts

                    target_s = 0.0
                    now_s = 0.0
                    lag_s = 0.0
                else:
                    # Drop overdue batches until we are back within tolerance,
                    # but cap the burst to avoid blocking forever when the
                    # source cannot keep up.
                    max_drop_burst = 4
                    drops = 0
                    while (
                        lag_s > self.drop_tolerance_s and drops < max_drop_burst
                    ):
                        self._dropped_batches += 1
                        self._dropped_pending += 1
                        drops += 1

                        try:
                            batch = next(iterator)
                        except StopIteration:
                            # Last one: still update stats based on what we
                            # *would* emit
                            yield from self._snapshot_and_yield(batch)
                            return

                        target_s = self._target_wall_s(self._end_ts(batch))
                        now_s = time.perf_counter() - self._t_start
                        lag_s = now_s - target_s

                    if (
                        lag_s > self.drop_tolerance_s
                        and drops >= max_drop_burst
                    ):
                        # Still behind after a drop burst: re-anchor pacing so
                        # playback can resume instead of stalling.
                        now = time.perf_counter()
                        start_ts = self._start_ts(batch)
                        self._t_start = now
                        self._e_start = start_ts
                        self._last_emit_ts_us = start_ts

                        target_s = 0.0
                        now_s = 0.0
                        lag_s = 0.0

            # We're early or within tolerance: optionally sleep, then emit
            if target_s > now_s:
                self._sleep_until(target_s)
            yield from self._snapshot_and_yield(batch)

            try:
                batch = next(iterator)
            except StopIteration:
                return

    # --- helper stats API (optional) ---

    @property
    def dropped_batches(self) -> int:
        """Number of batches skipped because they were overdue."""
        return self._dropped_batches

    @property
    def emitted_batches(self) -> int:
        """Number of batches actually yielded downstream."""
        return self._emitted_batches

    @property
    def instantaneous_drop_rate(self) -> float:
        """Drop rate for the most recently emitted window (drops / ms)."""
        return self._inst_drop_rate

    @property
    def instantaneous_speed(self) -> float:
        """Playback speed for the most recently emitted window
        (event-time/ wall-time)."""
        return self._inst_speed

    @property
    def average_drop_rate(self) -> float:
        """Running average drop rate across all emitted windows (drops / ms)."""
        return self._avg_drop_rate
