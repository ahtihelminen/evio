from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from evio.core.events import EventsBatch


@dataclass(frozen=True)
class PacerConfig:
    """Immutable configuration for the streaming pacer.

    All time-related config is stored in microseconds.
    """

    speed: float  # playback speed factor
    force_speed: bool  # drop when behind
    drop_tolerance: int = 0  # allowed lag in µs before dropping


@dataclass
class PacerState:
    """
    Pacer state. All time values are in microseconds.
    """

    # Anchors (set when first batch arrives)
    wall_start: int | None = None  # wall-clock start (perf_counter domain, µs)
    batch_start: int | None = None  # first event timestamp (µs)

    # HUD anchors (set once, never re-anchored)
    hud_wall_start: int | None = None  # wall start for HUD (µs)
    hud_rec_start: int | None = None  # recording start for HUD (µs)

    # Stats / runtime
    dropped_batches: int = 0
    emitted_batches: int = 0
    drops_since_last_emit: int = 0
    last_emit_event_time: int = 0  # last emitted event timestamp (µs)
    last_emit_wall_time: int = 0  # last emit wall time (µs)
    inst_drop_rate: float = 0.0  # drops / ms since last emit
    avg_drop_rate: float = 0.0  # running average drops / ms
    inst_speed: float = 0.0  # recent event-time / wall-time


class StreamingPacer:
    """Real-time pacer for an iterable of EventsBatch."""

    __slots__ = ("config", "state")

    def __init__(
        self,
        speed: float,
        force_speed: bool,
        drop_tolerance_ms: float = 0.0,
    ) -> None:
        drop_tolerance = int(drop_tolerance_ms * 1e3)
        self.config = PacerConfig(
            speed=speed,
            force_speed=force_speed,
            drop_tolerance=drop_tolerance,
        )
        self.state = PacerState()

    # --- HUD API (returns seconds) --------------------------------------

    def wall_elapsed(self) -> float:
        """Wall-clock time since first batch was seen (seconds)."""
        if self.state.hud_wall_start is None:
            msg = "HUD wall clock not initialized"
            raise ValueError(msg)
        now_us = int(time.perf_counter() * 1e6)
        return (now_us - self.state.hud_wall_start) / 1e6

    def recording_elapsed(self, end_time_us: int) -> float:
        """Recording time since first batch (seconds)."""
        if self.state.hud_rec_start is None:
            msg = "HUD recording clock not initialized"
            raise ValueError(msg)
        return (end_time_us - self.state.hud_rec_start) / 1e6

    def actual_playback_speed(self, end_time_us: int) -> float:
        """Long-term playback speed (event-time / wall-time)."""
        wall = self.wall_elapsed()
        rec = self.recording_elapsed(end_time_us)
        return rec / wall

    # --- basic timestamp helpers (event domain, µs) ----------------------

    @staticmethod
    def _batch_start(batch: EventsBatch) -> int:
        return int(batch.timestamps[0])

    @staticmethod
    def _batch_end(batch: EventsBatch) -> int:
        return int(batch.timestamps[-1])

    # --- internal helpers -----------------------------------------------

    def _initialize_pacer_state(self, first_batch: EventsBatch) -> None:
        """Initialize pacing and HUD anchors from the first batch."""
        if self.state.wall_start is not None:
            return

        now = int(time.perf_counter() * 1e6)
        recording_start_timestamp = self._batch_start(first_batch)

        # Pacing anchors
        self.state.wall_start = now
        self.state.batch_start = recording_start_timestamp

        # Last emit (event + wall) start from first batch
        self.state.last_emit_event_time = recording_start_timestamp
        self.state.last_emit_wall_time = now

        # HUD anchors set once
        if self.state.hud_wall_start is None:
            self.state.hud_wall_start = now
        if self.state.hud_rec_start is None:
            self.state.hud_rec_start = recording_start_timestamp

    def _reanchor(self, batch: EventsBatch) -> None:
        """Reset pacing anchors to this batch (used for resync)."""
        now = int(time.perf_counter() * 1e6)
        batch_start_timestamp = self._batch_start(batch)

        self.state.wall_start = now
        self.state.batch_start = batch_start_timestamp
        self.state.last_emit_event_time = batch_start_timestamp
        self.state.last_emit_wall_time = now

    def _target_wall_us(self, end_time: int) -> float:
        """Desired wall-clock elapsed time for this batch (µs)."""
        assert self.state.batch_start is not None
        event_elapsed = end_time - self.state.batch_start
        # Scale by playback speed; keep units in µs
        return event_elapsed / self.config.speed

    def _sleep_until(self, relative_targe_time: float) -> None:
        """Sleep until relative wall time reaches relative_target_time (µs)."""
        assert self.state.wall_start is not None
        while True:
            now = int(time.perf_counter() * 1e6) - self.state.wall_start
            time_diff = relative_targe_time - now
            if time_diff <= 0:
                break
            time.sleep(time_diff / 1e6)

    def _snapshot_and_yield(self, batch: EventsBatch) -> Iterator[EventsBatch]:
        """Emit batch and update instantaneous/average stats."""
        state = self.state
        end_time = self._batch_end(batch)

        state.emitted_batches += 1

        # Drop rate over event time since previous emit
        last_emit_event_time_diff = end_time - state.last_emit_event_time
        state.inst_drop_rate = (
            state.drops_since_last_emit * 1e3 / last_emit_event_time_diff
        )

        # Instantaneous playback speed (event-time / wall-time)
        now = int(time.perf_counter() * 1e6)
        last_emit_wall_time_diff = now - state.last_emit_wall_time

        state.inst_speed = last_emit_event_time_diff / last_emit_wall_time_diff

        # Running average drop rate
        state.avg_drop_rate += (
            state.inst_drop_rate - state.avg_drop_rate
        ) / state.emitted_batches

        # Reset per-window accumulator and update last timestamps
        state.drops_since_last_emit = 0
        state.last_emit_event_time = end_time
        state.last_emit_wall_time = now

        yield batch

    # --- main pacing API -------------------------------------------------

    def pace(self, batches: Iterable[EventsBatch]) -> Iterator[EventsBatch]:
        iterator = iter(batches)
        try:
            batch = next(iterator)
        except StopIteration:
            return

        self._initialize_pacer_state(batch)

        while True:
            end_time = self._batch_end(batch)
            target_time = self._target_wall_us(end_time)

            assert self.state.wall_start is not None
            now = int(time.perf_counter() * 1e6) - self.state.wall_start
            lag = now - target_time

            if self.config.force_speed and lag > 0:
                if self.state.emitted_batches == 0:
                    # One-time resync at startup
                    self._reanchor(batch)
                    target_time = 0.0
                    now = 0.0
                    lag = 0.0
                else:
                    max_drop_burst = 4
                    drops = 0
                    while (
                        lag > self.config.drop_tolerance
                        and drops < max_drop_burst
                    ):
                        self.state.dropped_batches += 1
                        self.state.drops_since_last_emit += 1
                        drops += 1

                        try:
                            batch = next(iterator)
                        except StopIteration:
                            # Still update stats for what we would emit
                            yield from self._snapshot_and_yield(batch)
                            return

                        end_time = self._batch_end(batch)
                        target_time = self._target_wall_us(end_time)
                        now = (
                            int(time.perf_counter() * 1e6)
                            - self.state.wall_start
                        )
                        lag = now - target_time

                    if (
                        lag > self.config.drop_tolerance
                        and drops >= max_drop_burst
                    ):
                        # Still behind after a drop burst: re-anchor
                        self._reanchor(batch)
                        target_time = 0.0
                        now = 0.0
                        lag = 0.0

            if target_time > now:
                self._sleep_until(target_time)

            yield from self._snapshot_and_yield(batch)

            try:
                batch = next(iterator)
            except StopIteration:
                return

    # --- stats API -------------------------------------------------------

    @property
    def dropped_batches(self) -> int:
        return self.state.dropped_batches

    @property
    def emitted_batches(self) -> int:
        return self.state.emitted_batches

    @property
    def instantaneous_drop_rate(self) -> float:
        return self.state.inst_drop_rate

    @property
    def instantaneous_speed(self) -> float:
        return self.state.inst_speed

    @property
    def average_drop_rate(self) -> float:
        return self.state.avg_drop_rate
