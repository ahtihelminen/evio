import time
from collections.abc import Iterable, Iterator

from evio.source.dat_file import BatchRange


class Pacer:
    __slots__ = (
        "_dropped_batches",
        "_e_start",
        "_emitted_batches",
        "_t_start",
        "drop_tolerance_s",
        "speed",
    )

    def __init__(
        self,
        speed: float = 1.0,
        drop_tolerance_s: float = 0.0,
    ) -> None:
        self.speed = max(float(speed), 1e-9)
        self.drop_tolerance_s = max(0.0, float(drop_tolerance_s))
        self._t_start: float | None = None  # wall-clock anchor
        self._e_start: int | None = None  # event-time anchor (µs)

        # --- new counters ---
        self._dropped_batches: int = 0
        self._emitted_batches: int = 0

    # --- new helper API (optional but handy) ---
    @property
    def dropped_batches(self) -> int:
        """Number of batches skipped because they were overdue."""
        return self._dropped_batches

    @property
    def emitted_batches(self) -> int:
        """Number of batches actually yielded downstream."""
        return self._emitted_batches

    def reset_stats(self) -> None:
        """Zero the counters (useful when starting a new clip/loop)."""
        self._dropped_batches = 0
        self._emitted_batches = 0

    # --- existing internals ---
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

    def pace(self, batches: Iterable[BatchRange]) -> Iterator[BatchRange]:
        iterator = iter(batches)
        try:
            batch_range = next(iterator)
        except StopIteration:
            return

        if self._t_start is None:
            self._t_start = time.perf_counter()
            self._e_start = batch_range.start_ts_us

        while True:
            # Missing timestamps? just yield (no pacing)
            if batch_range.end_ts_us == 0 or self._e_start is None:
                self._emitted_batches += 1
                yield batch_range
            else:
                target_s = self._target_wall_s(batch_range.end_ts_us)
                now_s = time.perf_counter() - self._t_start
                lag_s = now_s - target_s
                if lag_s > self.drop_tolerance_s:
                    # Drop overdue batch and continue catching up
                    self._dropped_batches += 1  # <-- increment here
                    try:
                        batch_range = next(iterator)
                        continue
                    except StopIteration:
                        # nothing more to drop; yield the last one we had
                        self._emitted_batches += 1
                        yield batch_range
                        return
                else:
                    # We're early: block until target_s (handles long windows)
                    if target_s > now_s:
                        self._sleep_until(target_s)
                    self._emitted_batches += 1
                    yield batch_range

            try:
                batch_range = next(iterator)
            except StopIteration:
                return
