# evio/pacing/pacer.py
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import Optional

from evio.source.dat_file import BatchRange


class Pacer:
    """
    'display' -> speed-based pacing tied to event timestamps (1.0 = realtime).
                 Drops overdue batches; if early, blocks until target time.
    'asap'    -> no pacing, no dropping.
    """

    __slots__ = ("mode", "speed", "drop_tolerance_s", "_t0", "_e0", "_pump_cb")

    def __init__(
        self,
        mode: str = "display",
        speed: float = 1.0,
        drop_tolerance_s: float = 0.0,
        pump_cb: Optional[
            Callable[[], None]
        ] = None,  # e.g. lambda: cv2.waitKey(1)
    ) -> None:
        assert mode in ("display", "asap")
        self.mode = mode
        self.speed = max(float(speed), 1e-9)
        self.drop_tolerance_s = max(0.0, float(drop_tolerance_s))
        self._t0: float | None = None  # wall-clock anchor
        self._e0: int | None = None  # event-time anchor (µs)
        self._pump_cb = pump_cb

    def _target_wall_s(self, end_ts_us: int) -> float:
        # event elapsed (s) scaled by speed factor
        return max(0.0, (end_ts_us - (self._e0 or 0)) / 1e6) / self.speed

    def _sleep_until(self, target_rel_s: float) -> None:
        # Sleep in chunks so UI can be pumped and OS timers stay accurate
        while True:
            now_s = time.perf_counter() - (self._t0 or time.perf_counter())
            dt = target_rel_s - now_s
            if dt <= 0:
                break
            time.sleep(dt if dt < 0.05 else 0.05)
            if self._pump_cb is not None:
                self._pump_cb()

    def pace(self, batches: Iterable[BatchRange]) -> Iterator[BatchRange]:
        if self.mode == "asap":
            yield from batches
            return

        it = iter(batches)
        try:
            b = next(it)
        except StopIteration:
            return

        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._e0 = b.start_ts_us

        while True:
            # Missing timestamps? just yield (no pacing)
            if b.end_ts_us == 0 or self._e0 is None:
                yield b
            else:
                target_s = self._target_wall_s(b.end_ts_us)
                now_s = time.perf_counter() - self._t0
                lag_s = now_s - target_s
                if lag_s > self.drop_tolerance_s:
                    # Drop overdue batch and continue catching up
                    try:
                        b = next(it)
                        continue
                    except StopIteration:
                        yield b
                        return
                else:
                    # We're early: block until target_s (handles long windows)
                    if target_s > now_s:
                        self._sleep_until(target_s)
                    yield b

            try:
                b = next(it)
            except StopIteration:
                return
