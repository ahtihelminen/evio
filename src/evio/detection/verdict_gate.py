"""K-of-M persistence gate for committing per-track drone verdicts.

A single window's transformer probability is noisy, so the gate only
declares a track a DRONE when at least ``m`` of its last ``k`` per-window
probabilities cross the live threshold. ``k == m == 1`` reduces to plain
single-window thresholding. Extracted from
:class:`evio.streaming.live_detection.LiveDetectionEngine` (``_gate_fires`` /
``verdict_history``) so the SAPIENT emit path can share the exact operating
point recommended by ``scripts/sweep_v4_operating_point.py``.
"""

from __future__ import annotations

from collections import deque


class PersistenceGate:
    """Per-(camera, track) ring of recent probabilities with an M-of-K test."""

    def __init__(self, *, k: int = 4, m: int = 2) -> None:
        self._k = max(1, int(k))
        self._m = max(1, min(int(m), self._k))
        self._rings: dict[tuple[str, str], deque[float]] = {}

    @property
    def k(self) -> int:
        return self._k

    @property
    def m(self) -> int:
        return self._m

    def update(
        self, serial: str, track_id: str, prob: float, threshold: float,
    ) -> bool:
        """Record ``prob`` for a track and return whether the gate fires now."""
        key = (serial, track_id)
        ring = self._rings.get(key)
        if ring is None:
            ring = deque(maxlen=self._k)
            self._rings[key] = ring
        ring.append(float(prob))
        above = sum(1 for p in ring if p >= threshold)
        return above >= self._m

    def prune(self, serial: str, live_ids: set[str]) -> None:
        """Drop rings for tracks of ``serial`` that are no longer alive."""
        stale = [
            key
            for key in self._rings
            if key[0] == serial and key[1] not in live_ids
        ]
        for key in stale:
            del self._rings[key]


__all__ = ["PersistenceGate"]
