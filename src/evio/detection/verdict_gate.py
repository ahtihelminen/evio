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

import math
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


def _logit(p: float, *, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def prob_to_log_odds(prob: float, *, temperature: float = 1.0) -> float:
    """Convert a calibrated drone probability to a log-odds score.

    The v4 runtime emits a sigmoid probability, not a logit; this inverts it
    (with a ``temperature`` fit on val — see ``evaluation.stats``) into the
    additive evidence :class:`SequentialGate` accumulates.
    """
    return _logit(prob) / max(temperature, 1e-6)


class SequentialGate:
    """Wald-SPRT log-odds accumulator — a drop-in for :class:`PersistenceGate`.

    Per (camera, track) it keeps a running log-odds ``S`` and adds, each
    observed window, ``logit(prob)/T - logit(threshold)`` (evidence vs the
    operating point). It declares DRONE at ``S >= A = log((1-beta)/alpha)``,
    latches there, and resets (KILL) at ``S <= B = log(beta/(1-alpha))``. A
    dropped window is simply no ``update`` call, so ``S`` is unchanged and a
    latched DRONE survives it: one R1 miss is non-fatal, which M-of-K is not.

    ``update`` returns the bool "fires now" for call-site parity. ``alpha``
    maps to the FP/hr budget, but because consecutive windows are NOT
    independent (overlapping 16-frame windows share events) the realised FP/hr
    MUST be validated on a real empty-sky corpus before this is trusted as the
    primary gate — until then it is advisory (execution plan, risk 9.4).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        beta: float = 0.1,
        temperature: float = 1.0,
        leak: float = 0.0,
    ) -> None:
        self._a = math.log((1.0 - beta) / alpha)
        self._b = math.log(beta / (1.0 - alpha))
        self._temperature = max(float(temperature), 1e-6)
        self._leak = min(max(float(leak), 0.0), 1.0)
        self._scores: dict[tuple[str, str], float] = {}
        self._fired: set[tuple[str, str]] = set()

    @property
    def upper(self) -> float:
        """The DRONE log-odds threshold A = log((1-beta)/alpha)."""
        return self._a

    @property
    def lower(self) -> float:
        """The KILL log-odds threshold B = log(beta/(1-alpha))."""
        return self._b

    def update(
        self, serial: str, track_id: str, prob: float, threshold: float,
    ) -> bool:
        """Accumulate evidence; return whether the track fires DRONE now."""
        key = (serial, track_id)
        increment = (
            prob_to_log_odds(prob, temperature=self._temperature)
            - _logit(threshold)
        )
        score = self._scores.get(key, 0.0) * (1.0 - self._leak) + increment
        score = min(self._a, max(self._b, score))
        self._scores[key] = score
        if score <= self._b:
            self._scores[key] = 0.0  # KILL — purge accumulated evidence
            self._fired.discard(key)
            return False
        if score >= self._a:
            self._fired.add(key)
            return True
        return key in self._fired  # PENDING — latched DRONE rides it out

    def prune(self, serial: str, live_ids: set[str]) -> None:
        """Drop state for tracks of ``serial`` that are no longer alive."""
        stale = [
            key
            for key in self._scores
            if key[0] == serial and key[1] not in live_ids
        ]
        for key in stale:
            del self._scores[key]
            self._fired.discard(key)


VerdictGate = PersistenceGate | SequentialGate


def build_verdict_gate(
    kind: str = "mofk",
    *,
    k: int = 4,
    m: int = 2,
    alpha: float = 0.05,
    beta: float = 0.1,
    temperature: float = 1.0,
    leak: float = 0.0,
) -> VerdictGate:
    """Construct the configured verdict gate (mofk default, sprt opt-in)."""
    if kind == "mofk":
        return PersistenceGate(k=k, m=m)
    if kind == "sprt":
        return SequentialGate(
            alpha=alpha, beta=beta, temperature=temperature, leak=leak,
        )
    msg = f"unknown verdict gate {kind!r} (use 'mofk' or 'sprt')"
    raise ValueError(msg)


__all__ = [
    "PersistenceGate",
    "SequentialGate",
    "VerdictGate",
    "build_verdict_gate",
    "prob_to_log_odds",
]
