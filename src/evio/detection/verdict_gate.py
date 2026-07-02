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
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class LLRArm(Protocol):
    """An additive evidence source for :class:`FusionGate`.

    Reads what it knows from the shared per-track features dict and returns a
    calibrated log-Bayes-factor (0 = neutral / nothing to say).
    """

    def llr(self, features: Mapping[str, float] | None) -> float: ...


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
        *, features: Mapping[str, float] | None = None,
    ) -> bool:
        """Record ``prob`` for a track and return whether the gate fires now.

        ``features`` is accepted for verdict-gate signature parity and ignored
        (M-of-K uses appearance probability only).
        """
        del features
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
        *, features: Mapping[str, float] | None = None, extra_llr: float = 0.0,
    ) -> bool:
        """Accumulate evidence; return whether the track fires DRONE now.

        ``extra_llr`` is an additive independent-evidence log-Bayes-factor
        (e.g. the kinematic LLR from :class:`FusionGate`) folded into the
        log-odds increment. ``features`` is ignored here (SPRT scores the
        appearance probability); the fusion gate consumes it upstream.
        """
        del features
        key = (serial, track_id)
        increment = (
            prob_to_log_odds(prob, temperature=self._temperature)
            - _logit(threshold)
            + float(extra_llr)
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


class FusionGate:
    """SPRT gate fed appearance + independent evidence arms additively.

    Wraps a :class:`SequentialGate` and one or more fitted LLR arms (kinematic
    :class:`~evio.detection.kinematic_llr.KinematicLLR`, rotor-periodicity,
    IRSTD, ...). Each window it sums the arms' log-Bayes-factors into the
    transformer's log-odds increment (naive Bayes: arms assumed conditionally
    independent given class), so a track with drone-like motion / a rotor comb
    / small-target contrast accrues evidence faster — without a second hard
    gate. Every arm exposes ``llr(features) -> float`` over the shared
    per-track features dict and returns 0 for anything it can't read, so the
    gate degrades to pure SPRT when evidence is missing.

    Same FP/hr caveat as :class:`SequentialGate`: advisory until the realised
    rate is validated on a real empty-sky corpus (EmptySky).
    """

    def __init__(
        self, models: Sequence[LLRArm], gate: SequentialGate,
    ) -> None:
        self._models = tuple(models)
        self._gate = gate

    @property
    def upper(self) -> float:
        return self._gate.upper

    @property
    def lower(self) -> float:
        return self._gate.lower

    def update(
        self, serial: str, track_id: str, prob: float, threshold: float,
        *, features: Mapping[str, float] | None = None,
    ) -> bool:
        """Fuse the evidence arms into the SPRT increment; return fires-now."""
        extra_llr = (
            sum(m.llr(features) for m in self._models) if features else 0.0
        )
        return self._gate.update(
            serial, track_id, prob, threshold, extra_llr=extra_llr,
        )

    def prune(self, serial: str, live_ids: set[str]) -> None:
        self._gate.prune(serial, live_ids)


VerdictGate = PersistenceGate | SequentialGate | FusionGate


def build_verdict_gate(
    kind: str = "mofk",
    *,
    k: int = 4,
    m: int = 2,
    alpha: float = 0.05,
    beta: float = 0.1,
    temperature: float = 1.0,
    leak: float = 0.0,
    kinematic_llr_path: Path | None = None,
    fusion_weight: float = 1.0,
    rotor_llr_path: Path | None = None,
    rotor_weight: float = 1.0,
    irstd_llr_path: Path | None = None,
    irstd_weight: float = 1.0,
) -> VerdictGate:
    """Construct the configured verdict gate (mofk default, sprt/fusion opt-in).

    ``fusion`` requires ``kinematic_llr_path`` (fit by
    ``scripts/train_kinematic_llr.py``); the rotor and IRSTD arms are appended
    when their model paths are given.
    """
    if kind == "mofk":
        return PersistenceGate(k=k, m=m)
    if kind == "sprt":
        return SequentialGate(
            alpha=alpha, beta=beta, temperature=temperature, leak=leak,
        )
    if kind == "fusion":
        if kinematic_llr_path is None:
            msg = "verdict gate 'fusion' requires kinematic_llr_path"
            raise ValueError(msg)
        from evio.detection.kinematic_llr import KinematicLLR
        models: list[LLRArm] = [
            KinematicLLR.load(kinematic_llr_path, weight=fusion_weight),
        ]
        if rotor_llr_path is not None:
            from evio.detection.rotor_llr import RotorLLR
            models.append(RotorLLR.load(rotor_llr_path, weight=rotor_weight))
        if irstd_llr_path is not None:
            models.append(
                KinematicLLR.load(irstd_llr_path, weight=irstd_weight),
            )
        gate = SequentialGate(
            alpha=alpha, beta=beta, temperature=temperature, leak=leak,
        )
        return FusionGate(models, gate)
    msg = f"unknown verdict gate {kind!r} (use 'mofk', 'sprt', or 'fusion')"
    raise ValueError(msg)


__all__ = [
    "FusionGate",
    "LLRArm",
    "PersistenceGate",
    "SequentialGate",
    "VerdictGate",
    "build_verdict_gate",
    "prob_to_log_odds",
]
