"""Calibrated rotor-periodicity LLR arm for the fusion verdict gate.

Maps ``periodicity_score`` ∈ [0, 1) to a calibrated log-Bayes-factor via a
1-D logistic (same IRLS fit + prior-subtraction honesty as
:mod:`evio.detection.kinematic_llr`). ONE-SIDED by design: periodicity is
presence-only evidence (an edge-on/unlit rotor or night physics explains a
zero score without making the track clutter), so negative log-odds are
clamped to 0 — the arm can only add evidence, never subtract it. A missing
score is exactly neutral.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evio.detection.kinematic_llr import _fit_logistic_irls, _logit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

FEATURE_KEY = "periodicity_score"


@dataclass(slots=True)
class RotorLLR:
    """A fitted one-sided rotor log-Bayes-factor model."""

    coef: float
    intercept: float
    prior_logit: float
    weight: float = 1.0
    one_sided: bool = True

    def llr(self, features: Mapping[str, float] | None) -> float:
        """Additive rotor evidence; missing/non-finite score → exactly 0."""
        if features is None:
            return 0.0
        score = features.get(FEATURE_KEY)
        if score is None or not math.isfinite(float(score)):
            return 0.0
        raw = self.coef * float(score) + self.intercept - self.prior_logit
        if self.one_sided:
            raw = max(0.0, raw)
        return self.weight * raw

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "coef": self.coef,
            "intercept": self.intercept,
            "prior_logit": self.prior_logit,
            "weight": self.weight,
            "one_sided": self.one_sided,
        }, indent=2))

    @classmethod
    def load(cls, path: Path, *, weight: float | None = None) -> RotorLLR:
        d = json.loads(path.read_text())
        return cls(
            coef=float(d["coef"]),
            intercept=float(d["intercept"]),
            prior_logit=float(d["prior_logit"]),
            weight=float(d["weight"] if weight is None else weight),
            one_sided=bool(d.get("one_sided", True)),
        )


def fit_rotor_llr(
    scores: Sequence[float],
    labels: Sequence[float],
    *,
    l2: float = 1.0,
    weight: float = 1.0,
    max_iter: int = 50,
) -> RotorLLR:
    """Fit the 1-D logistic score→log-odds map (drone=1, clutter=0)."""
    x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        msg = "scores and labels must be non-empty and aligned"
        raise ValueError(msg)
    if y.min() == y.max():
        msg = (
            "labels contain a single class - a degenerate fit maps every "
            "score to a constant LLR (fusion_G3 lesson); supply both classes"
        )
        raise ValueError(msg)
    coef, intercept = _fit_logistic_irls(x, y, l2=l2, max_iter=max_iter)
    prior = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    return RotorLLR(
        coef=float(coef[0]), intercept=intercept,
        prior_logit=_logit(prior), weight=weight,
    )


__all__ = ["FEATURE_KEY", "RotorLLR", "fit_rotor_llr"]
