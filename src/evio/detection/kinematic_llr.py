"""Kinematic log-likelihood-ratio fusion for the SPRT verdict gate.

The transformer scores *appearance*; the tracker scores *motion*. This module
learns ``P(drone | kinematics)`` and contributes its log-Bayes-factor

    llr = logit(P(drone | kinematics)) - logit(prior)

as ADDITIVE independent evidence to :class:`SequentialGate`'s log-odds
increment (naive-Bayes fusion — it assumes appearance and kinematics are
conditionally independent given the class, the standard approximation). The
prior is subtracted so a kinematically-average track contributes 0 and the base
rate is not double-counted (the transformer's log-odds already carries it).

The model is a calibrated logistic (linear-in-standardised-features) — a proper
log-odds model with ZERO extra runtime dependency, which matters on the lean
NUC deploy env (no scikit-learn there). A gradient-boosted variant with the same
``.llr()`` interface is a future drop-in for nonlinear feature interactions; it
is deliberately deferred to avoid a heavy dependency and a numpy tree-eval
reimplementation.

Like SequentialGate, this is UNTRUSTED as a primary kill/promote source until
its realised FP/hr is validated on a real empty-sky corpus (EmptySky).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from evio.detection.types import Detection

# Ordered feature contract. Callers assemble a dict with these keys from the
# tracker's per-track kinematics and the detection's quality scores via
# :func:`assemble_features`; a missing key falls back to the training mean
# (neutral evidence), so the fusion degrades gracefully.
KINEMATIC_FEATURES: tuple[str, ...] = (
    "speed_mean",       # px/s, mean over the track's speed ring
    "speed_cv",         # coefficient of variation of speed (steadiness)
    "straightness",     # endpoint distance / path length, in [0, 1]
    "vel_mag",          # px/s, current Kalman velocity magnitude
    "age_frames",       # windows observed
    "miss_count",       # consecutive missed windows
    "bbox_diag",        # px, bounding-box diagonal (apparent size)
    "pos_cov_trace",    # Kalman position-covariance trace (localisation unc.)
    "m_coh",            # detection motion-coherence
    "best_psr",         # detection peak-to-sidelobe ratio
    "log_event_count",  # log1p(event_count)
    "z_score",          # detection anomaly z-score
)


def _logit(p: float, *, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def assemble_features(
    det: Detection, track_feats: Mapping[str, float] | None,
) -> dict[str, float]:
    """Merge detection quality scores with tracker kinematics into one dict.

    ``track_feats`` is :meth:`TrackManager.track_features` output (may be None
    for a track the caller couldn't resolve). Used identically at train time
    and live so the feature vectors match.
    """
    feats: dict[str, float] = dict(track_feats or {})
    feats["m_coh"] = float(det.m_coh)
    feats["best_psr"] = float(det.best_psr)
    feats["log_event_count"] = math.log1p(max(0, int(det.event_count)))
    feats["z_score"] = float(det.z_score)
    # Rotor evidence rides the same bus (read by the RotorLLR arm; absent =
    # neutral). NOT in KINEMATIC_FEATURES — the kinematic arm ignores it.
    if det.periodicity_score is not None:
        feats["periodicity_score"] = float(det.periodicity_score)
    return feats


@dataclass(slots=True)
class KinematicLLR:
    """A fitted kinematic log-Bayes-factor model (logistic in std features)."""

    feature_names: tuple[str, ...]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    coef: NDArray[np.float64]
    intercept: float
    prior_logit: float
    weight: float = 1.0
    model_kind: str = "logistic"

    def _standardise(
        self, features: Mapping[str, float],
    ) -> NDArray[np.float64]:
        x = np.array(
            [float(features.get(n, m)) for n, m in zip(
                self.feature_names, self.mean, strict=True)],
            dtype=np.float64,
        )
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = self.mean[bad]  # non-finite -> neutral
        return (x - self.mean) / self.std

    def posterior_logit(self, features: Mapping[str, float]) -> float:
        """``logit(P(drone | kinematics))`` under the fitted model."""
        z = self._standardise(features)
        return float(z @ self.coef + self.intercept)

    def llr(self, features: Mapping[str, float] | None) -> float:
        """Additive log-Bayes-factor for the SPRT increment.

        0 = an average sample (no evidence either way); positive = more
        drone-like than the base rate; negative = less. When NONE of this
        arm's features are present the arm is exactly neutral — imputing
        everything to the training mean would otherwise emit a constant
        ``intercept - prior_logit`` bias on every window (e.g. the IRSTD arm
        with no patch attached), which against |A| ~ 2.9 dominates the SPRT
        within a few windows.
        """
        if features is None or not any(
            name in features for name in self.feature_names
        ):
            return 0.0
        return self.weight * (self.posterior_logit(features) - self.prior_logit)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "prior_logit": self.prior_logit,
            "weight": self.weight,
            "model_kind": self.model_kind,
        }, indent=2))

    @classmethod
    def load(cls, path: Path, *, weight: float | None = None) -> KinematicLLR:
        d = json.loads(path.read_text())
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
            prior_logit=float(d["prior_logit"]),
            weight=float(d["weight"] if weight is None else weight),
            model_kind=str(d.get("model_kind", "logistic")),
        )


def _fit_logistic_irls(
    z: NDArray[np.float64], y: NDArray[np.float64], *,
    l2: float, max_iter: int,
) -> tuple[NDArray[np.float64], float]:
    """Ridge-penalised logistic regression via IRLS (Newton) on std features.

    Returns ``(coef, intercept)``. The bias column is not penalised. A small
    weight floor keeps the Hessian non-singular on near-separable data.
    """
    n, d = z.shape
    design = np.hstack([z, np.ones((n, 1))])  # last column = bias
    penalty = np.eye(d + 1) * l2
    penalty[d, d] = 0.0  # do not shrink the intercept
    w = np.zeros(d + 1)
    for _ in range(max_iter):
        eta = design @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        s = np.clip(p * (1.0 - p), 1e-6, None)
        hessian = design.T @ (design * s[:, None]) + penalty
        grad = design.T @ (y - p) - penalty @ w
        step = np.linalg.solve(hessian, grad)
        w = w + step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return w[:d], float(w[d])


def fit_kinematic_llr(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[float],
    *,
    feature_names: tuple[str, ...] = KINEMATIC_FEATURES,
    l2: float = 1.0,
    weight: float = 1.0,
    max_iter: int = 50,
) -> KinematicLLR:
    """Fit a :class:`KinematicLLR` from per-(window, track) feature dicts.

    ``rows`` are feature dicts (see :func:`assemble_features`); ``labels`` are
    the matching is-drone flags (0/1). Standardisation stats are learned here so
    train/serve vectors align.
    """
    x = np.array(
        [[float(r.get(n, np.nan)) for n in feature_names] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        msg = "rows and labels must be non-empty and aligned"
        raise ValueError(msg)
    mean = np.nanmean(x, axis=0)
    fill = np.where(~np.isfinite(x))
    x[fill] = np.take(mean, fill[1])  # impute non-finite with column mean
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    coef, intercept = _fit_logistic_irls(z, y, l2=l2, max_iter=max_iter)
    prior = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    return KinematicLLR(
        feature_names=tuple(feature_names),
        mean=mean, std=std, coef=coef, intercept=intercept,
        prior_logit=_logit(prior), weight=weight,
    )


__all__ = [
    "KINEMATIC_FEATURES",
    "KinematicLLR",
    "assemble_features",
    "fit_kinematic_llr",
]
