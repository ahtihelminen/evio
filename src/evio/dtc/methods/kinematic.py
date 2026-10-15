"""Kinematic drone / no-drone heuristic: a logistic of motion features."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import Evidence, Step, Tracks, Verdict, Verdicts

UNKNOWN = 0.5


@dataclass(frozen=True)
class KinematicConfig:
    """Weights of the linear score; speed is in units of `speed_scale_px_s`
    and age saturates at `age_cap_frames`."""

    bias: float = -2.0
    speed_weight: float = 1.0
    straightness_weight: float = 2.0
    speed_cv_weight: float = -1.0
    age_weight: float = 1.5
    speed_scale_px_s: float = 100.0
    age_cap_frames: float = 30.0


@method(
    "kinematic",
    stages=(Stage.CLASSIFY,),
    config=KinematicConfig,
    evidence=("kinematics",),
)
class Kinematic:
    """Drone / no-drone from steady, straight, sustained motion."""

    def __init__(self, config: KinematicConfig, context: Context) -> None:
        self.config = config

    def reset(self) -> None:
        """Stateless: every step is computed from its own evidence."""

    def step(self, step: Step, tracks: Tracks, evidence: Evidence) -> Verdicts:
        features = evidence.get("kinematics", {})
        return Verdicts(
            tuple(
                Verdict(
                    track.stream,
                    track.id,
                    self._probability(features.get(track.id, {})),
                )
                for track in tracks.items
            )
        )

    def _probability(self, features: Mapping[str, float]) -> float:
        """The logistic of the weighted features (tanh form, so it saturates
        instead of overflowing); 0.5 if any is missing."""
        config = self.config
        try:
            score = (
                config.bias
                + config.speed_weight
                * features["speed_mean"]
                / config.speed_scale_px_s
                + config.straightness_weight * features["straightness"]
                + config.speed_cv_weight * features["speed_cv"]
                + config.age_weight
                * min(features["age_frames"], config.age_cap_frames)
                / config.age_cap_frames
            )
        except KeyError:
            return UNKNOWN
        return (1 + math.tanh(score / 2)) / 2
