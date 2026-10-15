"""Temporal transformer drone classifier over the runtime's patch ring.

The ring and the cadence belong to the runtime, so this wraps only the model:
`load_track_transformer` and `predict_drone_probs` of the existing
classifier. The manifest cadence stays 1 and the method has no cadence of its
own, so a pipeline spec cannot change it. Only the v2 patch-only checkpoint is
supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evio.dtc.errors import ConfigError, ResourceError
from evio.dtc.manifest import Compute, Cost, Stage
from evio.dtc.protocols import ClassifierCall, Context
from evio.dtc.registry import method
from evio.dtc.types import Evidence, Step, Tracks, Verdict, Verdicts

if TYPE_CHECKING:
    from collections.abc import Sequence

RING_LEN = 16


@dataclass(frozen=True)
class TransformerConfig:
    checkpoint: str
    device: str = "auto"


@method(
    "transformer",
    stages=(Stage.CLASSIFY,),
    config=TransformerConfig,
    compute=Compute.NPU,
    cost=Cost.HEAVY,
    evidence=("patch_ring",),
)
class Transformer:
    """Temporal transformer drone classifier, batched across streams."""

    def __init__(self, config: TransformerConfig, context: Context) -> None:
        from evio.detection import track_classifier

        loaded = track_classifier.load_track_transformer(
            Path(config.checkpoint), config.device
        )
        if loaded.bundle is None:
            raise ResourceError(loaded.load_warning)
        if loaded.is_v3 or getattr(loaded.bundle, "stateful", False):
            msg = "only v2 patch-only transformer checkpoints are supported"
            raise ConfigError(msg)
        self._bundle = loaded.bundle
        self._predict = track_classifier.predict_drone_probs

    def reset(self) -> None:
        """Stateless: the runtime owns the ring."""

    def step(self, step: Step, tracks: Tracks, evidence: Evidence) -> Verdicts:
        return self.step_many([ClassifierCall(step, tracks, evidence)])[0]

    def step_many(self, calls: Sequence[ClassifierCall]) -> list[Verdicts]:
        """Score every track with a full ring, in one model call."""
        ready = [
            (index, track, ring)
            for index, call in enumerate(calls)
            for track in call.tracks.items
            for ring in [call.evidence.get("patch_ring", {}).get(track.id, ())]
            if len(ring) == RING_LEN
        ]
        found: list[list[Verdict]] = [[] for _ in calls]
        if ready:
            patches = np.stack([np.asarray(ring) for *_, ring in ready])
            probs = self._predict(self._bundle, patches.astype(np.uint8))
            for (index, track, _), prob in zip(ready, probs, strict=True):
                verdict = Verdict(track.stream, track.id, float(prob))
                found[index].append(verdict)
        return [Verdicts(tuple(items)) for items in found]
