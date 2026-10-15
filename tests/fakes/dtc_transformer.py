"""Stand-ins for the transformer loader and predictor of `track_classifier`.

`FakeLoader` replaces `load_track_transformer`; `FakePredictor` replaces
`predict_drone_probs`, scoring each track by its mean patch brightness and
recording the batches it was given. Both keep the real signatures' shapes:
patches `(B, T, H, W, 3)` in, `(B,)` probabilities out, row order kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evio.detection.track_classifier import LoadedTransformer


class FakeLoader:
    """Returns a fixed `LoadedTransformer` and records the call."""

    def __init__(self, loaded: LoadedTransformer) -> None:
        self.loaded = loaded
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, ckpt: Path, device: str = "auto") -> LoadedTransformer:
        self.calls.append((ckpt, device))
        return self.loaded


class FakePredictor:
    """Probability = mean patch value / 255, one per batch row."""

    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def __call__(
        self, bundle: object, patches: np.ndarray, *args: object, **kw: object
    ) -> np.ndarray:
        self.batches.append(patches)
        return patches.mean(axis=(1, 2, 3, 4)) / 255.0
