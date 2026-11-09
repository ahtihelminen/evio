"""Fidelity for the transformer doubles in `tests/fakes/dtc_transformer.py`.

`FakeLoader` and `FakePredictor` stand in for `load_track_transformer` and
`predict_drone_probs` of `evio.detection.track_classifier` in the
transformer method tests. These tests run each beside the real function on
the same inputs and fail when the call shape, return type or output shape
drifts.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from fakes.dtc_transformer import FakeLoader, FakePredictor

from evio.detection.track_classifier import (
    LoadedTransformer,
    load_track_transformer,
    predict_drone_probs,
)

pytestmark = pytest.mark.t1

LOADED = LoadedTransformer(object(), False, False, None)


class _Bundle:
    """The one method the real predictor calls: logits follow brightness."""

    def predict(
        self, patches: np.ndarray, *rest: object
    ) -> dict[str, np.ndarray]:
        return {"drone_logits": patches.mean(axis=(2, 3, 4)) / 255.0 - 0.5}


def _patches(rows: int = 3) -> np.ndarray:
    """`(B, T, H, W, 3)` with row `i` brighter than row `i - 1`."""
    shape = (rows, 4, 5, 6, 3)
    return np.stack(
        [np.full(shape[1:], 60 + 50 * i, np.uint8) for i in range(rows)]
    )


def test_the_fake_loader_is_called_like_the_real_loader(tmp_path: Path) -> None:
    fake = FakeLoader(LOADED)
    ckpt = tmp_path / "missing.pt"
    real_args = inspect.signature(load_track_transformer).bind(ckpt, "cpu")
    fake_args = inspect.signature(fake).bind(ckpt, "cpu")
    assert list(real_args.arguments) == list(fake_args.arguments)
    assert inspect.signature(fake).bind(ckpt).arguments == {"ckpt": ckpt}
    assert inspect.signature(load_track_transformer).bind(ckpt).arguments == {
        "ckpt": ckpt
    }


def test_both_loaders_return_a_loaded_transformer_for_the_same_call(
    tmp_path: Path,
) -> None:
    """Deliberate difference: the real loader reads the checkpoint and, for a
    missing one, returns a warning with no bundle; the fake returns what it
    was given, so only the type and fields are compared."""
    ckpt = tmp_path / "missing.pt"
    real = load_track_transformer(ckpt, "cpu")
    fake = FakeLoader(LOADED)(ckpt, "cpu")
    assert type(real) is type(fake) is LoadedTransformer
    assert real.bundle is None
    assert real.load_warning
    assert fake.load_warning is None


def test_the_fake_loader_records_the_arguments_the_real_one_takes(
    tmp_path: Path,
) -> None:
    fake = FakeLoader(LOADED)
    fake(tmp_path / "a.pt")
    fake(tmp_path / "b.pt", "cpu")
    assert fake.calls == [
        (tmp_path / "a.pt", "auto"),
        (tmp_path / "b.pt", "cpu"),
    ]
    real_device = inspect.signature(load_track_transformer).parameters["device"]
    assert real_device.default == "auto"


@pytest.mark.parametrize("is_v4", [False, True])
def test_both_predictors_map_a_batch_to_one_probability_per_row(
    is_v4: bool,
) -> None:
    patches = _patches()
    real = predict_drone_probs(_Bundle(), patches, is_v4=is_v4)
    fake = FakePredictor()(_Bundle(), patches, is_v4=is_v4)
    assert real.shape == fake.shape == (len(patches),)
    assert real.dtype.kind == fake.dtype.kind == "f"
    for probs in (real, fake):
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert np.argsort(real).tolist() == np.argsort(fake).tolist()


def test_both_predictors_accept_the_v3_event_branch_arguments() -> None:
    patches = _patches(2)
    events, mask = np.zeros((2, 8, 4), np.float32), np.ones((2, 8), bool)
    real = predict_drone_probs(_Bundle(), patches, events, mask, is_v4=True)
    fake = FakePredictor()(_Bundle(), patches, events, mask, is_v4=True)
    assert real.shape == fake.shape == (2,)


def test_the_fake_predictor_keeps_the_row_order_the_real_one_keeps() -> None:
    patches = _patches()
    for probs in (
        predict_drone_probs(_Bundle(), patches[::-1]),
        FakePredictor()(_Bundle(), patches[::-1]),
    ):
        assert probs.tolist() == sorted(probs.tolist(), reverse=True)
