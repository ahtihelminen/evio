"""The transformer classifier batches ring evidence through the loaded model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fakes.dtc_transformer import FakeLoader, FakePredictor

from evio.detection import track_classifier
from evio.detection.track_classifier import LoadedTransformer
from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError, ResourceError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Compute, Cost, Stage
from evio.dtc.methods.transformer import (
    Transformer,
    TransformerConfig,
)
from evio.dtc.protocols import ClassifierCall, Context
from evio.dtc.registry import METHODS
from evio.dtc.types import Region, Track, Tracks
from tests.support import dtc

pytestmark = pytest.mark.t1

CONFIG = TransformerConfig("model.pt", "cpu")
LOADED = LoadedTransformer(object(), False, False, None)


def ring(value: int, length: int = 16) -> np.ndarray:
    return np.full((length, 32, 32, 3), value, dtype=np.uint8)


def call(stream: str, rings: dict[str, np.ndarray]) -> ClassifierCall:
    window = dtc.window(stream, dtc.empty_events(), 0, 10_000)
    tracks = Tracks(
        tuple(
            Track(stream, name, Region(point=(1.0, 1.0)), 1.0)
            for name in rings
        )
    )
    return ClassifierCall(
        dtc.step_of(0, 10_000, window), tracks, {"patch_ring": rings}
    )


@pytest.fixture
def predictor(monkeypatch: pytest.MonkeyPatch) -> FakePredictor:
    predict = FakePredictor()
    monkeypatch.setattr(track_classifier, "predict_drone_probs", predict)
    monkeypatch.setattr(
        track_classifier, "load_track_transformer", FakeLoader(LOADED)
    )
    return predict


def build() -> Transformer:
    context = Context("cam", ("cam",), {EXTENT: (64, 48)}, 10_000)
    return Transformer(CONFIG, context)


def test_registered_as_heavy_npu_classifier_without_dependencies() -> None:
    manifest = METHODS["transformer"].manifest
    assert manifest.stages == (Stage.CLASSIFY,)
    assert (manifest.compute, manifest.cost) == (Compute.NPU, Cost.HEAVY)
    assert manifest.evidence == ("patch_ring",)
    assert manifest.dependencies == ()
    assert manifest.cadence == 1


def test_loader_gets_the_configured_checkpoint_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = FakeLoader(LOADED)
    monkeypatch.setattr(track_classifier, "load_track_transformer", loader)
    build()
    assert [(str(path), device) for path, device in loader.calls] == [
        ("model.pt", "cpu")
    ]


def test_load_failure_raises_resource_error_with_the_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = LoadedTransformer(None, False, False, "no checkpoint")
    monkeypatch.setattr(
        track_classifier, "load_track_transformer", FakeLoader(failed)
    )
    with pytest.raises(ResourceError, match="no checkpoint"):
        build()


def test_event_token_checkpoint_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = LoadedTransformer(object(), True, True, None)
    monkeypatch.setattr(
        track_classifier, "load_track_transformer", FakeLoader(loaded)
    )
    with pytest.raises(ConfigError, match="v2"):
        build()


def test_scores_a_full_ring_and_skips_a_short_one(
    predictor: FakePredictor,
) -> None:
    one = call("cam", {"full": ring(255), "short": ring(255, 5)})
    verdicts = build().step(one.step, one.tracks, one.evidence)
    assert [
        (v.track_id, v.probability, v.age_steps) for v in verdicts.items
    ] == [("full", 1.0, 0)]


def test_step_many_makes_one_model_call_across_streams(
    predictor: FakePredictor,
) -> None:
    calls = [
        call("a", {"x": ring(0), "y": ring(255)}),
        call("b", {"x": ring(255)}),
    ]
    first, second = build().step_many(calls)
    assert len(predictor.batches) == 1
    assert predictor.batches[0].shape == (3, 16, 32, 32, 3)
    assert [(v.stream, v.track_id, v.probability) for v in first.items] == [
        ("a", "x", 0.0),
        ("a", "y", 1.0),
    ]
    assert [(v.stream, v.track_id) for v in second.items] == [("b", "x")]


def test_nothing_ready_makes_no_model_call(predictor: FakePredictor) -> None:
    (verdicts,) = build().step_many([call("a", {"x": ring(9, 3)})])
    assert verdicts.items == ()
    assert predictor.batches == []


def test_fake_predictor_keeps_batch_shape_and_row_order() -> None:
    patches = np.stack([ring(0), ring(255), ring(51)])
    probs = FakePredictor()(object(), patches)
    assert probs.shape == (3,)
    assert probs.tolist() == pytest.approx([0.0, 1.0, 0.2])


def test_fake_loader_returns_what_it_was_given() -> None:
    loader = FakeLoader(LOADED)
    assert loader(Path("m.pt")) is LOADED


def test_passes_the_conformance_kit(predictor: FakePredictor) -> None:
    failures = check_method(
        "transformer",
        facts={EXTENT: (128, 96)},
        config={"checkpoint": "model.pt"},
    )
    assert failures == []
