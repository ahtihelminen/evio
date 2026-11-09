"""Fidelity for the chain doubles: `FakeTimer`, `CountingPipeline`, the toys.

`FakeTimer` stands in for `SystemTimer`, `CountingPipeline` for the `Chain`
that `build` returns, and the toys in `tests/fakes/dtc_toys.py` for real
methods. Each runs beside the real class or through the real consumer's
checks, and fails when it drifts from it.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Iterator

import pytest
from fakes.dtc_counting_pipeline import CountingPipeline
from fakes.dtc_timer import FakeTimer
from fakes.dtc_toys import TOYS, registered

from evio.dtc import protocols
from evio.dtc.chain import build
from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.registry import METHODS
from evio.dtc.result import Pipeline, StepResult
from evio.dtc.timing import StageTiming, SystemTimer
from tests.support.dtc_chain import STEP_US, STREAMS, spec_of, step_at

pytestmark = pytest.mark.t1

importlib.import_module("evio.dtc.methods")
PROTOCOLS = {
    (Stage.DETECT,): protocols.Detector,
    (Stage.TRACK,): protocols.Tracker,
    (Stage.CLASSIFY,): protocols.Classifier,
    (Stage.DECIDE,): protocols.Decision,
    (Stage.DETECT, Stage.TRACK): protocols.FusedMethod,
}
SOUND_TOYS = (
    "toy-detector",
    "toy-tracker",
    "toy-joint-detector",
    "toy-detect-track",
    "toy-classifier",
    "toy-batch-classifier",
    "toy-decision",
)


@pytest.fixture
def toys() -> Iterator[None]:
    with registered():
        yield


def real_chain(timer: object) -> Pipeline:
    return build(
        spec_of("toy-detector", "toy-tracker"),
        {EXTENT: (1280, 720)},
        streams=STREAMS,
        step_us=STEP_US,
        timer=timer,
    )


def parameters(function: object) -> list[str]:
    return list(inspect.signature(function).parameters)


def test_the_fake_timer_reads_like_the_system_timer() -> None:
    for name in ("wall_ns", "cpu_ns"):
        assert parameters(getattr(FakeTimer, name)) == parameters(
            getattr(SystemTimer, name)
        ), name
        for timer in (SystemTimer(), FakeTimer()):
            first, second = getattr(timer, name)(), getattr(timer, name)()
            assert type(first) is type(second) is int, (name, timer)
            assert second >= first, (name, timer)


@pytest.mark.usefixtures("toys")
def test_a_chain_times_its_stages_the_same_way_with_either_timer() -> None:
    results = [
        real_chain(timer).step(step_at(0, *STREAMS))
        for timer in (SystemTimer(), FakeTimer())
    ]
    real, fake = (r.timings for r in results)
    assert set(real) == set(fake) == {"toy-detector", "toy-tracker"}
    for timings in (real, fake):
        for timing in timings.values():
            assert isinstance(timing, StageTiming)
            assert type(timing.wall_ns) is type(timing.cpu_ns) is int
            assert timing.wall_ns >= 0
            assert timing.cpu_ns >= 0


@pytest.mark.usefixtures("toys")
def test_the_counting_pipeline_offers_what_the_real_pipeline_offers() -> None:
    real, fake = real_chain(FakeTimer()), CountingPipeline()
    for name in ("provenance", "bus", "stats"):
        assert hasattr(fake, name), name
        assert type(getattr(fake, name)) is type(getattr(real, name)), name
    for name in ("step", "reset"):
        assert parameters(getattr(CountingPipeline, name)) == parameters(
            getattr(type(real), name)
        ), name


@pytest.mark.usefixtures("toys")
def test_a_counting_pipeline_step_has_the_shape_of_a_real_step_result() -> None:
    step = step_at(0, *STREAMS)
    real = real_chain(FakeTimer()).step(step)
    fake = CountingPipeline().step(step)
    assert type(fake) is type(real) is StepResult
    for field in dataclasses.fields(StepResult):
        assert type(getattr(fake, field.name)) is type(
            getattr(real, field.name)
        ), field.name
    assert (fake.index, fake.t_start_us, fake.t_end_us) == (
        real.index,
        real.t_start_us,
        real.t_end_us,
    )


@pytest.mark.usefixtures("toys")
def test_reset_returns_a_counting_and_a_real_pipeline_to_the_start() -> None:
    for pipeline in (real_chain(FakeTimer()), CountingPipeline()):
        first = pipeline.step(step_at(0, *STREAMS))
        pipeline.step(step_at(1, *STREAMS))
        pipeline.reset()
        assert pipeline.step(step_at(0, *STREAMS)) == first, pipeline


@pytest.mark.usefixtures("toys")
def test_the_real_pipeline_refuses_a_replayed_step_without_a_reset() -> None:
    """Deliberate difference: the real chain raises on an out-of-order step,
    where the counting pipeline only counts it, so a driver that repeats a
    step shows in the count rather than in an error."""
    real, fake = real_chain(FakeTimer()), CountingPipeline()
    step = step_at(0, *STREAMS)
    real.step(step)
    fake.step(step)
    with pytest.raises(ConfigError):
        real.step(step)
    assert fake.step(step).provenance == "2"


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize("name", sorted(TOYS))
def test_each_toy_steps_with_the_arguments_of_its_stage_protocol(
    name: str,
) -> None:
    toy = METHODS[name]
    protocol = PROTOCOLS[toy.manifest.stages]
    assert parameters(toy.step) == parameters(protocol.step), name
    assert parameters(toy.reset) == parameters(protocol.reset), name


@pytest.mark.usefixtures("toys")
def test_the_batch_toy_steps_many_like_the_batch_protocol() -> None:
    assert parameters(METHODS["toy-batch-classifier"].step_many) == parameters(
        protocols.BatchClassifier.step_many
    )


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize("name", SOUND_TOYS)
def test_a_sound_toy_passes_the_real_conformance_kit(name: str) -> None:
    assert check_method(name, facts={EXTENT: (128, 96)}) == []
