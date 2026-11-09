"""The DTC test builders emit objects the real chain accepts."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fakes.dtc_toys import registered

from evio.core.events import EventsBatch
from evio.dtc.chain import build
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.result import Pipeline
from evio.dtc.spec import CandidateFilter, PipelineSpec
from evio.dtc.types import Step, Window
from tests.support import dtc, dtc_chain
from tests.support.dtc_chain import CENTER, STEP_US, STREAMS, spec_of, step_at

pytestmark = pytest.mark.t1

importlib.import_module("evio.dtc.methods")


@pytest.fixture(autouse=True)
def toys() -> Iterator[None]:
    with registered():
        yield


def chain_for(
    spec: PipelineSpec, streams: tuple[str, ...] = STREAMS
) -> Pipeline:
    return build(
        spec, {EXTENT: (1280, 720)}, streams=streams, step_us=STEP_US
    )


def test_built_spec_and_steps_pass_the_real_chain_validation() -> None:
    chain = chain_for(spec_of("toy-detector"))
    results = [chain.step(step_at(i, "cam0")) for i in range(3)]
    assert [r.index for r in results] == [0, 1, 2]
    for result in results:
        assert [c.stream for c in result.candidates] == ["cam0"]
        x0, y0, x1, y1 = result.candidates[0].region.box
        assert x0 <= CENTER[0] <= x1
        assert y0 <= CENTER[1] <= y1


def test_a_step_without_events_in_a_stream_yields_no_candidate_there() -> None:
    chain = chain_for(spec_of("toy-detector"))
    result = chain.step(step_at(0, "cam1"))
    assert [c.stream for c in result.candidates] == ["cam1"]


def test_spec_of_builds_what_from_mapping_builds() -> None:
    built = spec_of(("toy-detector", {"boxes": 2}), "toy-tracker")
    parsed = PipelineSpec.from_mapping(
        {
            "stages": [
                {"method": "toy-detector", "config": {"boxes": 2}},
                {"method": "toy-tracker"},
            ]
        }
    )
    assert built == parsed
    assert built.candidate_filter == CandidateFilter()


def test_builders_produce_time_ordered_events_inside_their_window() -> None:
    batch = dtc.blob_events(1_000, 2_000, 40, 30, count=50, seed=3)
    t = batch.timestamps
    assert len(t) == 50
    assert (t[:-1] <= t[1:]).all()
    assert t.min() >= 1_000
    assert t.max() < 2_000
    assert len(dtc.empty_events().timestamps) == 0


def test_step_of_spaces_steps_by_step_us_and_keys_windows_by_stream() -> None:
    step = step_at(4, "cam0")
    assert (step.index, step.t_start_us, step.t_end_us) == (
        4,
        4 * STEP_US,
        5 * STEP_US,
    )
    assert set(step.windows) == set(STREAMS)
    for stream, window in step.windows.items():
        assert window.stream == stream
        assert (window.t_start_us, window.t_end_us) == (
            step.t_start_us,
            step.t_end_us,
        )


def test_a_builder_emitting_an_unbuilt_stream_is_rejected_by_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = dtc.window

    def wrong_stream(
        stream: str,
        batch: EventsBatch,
        t_start_us: int,
        t_end_us: int,
        extent: tuple[int, int] | None = (1280, 720),
    ) -> Window:
        return real(stream + "-x", batch, t_start_us, t_end_us, extent)

    monkeypatch.setattr(dtc, "window", wrong_stream)
    with pytest.raises(ConfigError, match="streams not built for"):
        chain_for(spec_of("toy-detector")).step(dtc_chain.step_at(0, "cam0"))


def test_a_builder_emitting_a_repeated_index_is_rejected_by_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = dtc.step_of

    def stuck_index(index: int, step_us: int, *windows: Window) -> Step:
        return real(0, step_us, *windows)

    monkeypatch.setattr(dtc, "step_of", stuck_index)
    chain = chain_for(spec_of("toy-detector"))
    chain.step(step_at(0, "cam0"))
    with pytest.raises(ConfigError, match="does not follow"):
        chain.step(step_at(1, "cam0"))


def test_a_builder_emitting_an_invalid_filter_is_rejected_by_the_spec() -> None:
    with pytest.raises(ValueError, match="top_k"):
        spec_of("toy-detector", candidate_filter=CandidateFilter(top_k=-1))
