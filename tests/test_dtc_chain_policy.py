"""Faults, taps, timing and provenance of a built chain."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from fakes.dtc_timer import FakeTimer
from fakes.dtc_toys import registered

from evio.dtc.chain import _provenance, build
from evio.dtc.errors import ConfigError, StageFault
from evio.dtc.registry import METHODS
from evio.dtc.result import FaultPolicy, Pipeline
from evio.dtc.spec import CandidateFilter
from evio.dtc.taps import TapBus
from evio.dtc.timing import StageTiming
from evio.dtc.types import OK, Fault, State
from tests.support.dtc_chain import STEP_US, STREAMS, spec_of, step_at

pytestmark = pytest.mark.t1

BOTH = STREAMS
EXPLODE = ("toy-exploder", {"stream": "cam0", "at": 1})


@pytest.fixture(autouse=True)
def toys() -> Iterator[None]:
    with registered():
        yield


def make(
    *methods: str | tuple[str, dict],
    policy: FaultPolicy = FaultPolicy.FIELD,
    bus: TapBus | None = None,
    candidate_filter: CandidateFilter | None = None,
) -> Pipeline:
    return build(
        spec_of(*methods, candidate_filter=candidate_filter),
        {},
        streams=STREAMS,
        step_us=STEP_US,
        policy=policy,
        timer=FakeTimer(),
        bus=bus,
    )


def test_a_field_fault_degrades_only_the_faulting_stream_and_stage() -> None:
    pipeline = make(EXPLODE, "toy-tracker")
    pipeline.step(step_at(0, *BOTH))
    result = pipeline.step(step_at(1, *BOTH))
    assert result.faults == (
        Fault("toy-exploder", "cam0", 1, "ValueError('boom at 1')"),
    )
    assert result.status["cam0"].state is State.DEGRADED
    assert "boom at 1" in result.status["cam0"].detail
    assert result.status["cam1"] == OK
    assert [c.stream for c in result.candidates] == ["cam1"]
    assert [t.stream for t in result.tracks] == ["cam1"]


def test_a_faulted_stream_recovers_on_the_next_step() -> None:
    pipeline = make(EXPLODE, "toy-tracker")
    for index in (0, 1):
        pipeline.step(step_at(index, *BOTH))
    result = pipeline.step(step_at(2, *BOTH))
    assert result.faults == ()
    assert result.status == {"cam0": OK, "cam1": OK}
    assert len(result.tracks) == 2


def test_faults_are_counted_per_stage_and_cleared_by_reset() -> None:
    pipeline = make(EXPLODE)
    for index in range(3):
        pipeline.step(step_at(index, *BOTH))
    assert pipeline.fault_counts == {"toy-exploder": 1}
    pipeline.reset()
    assert pipeline.fault_counts == {}


def test_a_research_fault_raises_a_stage_fault_from_the_cause() -> None:
    pipeline = make(EXPLODE, policy=FaultPolicy.RESEARCH)
    pipeline.step(step_at(0, *BOTH))
    with pytest.raises(StageFault) as raised:
        pipeline.step(step_at(1, *BOTH))
    fault = raised.value
    assert (fault.stage, fault.stream, fault.step_index) == (
        "toy-exploder",
        "cam0",
        1,
    )
    assert isinstance(fault.cause, ValueError)
    assert fault.__cause__ is fault.cause


def test_nothing_is_published_when_nobody_listens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = TapBus()
    published: list[str] = []
    monkeypatch.setattr(bus, "publish", lambda name, *_: published.append(name))
    pipeline = make(EXPLODE, "toy-tracker", bus=bus)
    for index in range(2):
        pipeline.step(step_at(index, *BOTH))
    assert published == []


def test_only_the_taps_somebody_listens_to_are_published() -> None:
    bus = TapBus()
    seen: dict[str, list] = {"alarms": [], "tracks": [], "fault": []}
    for name, values in seen.items():
        bus.subscribe(name, lambda _, value, v=values: v.append(value))
    pipeline = make(EXPLODE, "toy-tracker", bus=bus)
    for index in range(2):
        result = pipeline.step(step_at(index, *BOTH))
    assert seen["alarms"] == [(), ()]
    assert seen["tracks"][-1] == result.tracks
    assert seen["fault"] == list(result.faults)


def test_every_named_tap_carries_its_stage_output() -> None:
    bus = TapBus()
    names = ("candidates", "result", "stage:toy-detector", "stage:toy-tracker")
    seen = {name: [] for name in names}
    for name, values in seen.items():
        bus.subscribe(name, lambda _, value, v=values: v.append(value))
    pipeline = make("toy-detector", "toy-tracker", bus=bus)
    result = pipeline.step(step_at(0, *BOTH))
    assert seen["result"] == [result]
    assert seen["candidates"] == [result.candidates]
    assert [len(c.items) for c in seen["stage:toy-detector"]] == [1, 1]
    assert [len(t.items) for t in seen["stage:toy-tracker"]] == [1, 1]


def test_a_tap_callback_receives_the_step_index_and_the_value() -> None:
    bus = TapBus()
    seen: list[tuple[int, object]] = []
    bus.subscribe("stage:toy-detector", lambda i, v: seen.append((i, v)))
    pipeline = make("toy-detector", bus=bus)
    pipeline.step(step_at(0, *BOTH))
    pipeline.step(step_at(1, *BOTH))
    assert [i for i, _ in seen] == [0, 0, 1, 1]


def raising_tap(_: int, __: object) -> None:
    msg = "bad subscriber"
    raise RuntimeError(msg)


def test_a_field_tap_fault_is_surfaced_and_the_step_completes() -> None:
    bus = TapBus()
    bus.subscribe("stage:toy-detector", raising_tap)
    pipeline = make("toy-detector", "toy-tracker", bus=bus)
    result = pipeline.step(step_at(0, *BOTH))
    assert [f.stage for f in result.faults] == ["tap:stage:toy-detector"] * 2
    assert pipeline.fault_counts == {"tap:stage:toy-detector": 2}
    assert len(result.tracks) == 2
    assert result.status == {"cam0": OK, "cam1": OK}
    assert len(pipeline.step(step_at(1, *BOTH)).tracks) == 2


def test_a_field_fault_in_the_result_tap_reaches_the_result() -> None:
    bus = TapBus()
    bus.subscribe("result", raising_tap)
    pipeline = make("toy-detector", bus=bus)
    result = pipeline.step(step_at(0, *BOTH))
    assert [f.stage for f in result.faults] == ["tap:result"]


def test_a_research_tap_fault_raises_a_stage_fault() -> None:
    bus = TapBus()
    bus.subscribe("stage:toy-detector", raising_tap)
    pipeline = make("toy-detector", bus=bus, policy=FaultPolicy.RESEARCH)
    with pytest.raises(StageFault) as raised:
        pipeline.step(step_at(0, *BOTH))
    assert raised.value.stage == "tap:stage:toy-detector"


def test_timings_are_summed_per_stage_over_streams_with_the_timer() -> None:
    pipeline = make("toy-detector", "toy-tracker")
    result = pipeline.step(step_at(0, *BOTH))
    assert result.timings == {
        "toy-detector": StageTiming(wall_ns=2_000, cpu_ns=800),
        "toy-tracker": StageTiming(wall_ns=2_000, cpu_ns=800),
    }


def test_stats_record_each_step_under_the_method_name() -> None:
    pipeline = make("toy-detector")
    for index in range(3):
        pipeline.step(step_at(index, *BOTH))
    stats = pipeline.stats["toy-detector"]
    assert stats.count == 3
    assert stats.percentile_ns(50) == 2_000
    assert stats.real_time_factor == 2_000 / (STEP_US * 1_000)


def test_a_joint_stage_is_timed_once_per_step() -> None:
    pipeline = make("joint")
    assert pipeline.step(step_at(0, *BOTH)).timings["joint"].wall_ns == 1_000


def test_provenance_is_stable_across_builds() -> None:
    assert make("toy-detector").provenance == make("toy-detector").provenance


@pytest.mark.parametrize(
    "other",
    [
        {"methods": (("toy-detector", {"boxes": 2}),)},
        {"methods": ("toy-detector", "toy-tracker")},
        {"methods": ("toy-detector",), "candidate_filter": CandidateFilter(5)},
    ],
)
def test_provenance_changes_with_the_spec(other: dict) -> None:
    changed = make(
        *other["methods"], candidate_filter=other.get("candidate_filter")
    )
    assert changed.provenance != make("toy-detector").provenance


def test_provenance_changes_with_a_method_version() -> None:
    before = make("toy-detector").provenance
    toy = METHODS["toy-detector"]
    toy.manifest = replace(toy.manifest, version="2")
    assert make("toy-detector").provenance != before


def test_provenance_accepts_mixed_key_types_and_nested_values() -> None:
    config = {"a": {1: "x", "b": (1, 2.5, None, True)}, 2: [{3: 4}]}
    spec = spec_of(("toy-detector", config))
    assert _provenance(spec, []) == _provenance(
        spec_of(("toy-detector", dict(config))), []
    )


def test_provenance_rejects_a_value_that_is_not_plain_data() -> None:
    spec = spec_of(("toy-detector", {"boxes": [1, object()]}))
    with pytest.raises(ConfigError, match=r"\.boxes\[1\]: object"):
        _provenance(spec, [])


def test_provenance_is_on_every_result() -> None:
    pipeline = make("toy-detector")
    assert pipeline.step(step_at(0, *BOTH)).provenance == pipeline.provenance
