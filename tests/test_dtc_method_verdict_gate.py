"""The verdict-gate decision counts fresh verdicts, prunes by live ids."""

from __future__ import annotations

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.methods.verdict_gate import VerdictGate, VerdictGateConfig
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS
from evio.dtc.types import Alarm, Region, Track, Tracks, Verdict, Verdicts
from tests.support import dtc

pytestmark = pytest.mark.t1

STEP = dtc.step_of(0, 10_000, dtc.window("cam", dtc.empty_events(), 0, 10_000))
TRACK = Track("cam", "t", Region(point=(1.0, 1.0)), 1.0)
LIVE = Tracks((TRACK,), frozenset({"t"}))


def build(**config: object) -> VerdictGate:
    context = Context("cam", ("cam",), {EXTENT: (64, 48)}, 10_000)
    return VerdictGate(VerdictGateConfig(**config), context)


def decide(
    gate: VerdictGate,
    probability: float,
    age_steps: int = 0,
    tracks: Tracks = LIVE,
) -> tuple[Alarm, ...]:
    verdicts = Verdicts((Verdict("cam", "t", probability, age_steps),))
    return gate.step(STEP, tracks, verdicts, {"kinematics": {}}).items


def test_registered_as_decide_with_kinematics_evidence() -> None:
    manifest = METHODS["verdict-gate"].manifest
    assert manifest.stages == (Stage.DECIDE,)
    assert manifest.evidence == ("kinematics",)


def test_mofk_fires_on_second_fresh_high_verdict() -> None:
    gate = build()
    assert decide(gate, 0.9) == ()
    assert decide(gate, 0.9) == (Alarm("cam", "t", 0.0),)


def test_stale_verdicts_are_not_counted() -> None:
    gate = build()
    decide(gate, 0.9)
    assert decide(gate, 0.9, age_steps=3) == ()
    assert decide(gate, 0.9, age_steps=4) == ()


def test_fired_track_keeps_alarming_on_stale_steps() -> None:
    gate = build(k=1, m=1)
    assert decide(gate, 0.9)
    assert decide(gate, 0.9, age_steps=1)


def test_low_verdict_clears_the_alarm() -> None:
    gate = build(k=1, m=1)
    decide(gate, 0.9)
    assert decide(gate, 0.1) == ()


def test_track_no_longer_live_is_pruned() -> None:
    gate = build(k=2, m=2)
    decide(gate, 0.9)
    decide(gate, 0.1, tracks=Tracks((TRACK,), frozenset()))
    assert decide(gate, 0.9) == ()


def test_sprt_alarm_carries_the_accumulated_score() -> None:
    gate = build(kind="sprt")
    alarms: tuple[Alarm, ...] = ()
    for _ in range(5):
        alarms = decide(gate, 0.99)
    assert len(alarms) == 1
    assert alarms[0].score > 0.0


def test_reset_forgets_gate_state() -> None:
    gate = build()
    decide(gate, 0.9)
    gate.reset()
    assert decide(gate, 0.9) == ()


def test_reset_clears_the_firing_alarm() -> None:
    gate = build(k=1, m=1)
    decide(gate, 0.9)
    gate.reset()
    assert decide(gate, 0.9, age_steps=1) == ()


def test_fired_track_that_is_no_longer_live_stops_alarming() -> None:
    gate = build(k=1, m=1)
    decide(gate, 0.9)
    assert decide(gate, 0.9, age_steps=1, tracks=Tracks()) == ()


def test_streams_do_not_share_gate_state() -> None:
    a, b = build(), build()
    decide(a, 0.9)
    assert decide(b, 0.9) == ()


@pytest.mark.parametrize(
    "config", [{"kind": "nope"}, {"kind": "fusion"}], ids=["kind", "no-path"]
)
def test_bad_gate_config_fails_at_build(config: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        build(**config)


def test_passes_the_conformance_kit() -> None:
    assert check_method("verdict-gate", facts={EXTENT: (128, 96)}) == []
