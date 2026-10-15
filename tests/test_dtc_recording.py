"""The tap recorder keeps what named taps publish, until closed."""

from __future__ import annotations

import pytest

from evio.dtc.recording import TapRecorder
from evio.dtc.taps import TapBus

pytestmark = pytest.mark.t1


def test_records_named_taps_with_their_step_index() -> None:
    bus = TapBus()
    recorder = TapRecorder(bus, ["detect", "track"])

    bus.publish("detect", 3, "d3")
    bus.publish("track", 3, "t3")
    bus.publish("detect", 4, "d4")

    assert recorder.recorded == {
        "detect": [(3, "d3"), (4, "d4")],
        "track": [(3, "t3")],
    }


def test_a_tap_published_per_stream_keeps_every_call_of_a_step() -> None:
    bus = TapBus()
    recorder = TapRecorder(bus, ["stage:detect"])

    bus.publish("stage:detect", 7, "cam0")
    bus.publish("stage:detect", 7, "cam1")

    assert recorder.recorded["stage:detect"] == [(7, "cam0"), (7, "cam1")]


def test_other_taps_are_not_recorded() -> None:
    bus = TapBus()
    recorder = TapRecorder(bus, ["detect"])

    bus.publish("verdict", 0, 1)

    assert recorder.recorded == {"detect": []}


def test_subscribing_makes_the_tap_listened_to_and_close_ends_it() -> None:
    bus = TapBus()
    recorder = TapRecorder(bus, ["detect"])
    assert bus.listening("detect")
    bus.publish("detect", 2, "kept")

    recorder.close()
    bus.publish("detect", 3, "ignored")

    assert not bus.listening("detect")
    assert recorder.recorded["detect"] == [(2, "kept")]


def test_two_recorders_record_independently() -> None:
    bus = TapBus()
    first, second = TapRecorder(bus, ["x"]), TapRecorder(bus, ["x"])

    first.close()
    bus.publish("x", 5, 1)

    assert first.recorded["x"] == []
    assert second.recorded["x"] == [(5, 1)]
