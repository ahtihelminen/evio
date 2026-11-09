"""Fidelity for `FakeClock` in `tests/fakes/clock.py`.

`FakeClock` stands in for the `time` module in the pacer, imx636 and DTC
tests. These tests run it beside the real module and fail when the surface
it models drifts from `time`. They are t2 because the real `time.sleep` is
called.
"""

from __future__ import annotations

import time

import pytest
from fakes.clock import FakeClock

pytestmark = pytest.mark.t2

MODELLED = ("perf_counter", "monotonic", "sleep")


def test_the_fake_clock_has_every_time_function_it_models() -> None:
    clock = FakeClock()
    for name in MODELLED:
        assert callable(getattr(time, name)), name
        assert callable(getattr(clock, name)), name


def test_fake_clock_reads_are_floats_like_the_real_clocks() -> None:
    clock = FakeClock()
    for name in ("perf_counter", "monotonic"):
        assert type(getattr(time, name)()) is float, name
        assert type(getattr(clock, name)()) is float, name


def test_a_sleep_returns_none_and_moves_both_clocks_forward() -> None:
    """Deliberate difference: the real sleep overshoots, the fake advances
    `now` by exactly the time asked, so only the direction is compared."""
    clock = FakeClock()
    real_before = {n: getattr(time, n)() for n in ("perf_counter", "monotonic")}
    fake_before = {n: getattr(clock, n)() for n in real_before}
    assert time.sleep(0.001) is None
    assert clock.sleep(0.001) is None
    for name, before in real_before.items():
        assert getattr(time, name)() > before, name
        assert getattr(clock, name)() > fake_before[name], name


def test_a_sleep_of_zero_is_accepted_by_both() -> None:
    clock = FakeClock()
    assert time.sleep(0) is None
    assert clock.sleep(0) is None
    assert clock.now > 0
