"""Fidelity for the IMX636 double in `tests/fakes/imx636_camera.py`.

The real `imx636` driver is a native module that is absent off the rigs, so
the real side is its callers: `evio.streaming.stuck_pixel_scan` drives the
double through the calls it makes on the real `IMX636`, and those calls and
the behaviours the double's docstring claims are asserted here.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest
from fakes.clock import FakeClock
from fakes.imx636_camera import (
    PIXEL_MASK_SLOTS,
    FakeAntiFlickerConfig,
    FakeImx636Camera,
    FakePixelMask,
    fake_imx636_module,
)

from evio.streaming import stuck_pixel_scan

pytestmark = pytest.mark.t1

# Name and parameters of each call the driver's callers make, `self` aside.
CALLED_ON_THE_DRIVER = {
    "width": [],
    "height": [],
    "start": [],
    "poll_events": ["_timeout_ms"],
    "set_pixel_mask": ["slot", "mask"],
    "set_antiflicker": ["cfg"],
}


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "imx636", fake_imx636_module())


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock(now=100.0)
    monkeypatch.setattr(stuck_pixel_scan, "time", clock)
    return clock


@pytest.mark.parametrize(("name", "parameters"), CALLED_ON_THE_DRIVER.items())
def test_each_call_the_callers_make_exists_with_its_signature(
    name: str, parameters: list[str],
) -> None:
    signature = inspect.signature(getattr(FakeImx636Camera, name))

    assert list(signature.parameters)[1:] == parameters


def test_the_module_offers_the_names_the_callers_read() -> None:
    module = fake_imx636_module()

    assert module.PIXEL_MASK_SLOTS == stuck_pixel_scan.DEM_MAX_SLOTS
    assert module.PixelMask is FakePixelMask
    assert module.AntiFlickerConfig is FakeAntiFlickerConfig
    assert list(inspect.signature(FakePixelMask).parameters) == [
        "x", "y", "enabled",
    ]
    config = module.AntiFlickerConfig()
    assert {"enabled", "band_stop", "low_freq", "high_freq"} <= set(dir(config))


def test_poll_events_returns_status_events_and_a_third_value() -> None:
    on = np.array([[2, 0], [0, 1]])
    off = np.array([[0, 0], [3, 0]])
    camera = FakeImx636Camera(on, off)

    status, cds, extra = camera.poll_events(2)

    assert (status, extra) == (0, 0)
    assert cds.dtype.names == ("x", "y", "p")
    assert cds.dtype["p"].kind == "i"


def test_polarity_is_positive_for_on_and_not_for_off() -> None:
    camera = FakeImx636Camera(np.array([[1, 0]]), np.array([[0, 1]]))

    _, cds, _ = camera.poll_events(2)

    assert cds["p"].tolist() == [1, 0]
    assert (cds["p"] > 0).tolist() == [True, False]


def test_the_scene_is_static_ordered_by_pixel_and_untimed() -> None:
    camera = FakeImx636Camera(
        np.array([[0, 2], [1, 0]]), np.array([[1, 0], [0, 0]]),
    )

    first = camera.poll_events(2)[1]
    second = camera.poll_events(2)[1]

    assert first.tolist() == second.tolist()
    on = first[first["p"] > 0]
    assert (on["y"] * 2 + on["x"]).tolist() == sorted(
        (on["y"] * 2 + on["x"]).tolist(),
    )
    assert "t" not in first.dtype.names


def test_each_poll_moves_the_clock_by_poll_s() -> None:
    clock = FakeClock()
    camera = FakeImx636Camera.idle(4, 2, clock=clock, poll_s=0.25)

    camera.poll_events(2)
    camera.poll_events(2)

    assert clock.now == pytest.approx(0.5)


def test_the_real_scan_reads_the_rates_the_double_scripts(
    driver: None, clock: FakeClock,
) -> None:
    on = np.zeros((4, 8), dtype=np.int64)
    on[2, 5] = 100
    camera = FakeImx636Camera(on, np.zeros_like(on), clock=clock)

    result = stuck_pixel_scan.scan(camera, 1.0)

    assert camera.started
    assert result.complete
    (candidate,) = result.candidates
    assert (candidate.x, candidate.y, candidate.polarity) == (5, 2, 1)
    assert candidate.rate_eps == pytest.approx(100 / 0.01, rel=0.02)


def test_the_real_apply_masks_programs_the_dem_slots_it_records(
    driver: None,
) -> None:
    camera = FakeImx636Camera.idle(8, 4)
    candidates = [
        stuck_pixel_scan.StuckCandidate(1, 2, 10, 1000.0, 1.0, 1),
        stuck_pixel_scan.StuckCandidate(3, 0, 10, 900.0, 1.0, 0),
    ]
    applied = stuck_pixel_scan.apply_masks(camera, candidates)

    assert applied == candidates
    assert camera.pixel_masks == {
        0: FakePixelMask(1, 2, True), 1: FakePixelMask(3, 0, True),
    }


def test_set_pixel_mask_accepts_every_slot_and_rejects_the_next() -> None:
    camera = FakeImx636Camera.idle(2, 2)
    mask = FakePixelMask(0, 0, True)

    camera.set_pixel_mask(0, mask)
    camera.set_pixel_mask(PIXEL_MASK_SLOTS - 1, mask)
    with pytest.raises(ValueError, match="out of range"):
        camera.set_pixel_mask(PIXEL_MASK_SLOTS, mask)
    with pytest.raises(ValueError, match="out of range"):
        camera.set_pixel_mask(-1, mask)

    assert sorted(camera.pixel_masks) == [0, PIXEL_MASK_SLOTS - 1]


def test_the_real_notch_helper_gets_the_config_it_set(driver: None) -> None:
    camera = FakeImx636Camera.idle(2, 2)

    assert stuck_pixel_scan.set_afk_notch(camera, 31, 55)

    cfg = camera.antiflicker
    assert cfg is not None
    assert (cfg.enabled, cfg.band_stop, cfg.low_freq, cfg.high_freq) == (
        True, True, 31, 55,
    )


def test_a_rejected_config_raises_runtime_error_and_the_helper_says_false(
    driver: None,
) -> None:
    camera = FakeImx636Camera.idle(2, 2, rejects_antiflicker=True)

    with pytest.raises(RuntimeError):
        camera.set_antiflicker(FakeAntiFlickerConfig())
    assert not stuck_pixel_scan.set_afk_notch(camera)
    assert camera.antiflicker is None
