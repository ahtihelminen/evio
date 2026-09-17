"""Imx636CameraSource.read_env ERC sampling: pairing, sums, driver skew.

The in-house ``imx636`` binding is absent on dev machines, so the module
import is satisfied with a stub and ``cam`` is replaced by a fake — the
sampling logic under test is pure Python. The decoder stamps the 0x014
(pre-ERC) and 0x016 (post-ERC) reports independently, so a sample is
only trustworthy when both report counters advanced in lockstep and the
timestamps sit within one count period.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.t1


@dataclass
class _Counters:
    """Mirror of the driver's ErcCounters fields."""

    in_t: int = -1
    in_count: int = 0
    in_n: int = 0
    out_t: int = -1
    out_count: int = 0
    out_n: int = 0
    count_period_us: int = 200


class _FakeCam:
    def __init__(self, samples: list[object]) -> None:
        self.samples = list(samples)
        self.erc_calls = 0

    def illumination(self) -> float:
        return 1_000.0

    def temperature(self) -> float:
        return 40.0

    def pixel_dead_time(self) -> float:
        return -1.0

    def erc_counters(self) -> object:
        self.erc_calls += 1
        sample = self.samples.pop(0)
        if isinstance(sample, Exception):
            raise sample
        return sample


def _source(monkeypatch: pytest.MonkeyPatch, cam: object) -> object:
    monkeypatch.setitem(
        sys.modules, "imx636",
        sys.modules.get("imx636", types.ModuleType("imx636")),
    )
    from evio.io.imx636_camera import Imx636CameraSource
    source = Imx636CameraSource(serial="test")
    source.cam = cam
    return source


def _pair(t: int, n: int, in_count: int, out_count: int) -> _Counters:
    """One same-period sample: lockstep counters, matching timestamps."""
    return _Counters(in_t=t, in_count=in_count, in_n=n,
                     out_t=t, out_count=out_count, out_n=n)


def test_cross_period_pair_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # in/out stamped 3 periods apart (a decode-chunk boundary between a
    # period's two reports): the sample must not fabricate a drop.
    cam = _FakeCam([
        _Counters(in_t=1000, in_count=2, in_n=5,
                  out_t=400, out_count=0, out_n=5),
    ])
    env = _source(monkeypatch, cam).read_env()
    assert "erc_in_eps" not in env
    assert "erc_out_eps" not in env


def test_stale_in_counter_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OUT advanced but IN did not (lockstep broken): a stale higher IN
    # count would read as a spurious positive drop.
    source = _source(monkeypatch, _FakeCam([
        _pair(1000, 5, 100, 100),
        _Counters(in_t=1000, in_count=100, in_n=5,
                  out_t=1200, out_count=40, out_n=6),
    ]))
    assert "erc_in_eps" in source.read_env()
    env = source.read_env()
    assert "erc_in_eps" not in env  # stale pair: proxies keep the tick


def test_rates_come_from_interval_sums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two accepted 200 us periods (100/100 then 300/200): the emitted
    # rates are the SUMS over both periods, never one extrapolated
    # period (which would read drop = 1 - 200/300 = 0.33 here).
    source = _source(monkeypatch, _FakeCam([
        _pair(1000, 1, 100, 100),
        _pair(2000, 2, 300, 200),
    ]))
    source.read_env()
    env = source.read_env()
    assert env["erc_in_eps"] == pytest.approx(400 / (2 * 200e-6))
    assert env["erc_out_eps"] == pytest.approx(300 / (2 * 200e-6))
    drop = 1.0 - env["erc_out_eps"] / env["erc_in_eps"]
    assert drop == pytest.approx(0.25)


def test_missing_erc_counters_disables_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Driver-version skew (recording-never-dies): a build without
    # erc_counters() must degrade to the proxies, never raise.
    class _BareCam:
        def illumination(self) -> float:
            return 1_000.0

        def temperature(self) -> float:
            return 40.0

        def pixel_dead_time(self) -> float:
            return -1.0

    source = _source(monkeypatch, _BareCam())
    for _ in range(2):
        env = source.read_env()
        assert env["lux"] == 1_000.0
        assert "erc_in_eps" not in env
    assert source._erc_capable is False


def test_partial_binding_disables_sampling_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A binding whose ErcCounters lacks fields raises AttributeError once;
    # sampling is then off for the session (no repeated raises).
    @dataclass
    class _Partial:
        in_t: int = 0  # no out_t / in_n / count_period_us

    cam = _FakeCam([_Partial(), _pair(1000, 1, 10, 10)])
    source = _source(monkeypatch, cam)
    assert "erc_in_eps" not in source.read_env()
    source.read_env()
    assert cam.erc_calls == 1  # disabled after the first failure


def test_runtime_error_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A USB hiccup skips one sample but keeps the capability armed.
    cam = _FakeCam([RuntimeError("usb"), _pair(1000, 1, 100, 100)])
    source = _source(monkeypatch, cam)
    assert "erc_in_eps" not in source.read_env()
    assert "erc_in_eps" in source.read_env()
    assert cam.erc_calls == 2


def test_erc_sample_below_min_count_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Near-silent camera: single-digit counts per period read as a huge
    # fabricated drop (field 2026-07-14, cam 38339 recovery deadlock).
    cam = _FakeCam([_pair(t=100, n=1, in_count=3, out_count=1)])
    src = _source(monkeypatch, cam)
    env = src.read_env()
    assert "erc_in_eps" not in env and "erc_out_eps" not in env


def test_erc_sample_above_min_count_emits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = _FakeCam([_pair(t=100, n=1, in_count=120, out_count=90)])
    src = _source(monkeypatch, cam)
    env = src.read_env()
    assert env["erc_in_eps"] > env["erc_out_eps"] > 0
