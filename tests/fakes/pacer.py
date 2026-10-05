"""Pacer-shaped doubles holding fixed HUD readings.

`FakePacer` carries the `Pacer` fields `draw_hud` reads (its wall clock is
`time.perf_counter() - _t_start`, so the test pins `perf_counter` too);
`FakeStreamingPacer` answers the `StreamingPacer` HUD API with constants.
Used by `tests/test_rendering_hud.py`, which renders the expected strings
itself and compares pixels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakePacer:
    speed: float = 2.0
    force_speed: bool = False
    _t_start: float = 100.0
    _e_start: int = 1_000_000
    instantaneous_drop_rate: float = 0.0
    average_drop_rate: float = 0.0


@dataclass(frozen=True)
class FakePacerConfig:
    speed: float = 2.0
    force_speed: bool = False


@dataclass
class FakeStreamingPacer:
    config: FakePacerConfig = FakePacerConfig()
    wall_s: float = 1.5
    rec_start_us: int = 1_000_000
    instantaneous_speed: float = 1.75
    instantaneous_drop_rate: float = 0.0
    average_drop_rate: float = 0.0

    def wall_elapsed(self) -> float:
        return self.wall_s

    def recording_elapsed(self, end_ts_us: int) -> float:
        return (end_ts_us - self.rec_start_us) / 1e6

    def actual_playback_speed(self, end_ts_us: int) -> float:
        return self.recording_elapsed(end_ts_us) / self.wall_s
