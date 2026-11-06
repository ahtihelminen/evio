"""An IMX636 stand-in that serves scripted per-pixel event counts.

`FakeImx636Camera` replays the same `on`/`off` count grids (shape
`(height, width)`) on every `poll_events`, and each poll moves the shared
`FakeClock` on by `poll_s`, so a pixel's rate is `count / poll_s` ev/s. It
records what `stuck_pixel_scan` programs: DEM slots in `pixel_masks`, the
accepted anti-flicker config in `antiflicker`. `fake_imx636_module()` is the
driver module (`PixelMask`, `AntiFlickerConfig`, `PIXEL_MASK_SLOTS`) to put
in `sys.modules["imx636"]`.

Behaviours assumed of the real driver, not proved here (that is SN-586):
- `poll_events(timeout_ms)` returns `(status, cds, _)`, `status < 0` on a
  USB error, `cds` a structured array with `x`, `y`, `p` (`p > 0` is ON);
- the scene is static: every poll carries the same counts, events in a
  batch are ordered by pixel, not by time, and there is no timestamp field;
- `set_pixel_mask(slot, mask)` accepts slots `0..PIXEL_MASK_SLOTS-1` and
  raises `ValueError` past them;
- `set_antiflicker(cfg)` raises `RuntimeError` to reject a config.

Differences from the real device:
- the scene is static and has no timestamps, where the sensor reports a
  moving scene with a clock;
- a poll costs exactly `poll_s` of fake time and never blocks or times out;
- the assumed behaviours above are not checked against the driver.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import numpy as np
from fakes.clock import FakeClock

PIXEL_MASK_SLOTS = 64
EVENT_DTYPE = np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.int16)])


@dataclass(frozen=True)
class FakePixelMask:
    x: int
    y: int
    enabled: bool


class FakeAntiFlickerConfig:
    enabled = False
    band_stop = False
    low_freq = 0
    high_freq = 0


def fake_imx636_module() -> types.ModuleType:
    module = types.ModuleType("imx636")
    module.PixelMask = FakePixelMask
    module.AntiFlickerConfig = FakeAntiFlickerConfig
    module.PIXEL_MASK_SLOTS = PIXEL_MASK_SLOTS
    return module


def _events(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    """One batch: each pixel repeated by its count, ON then OFF."""
    width = on.shape[1]
    batches = []
    for counts, polarity in ((on, 1), (off, 0)):
        flat = np.repeat(np.arange(counts.size), counts.ravel())
        batch = np.empty(flat.size, dtype=EVENT_DTYPE)
        batch["x"], batch["y"] = flat % width, flat // width
        batch["p"] = polarity
        batches.append(batch)
    return np.concatenate(batches)


@dataclass
class FakeImx636Camera:
    on: np.ndarray
    off: np.ndarray
    clock: FakeClock = field(default_factory=FakeClock)
    poll_s: float = 0.01
    rejects_antiflicker: bool = False
    started: bool = False
    pixel_masks: dict[int, FakePixelMask] = field(default_factory=dict)
    antiflicker: FakeAntiFlickerConfig | None = None

    @classmethod
    def idle(
        cls, width: int, height: int, **kwargs: object,
    ) -> FakeImx636Camera:
        """A camera seeing nothing, for the tests that only program it."""
        dark = np.zeros((height, width), dtype=np.int64)
        return cls(dark, dark, **kwargs)

    def width(self) -> int:
        return self.on.shape[1]

    def height(self) -> int:
        return self.on.shape[0]

    def start(self) -> None:
        self.started = True

    def poll_events(self, _timeout_ms: int) -> tuple[int, np.ndarray, int]:
        self.clock.now += self.poll_s
        return 0, _events(self.on, self.off), 0

    def set_pixel_mask(self, slot: int, mask: FakePixelMask) -> None:
        if not 0 <= slot < PIXEL_MASK_SLOTS:
            message = f"DEM slot {slot} out of range"
            raise ValueError(message)
        self.pixel_masks[slot] = mask

    def set_antiflicker(self, cfg: FakeAntiFlickerConfig) -> None:
        if self.rejects_antiflicker:
            message = "anti-flicker config rejected"
            raise RuntimeError(message)
        self.antiflicker = cfg
