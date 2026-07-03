"""IMX636 event-camera source — the only live-camera path in evio.

Context manager opens/closes the device; `__next__()` returns an
`EventsBatch` over the configured rolling window.

Backend: the in-house `imx636` Python module (built from
github.com/sensofusion/IMX636 → `~/Documents/IMX636`), which talks
directly to the Cypress FX3 / CX3 USB bridge via libusb-1.0. No
Prophesee / Metavision / openeb dependency at runtime.

The IMX636 driver is push-style (registers a CD callback, host calls
`poll()` to pump USB transfers); we adapt to evio's pull-style iterator
by accumulating callback deliveries into a deque and draining it in
`__next__()`.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import imx636  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.core.window import RollingWindowBatchProcessor


def _empty_batch() -> EventsBatch:
    return EventsBatch(
        timestamps=np.zeros(0, dtype=np.int64),
        x_coords=np.zeros(0, dtype=np.int16),
        y_coords=np.zeros(0, dtype=np.int16),
        polarities=np.zeros(0, dtype=np.int8),
    )


def get_camera_serials() -> list[str]:
    """Return the real serial number of each connected IMX636 camera.

    Sorted ascending; uniquely identifies each camera regardless of USB
    port (Treuzell GET_SERIAL — see imx636 driver). Any camera that
    reports an empty serial here means its EEPROM/firmware is older than
    the Treuzell read path landed in (commit 4dc6006 on the imx636 repo)
    — rebuild the driver on the host before deploying.
    """
    serials = [s for s in imx636.IMX636.list_cameras() if s]
    return sorted(serials)


@dataclass()
class Imx636CameraSource:
    serial: str = ""
    window_length_us: int = 10_000  # 10 ms rolling window
    step_us: int | None = None
    # Optional band-stop filter for mains-lighting flicker (50/60 Hz +
    # ballast harmonics). Useful indoors under fluorescents/LEDs;
    # outdoor scenes generally don't need it and a wide notch can clip
    # real motion. Off by default — opt in per source.
    anti_flicker: bool = False
    # Optional Event Rate Controller — caps events/sec at the sensor
    # before they leave the chip. Off by default (0); set to a positive
    # value (e.g. 1_000_000 = 1 M ev/sec/cam) for high-clutter scenes
    # where downstream consumers can't keep up.
    erc_event_rate: int = 0
    # DEM masks are calibration data, stored per serial in config/dem_masks/
    # <serial>.json (factory-vetted by `scan_stuck_pixels.py --apply`, then
    # maintained by the live autotuner). The on-sensor DEM is volatile (lost on
    # close), so this manifest is re-programmed at every open — that reload IS
    # the mask persistence mechanism. None -> the default config/dem_masks/ dir.
    dem_mask_dir: Path | None = None

    # Software stuck-pixel masks (the tier beyond the 64 DEM slots), parsed
    # from the same config file at open / boot scan. The camera worker seeds
    # its post-record event-drop bitmap from this; the driver itself never
    # filters (recording must stay a lossless record of the sensor stream).
    soft_mask_coords: list[tuple[int, int]] = field(
        init=False, default_factory=list,
    )

    cam: Any = field(init=False, default=None)
    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)
    decoded_queue: deque[NDArray[np.void]] = field(
        init=False, default_factory=deque,
    )
    window_queue: deque[EventsBatch] = field(init=False, default_factory=deque)
    _processor: RollingWindowBatchProcessor | None = field(
        init=False, default=None,
    )
    _processor_window_us: int = field(init=False, default=0)
    _processor_step_us: int = field(init=False, default=0)
    _streaming: bool = field(init=False, default=False)

    def __enter__(self) -> Imx636CameraSource:
        # ONLY claim the USB device + sensor init writes here. The
        # actual streaming start happens lazily on first __next__().
        # Why: when this object is constructed under a multi-process
        # camera_open_lock, holding the lock through cam.start() means
        # the next worker's sensor-init register writes contend with
        # the prior camera's streaming traffic on the xHCI controller
        # — ISSD writes start failing on the 3rd+ camera. Mirror the
        # view_event_cameras.py pattern: open + config inside the
        # lock, cam.start() outside.
        real_serial = (
            "" if self.serial.startswith("event-") else self.serial
        )
        self.cam = imx636.IMX636.open(real_serial)
        self.width = int(self.cam.width())
        self.height = int(self.cam.height())
        self.cam.set_cd_callback(self._on_cd_events)
        if self.anti_flicker:
            af = imx636.AntiFlickerConfig()
            af.enabled = True
            af.band_stop = True
            af.low_freq = 50
            af.high_freq = 500
            af.duty_cycle = 50
            self.cam.set_antiflicker(af)
        if self.erc_event_rate > 0:
            erc = imx636.ERCConfig()
            erc.enabled = True
            erc.event_rate = int(self.erc_event_rate)
            self.cam.set_erc(erc)
        self._reload_dem_masks()
        return self

    def _reload_dem_masks(self) -> None:
        """Re-program the operator-vetted DEM masks for this serial, if any."""
        if self.cam is None or self.serial.startswith("event-"):
            return
        mask_dir = self.dem_mask_dir or Path("config/dem_masks")
        path = mask_dir / f"{self.serial}.json"
        if not path.exists():
            return
        import json
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        masks = data.get("masks", [])
        applied = 0
        for idx, m in enumerate(masks[:64]):
            # Program each pixel into its stored DEM slot (one stable slot map,
            # shared with the main-process registry); fall back to the list
            # index for a v1 manifest that predates the slot field.
            slot = int(m.get("slot", idx))
            with contextlib.suppress(RuntimeError, ValueError, AttributeError,
                                     KeyError):
                self.cam.set_pixel_mask(
                    slot, imx636.PixelMask(int(m["x"]), int(m["y"]), True))
                applied += 1
        # Software tier (schema v3): surfaced for the camera worker's
        # post-record drop bitmap; nothing to program on-sensor.
        self.soft_mask_coords = [
            (int(m["x"]), int(m["y"]))
            for m in data.get("soft_masks", [])
            if "x" in m and "y" in m
        ]
        if applied or self.soft_mask_coords:
            print(
                f"[imx636] reloaded {applied} DEM mask(s) + "
                f"{len(self.soft_mask_coords)} software mask(s) "
                f"for {self.serial}",
                flush=True,
            )

    def boot_scan_stuck_pixels(self, duration_s: float) -> int:
        """Release all DEM masks, watch the raw stream, mask fresh flooders.

        Stuck oscillators DRIFT with the bias operating point (offline
        2026-07-02: mask sets from 12 s slices 18 min apart overlap at
        Jaccard ~0.02), so the config masks a previous session promoted
        usually miss today's flooders. At every camera open this releases
        the stale masks, counts the raw per-pixel stream for ``duration_s``,
        re-masks the current offenders (``boot_candidates`` gates) and
        rewrites ``config/dem_masks/<serial>.json`` -- which the autotuner's
        registry then seeds from (its lazy first-observation read), keeping
        the main-process slot map in sync. On any failure the old config is
        re-programmed so a broken scan never leaves the DEM unmasked.
        """
        if self.cam is None or duration_s <= 0:
            return 0
        from evio.streaming.stuck_pixel_scan import (
            DEM_MAX_SLOTS,
            SOFT_MASK_MAX,
            boot_candidates,
            write_applied_masks,
        )
        self._ensure_streaming()
        try:
            for slot in range(DEM_MAX_SLOTS):  # fresh eyes on masked pixels
                with contextlib.suppress(RuntimeError, ValueError,
                                         AttributeError, KeyError):
                    self.cam.disable_pixel_mask(slot)
            n_px = self.width * self.height
            on = np.zeros(n_px, dtype=np.int64)
            off = np.zeros(n_px, dtype=np.int64)
            # Drain the pre-scan backlog so a buffered dump can't inflate
            # the counts against the timed window (mirrors scan()).
            drain_until = time.monotonic() + 0.2
            while time.monotonic() < drain_until:
                self.cam.poll()
            self.decoded_queue.clear()
            complete = True
            start = time.monotonic()
            while time.monotonic() - start < duration_s:
                if self.cam.poll() < 0:  # USB error: abort, restore below
                    complete = False
                    break
                while self.decoded_queue:
                    buf = self.decoded_queue.popleft()
                    if buf.size == 0:
                        continue
                    flat = (buf["y"].astype(np.intp) * self.width
                            + buf["x"].astype(np.intp))
                    is_on = buf["p"] > 0
                    on += np.bincount(flat[is_on], minlength=n_px)
                    off += np.bincount(flat[~is_on], minlength=n_px)
            if not complete:
                print(
                    f"[warn] boot scan cut short (USB) serial={self.serial}; "
                    "restoring config masks", flush=True,
                )
                self._reload_dem_masks()
                return 0
            elapsed = max(time.monotonic() - start, 1e-6)
            candidates, diag = boot_candidates(
                on, off, elapsed, self.width,
                max_masks=DEM_MAX_SLOTS + SOFT_MASK_MAX,
            )
            # Top-rate candidates take the 64 hardware slots (DEM saves
            # ERC/USB transport budget); the remainder become software
            # masks the camera worker drops post-record.
            applied = candidates[:DEM_MAX_SLOTS]
            soft = candidates[DEM_MAX_SLOTS:]
            for slot, c in enumerate(applied):
                self.cam.set_pixel_mask(
                    slot, imx636.PixelMask(c.x, c.y, True))
            self.soft_mask_coords = [(c.x, c.y) for c in soft]
            mask_dir = self.dem_mask_dir or Path("config/dem_masks")
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_applied_masks(
                mask_dir / f"{self.serial}.json", self.serial, applied,
                now_iso, source="boot_scan", soft=soft)
            total = int(on.sum() + off.sum())
            print(
                f"[imx636] boot scan {self.serial}: masked {len(applied)} "
                f"px (+{len(soft)} software) of {total / elapsed:.0f} ev/s "
                f"(ambient={diag['ambient_eps']:.0f} "
                f"floor={diag['floor_eps']:.0f} "
                f"spared_mixed={diag['over_floor_mixed']})",
                flush=True,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(
                f"[warn] boot scan failed serial={self.serial} err={exc}; "
                "restoring config masks", flush=True,
            )
            self._reload_dem_masks()
            return 0
        else:
            self.decoded_queue.clear()  # scan spill must not enter the pipe
            return len(applied)

    def _ensure_streaming(self) -> None:
        if not self._streaming and self.cam is not None:
            self.cam.start()
            self._streaming = True

    def read_env(self) -> dict[str, float]:
        """Sample the on-chip environment: lux, die temperature, refractory.

        Slow-cadence feedforward for the adaptive bias controller (regime
        night decision + refr thermal pre-emption) and capture provenance.
        Each read is an I2C/ADC transaction over USB that can time out
        (``illumination``/``pixel_dead_time`` return -1) or, on an
        unreachable device, raise; failures are skipped so a transient miss
        just leaves the controller on its last good value. A -1 lux is
        forwarded as-is — the controller treats negative lux as 'unknown'.
        """
        env: dict[str, float] = {}
        if self.cam is None:
            return env
        with contextlib.suppress(RuntimeError):
            env["lux"] = float(self.cam.illumination())
        with contextlib.suppress(RuntimeError):
            env["die_temp_c"] = float(self.cam.temperature())
        with contextlib.suppress(RuntimeError):
            dead = float(self.cam.pixel_dead_time())
            if dead >= 0.0:
                env["pixel_dead_time_us"] = dead
        return env

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self.cam is None or not self._streaming:
            return
        # cam.stop() writes the ISSD_STOP register sequence over USB.
        # If we're tearing down because of a prior USB error (poll()
        # returned -1) the device is unreachable and the very first
        # register write throws "ISSD write_field failed at step 0".
        # Cleanup should be best-effort here — swallow the secondary
        # failure so the worker exits cleanly.
        with contextlib.suppress(RuntimeError):
            self.cam.stop()

    def __iter__(self) -> Imx636CameraSource:
        return self

    def _on_cd_events(self, arr: NDArray[np.void]) -> None:
        # Callback fires from poll(); arr is dtype imx636.CD_DTYPE
        # with fields (x int16, y int16, p int8, _pad u1x3, t int64).
        if arr.size:
            self.decoded_queue.append(arr.copy())

    def _ensure_processor(self) -> None:
        window_us = max(1, int(self.window_length_us))
        step_us = max(1, int(self.step_us or window_us))
        if (
            self._processor is None
            or window_us != self._processor_window_us
            or step_us != self._processor_step_us
        ):
            self._processor = RollingWindowBatchProcessor(window_us, step_us)
            self._processor_window_us = window_us
            self._processor_step_us = step_us
            self.window_queue.clear()

    def __next__(self) -> EventsBatch:
        if self.cam is None:
            msg = "Imx636CameraSource not started; use as a context manager."
            raise RuntimeError(msg)
        self._ensure_streaming()
        self._ensure_processor()

        # Pace at the configured window so a static scene still ticks the
        # iterator (no events ⇒ empty batch ⇒ downstream renderer paints
        # baseline). The idle clock RESETS every time poll() actually
        # delivers data — otherwise events that arrived but didn't form
        # a complete window before the first deadline get stranded in
        # the processor and "come and go" between calls.
        idle_budget_ns = int(self.window_length_us) * 1000
        last_data_ns = time.monotonic_ns()

        while True:
            # Drain the driver pipe COMPLETELY before returning anything.
            # poll() consumes ONE ready slot per call, and the async reader
            # STALLS when its ring fills — the ring is only SLOTS x 20 ms
            # deep in TIME at low rates, so a consumer that returns
            # per-window while slots sit parked lets the device FIFO shed
            # the stream on-chip (2026-07-02: ~95% of all events lost this
            # way; the recorded stream was 4 ms islands every ~200 ms).
            # Draining every call keeps the ring empty no matter how slowly
            # the worker loop ticks; the windows queue absorbs the backlog
            # and drain_window_backlog still records every stale window.
            while True:
                r = self.cam.poll()
                if r < 0:
                    msg = "imx636: USB error from poll()"
                    raise RuntimeError(msg)
                if r != 1:
                    break

            if not self.decoded_queue and not self.window_queue:
                if time.monotonic_ns() - last_data_ns >= idle_budget_ns:
                    return _empty_batch()
                continue
            if self.decoded_queue:
                # Got fresh data — defer the empty timeout.
                last_data_ns = time.monotonic_ns()

            # Drain EVERY decoded chunk now pending, not just one, so
            # window_queue reflects the true backlog. Under a storm a single
            # poll() fires the callback many times; processing one chunk per
            # __next__ left the rest hidden in decoded_queue, where the
            # worker's drop-to-latest (it only inspects window_queue) couldn't
            # reach them and the displayed frame drifted seconds behind.
            # Draining them all lets drain_window_backlog collapse to the
            # freshest window (and record the stale ones).
            while self.decoded_queue:
                buffer = self.decoded_queue.popleft()
                if buffer.size == 0 or self._processor is None:
                    continue
                batch = EventsBatch(
                    x_coords=buffer["x"].astype(np.int16, copy=False),
                    y_coords=buffer["y"].astype(np.int16, copy=False),
                    timestamps=buffer["t"].astype(np.int64, copy=False),
                    polarities=buffer["p"].astype(np.int8, copy=False),
                )
                for window in self._processor.process((batch,)):
                    self.window_queue.append(window)

            if self.window_queue:
                return self.window_queue.popleft()
