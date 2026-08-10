"""Pane workers: live IMX636 cameras, .dat replay, synthetic scenes.

Each worker owns one camera/file/generator, scores every rolling window
(default 10 ms = 100 Hz) and publishes results into its ``PaneState``,
which the UI thread snapshots for rendering. All PaneState mutation
happens under ``PaneState.lock``; compute happens outside it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evio.focus.metric import FocusResult, ScoreHistory, WindowScorer

SENSOR_W = 1280
SENSOR_H = 720
_ENV_EVERY_WINDOWS = 100  # ~1 Hz die-temp/lux sampling at 100 Hz windows
_REVIVE_ATTEMPTS = 2  # rotor-box lesson: >2 is hopeless until physical fix
_DT8_X_MASK = 0x3FFF
_DT8_Y_SHIFT = 14
_DT8_P_SHIFT = 28

# Sensor-init register writes contend on the USB controller when several
# cameras open concurrently (ISSD failures on the 3rd+ camera) — open +
# configure one camera at a time; streaming may overlap freely.
_OPEN_LOCK = threading.Lock()


@dataclass
class PaneConfig:
    window_ms: float = 10.0
    roi_size: int = 384
    clip: float = 10.0
    hf_cutoff: float = 0.25
    target_events: float = 4000.0
    view_ds: int = 4
    view_decay: float = 0.65
    boot_scan_s: float = 3.0
    history_s: float = 60.0

    @property
    def window_s(self) -> float:
        return self.window_ms / 1000.0

    def history_capacity(self) -> int:
        return max(256, int(self.history_s * 1000.0 / self.window_ms))


@dataclass
class PaneSnapshot:
    """Immutable copy of a pane's state for the render thread."""

    serial: str
    status: str
    view: NDArray[np.float32] | None
    roi: tuple[int, int, int, int]
    raw: NDArray[np.float32]
    ema: NDArray[np.float32]
    last: FocusResult | None
    ema_now: float
    peak_ema: float
    die_temp_c: float | None
    windows: int


class PaneState:
    """Mutable per-camera state shared between worker and UI threads."""

    def __init__(self, serial: str, config: PaneConfig) -> None:
        self.lock = threading.Lock()
        self.serial = serial
        self.config = config
        self.status = "starting"
        self.roi = _centered_roi(config.roi_size)
        self.history = ScoreHistory(capacity=config.history_capacity())
        self.view: NDArray[np.float32] | None = None
        self.last: FocusResult | None = None
        self.die_temp_c: float | None = None
        self.windows = 0

    def snapshot(self, graph_samples: int) -> PaneSnapshot:
        with self.lock:
            raw, ema = self.history.recent(graph_samples)
            return PaneSnapshot(
                serial=self.serial,
                status=self.status,
                view=None if self.view is None else self.view.copy(),
                roi=self.roi,
                raw=raw,
                ema=ema,
                last=self.last,
                ema_now=self.history.ema,
                peak_ema=self.history.peak_ema,
                die_temp_c=self.die_temp_c,
                windows=self.windows,
            )

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def move_roi(self, dx: int, dy: int) -> None:
        with self.lock:
            self.roi = _clamp_roi(
                self.roi[0] + dx, self.roi[1] + dy,
                self.roi[2], self.roi[3],
            )

    def resize_roi(self, delta: int) -> None:
        with self.lock:
            rx, ry, rw, rh = self.roi
            size = int(np.clip(rw + delta, 64, min(SENSOR_W, SENSOR_H)))
            cx, cy = rx + rw // 2, ry + rh // 2
            self.roi = _clamp_roi(
                cx - size // 2, cy - size // 2, size, size,
            )

    def center_roi_on_activity(self) -> None:
        """Recenter the ROI on the brightest view block (delta-locate)."""
        with self.lock:
            if self.view is None or float(self.view.max()) <= 0.0:
                return
            ds = self.config.view_ds
            by, bx = np.unravel_index(
                int(self.view.argmax()), self.view.shape,
            )
            cx, cy = int(bx) * ds + ds // 2, int(by) * ds + ds // 2
            _, _, rw, rh = self.roi
            self.roi = _clamp_roi(cx - rw // 2, cy - rh // 2, rw, rh)

    def reset_peak(self) -> None:
        with self.lock:
            self.history.reset_peak()


def _centered_roi(size: int) -> tuple[int, int, int, int]:
    size = min(size, SENSOR_H)
    return ((SENSOR_W - size) // 2, (SENSOR_H - size) // 2, size, size)


def _clamp_roi(
    rx: int, ry: int, rw: int, rh: int,
) -> tuple[int, int, int, int]:
    rw, rh = min(rw, SENSOR_W), min(rh, SENSOR_H)
    rx = int(np.clip(rx, 0, SENSOR_W - rw))
    ry = int(np.clip(ry, 0, SENSOR_H - rh))
    return rx, ry, rw, rh


class PaneWorker(threading.Thread):
    """Base worker: consumes event windows, publishes scores + view."""

    def __init__(self, pane: PaneState, stop: threading.Event) -> None:
        super().__init__(daemon=True, name=f"pane-{pane.serial}")
        self.pane = pane
        self.stop_event = stop
        self.stuck_mask: NDArray[np.bool_] | None = None
        cfg = pane.config
        self._view_shape = (
            SENSOR_H // cfg.view_ds, SENSOR_W // cfg.view_ds,
        )
        self._view_acc = np.zeros(self._view_shape, dtype=np.float32)
        self._scorer = WindowScorer(
            window_s=cfg.window_s,
            clip=cfg.clip,
            hf_cutoff=cfg.hf_cutoff,
            target_events=cfg.target_events,
        )

    def _consume(
        self,
        x_coords: NDArray[np.integer],
        y_coords: NDArray[np.integer],
    ) -> None:
        cfg = self.pane.config
        with self.pane.lock:
            roi = self.pane.roi
        result = self._scorer.score(
            x_coords, y_coords, roi=roi, stuck_mask=self.stuck_mask,
        )
        view_h, view_w = self._view_shape
        flat = (
            (y_coords.astype(np.intp) // cfg.view_ds) * view_w
            + x_coords.astype(np.intp) // cfg.view_ds
        )
        counts = np.bincount(flat, minlength=view_h * view_w)
        self._view_acc *= cfg.view_decay
        self._view_acc += counts.astype(np.float32).reshape(
            self._view_shape,
        )
        with self.pane.lock:
            self.pane.history.push(result.hf)
            self.pane.last = result
            self.pane.view = self._view_acc
            self.pane.windows += 1


class LiveCameraWorker(PaneWorker):
    """One live IMX636 camera via the in-house driver.

    Anti-flicker is NEVER enabled here — the AFK band-stop deletes
    exactly the modulated stimuli this tool depends on. Biases are not
    touched; default biases are fine for focusing.
    """

    def run(self) -> None:
        from evio.io.imx636_camera import Imx636CameraSource

        cfg = self.pane.config
        for attempt in range(1 + _REVIVE_ATTEMPTS):
            if self.stop_event.is_set():
                return
            if attempt:
                self.pane.set_status(f"reviving ({attempt})")
                time.sleep(2.0)
            source = Imx636CameraSource(
                serial=self.pane.serial,
                window_length_us=int(cfg.window_ms * 1000),
            )
            try:
                self._stream(source)
            except RuntimeError as exc:
                self.pane.set_status(f"USB dead: {exc}")
            else:
                return  # clean stop
        self.pane.set_status("dead (revive attempts exhausted)")

    def _stream(self, source: object) -> None:
        cfg = self.pane.config
        self.pane.set_status("opening")
        with _OPEN_LOCK:
            source.__enter__()
        try:
            if cfg.boot_scan_s > 0:
                self.pane.set_status(
                    f"boot scan {cfg.boot_scan_s:.0f}s (keep scene still)",
                )
                source.boot_scan_stuck_pixels(cfg.boot_scan_s)
            self.stuck_mask = _mask_from_coords(source.soft_mask_coords)
            self.pane.set_status("live (AFK off)")
            since_env = 0
            for batch in source:
                if self.stop_event.is_set():
                    return
                self._consume(batch.x_coords, batch.y_coords)
                since_env += 1
                if since_env >= _ENV_EVERY_WINDOWS:
                    since_env = 0
                    env = source.read_env()
                    if "die_temp_c" in env:
                        with self.pane.lock:
                            self.pane.die_temp_c = env["die_temp_c"]
        finally:
            source.__exit__(None, None, None)


def _mask_from_coords(
    coords: list[tuple[int, int]],
) -> NDArray[np.bool_] | None:
    if not coords:
        return None
    mask = np.zeros((SENSOR_H, SENSOR_W), dtype=bool)
    for x, y in coords:
        mask[y, x] = True
    return mask


class ReplayWorker(PaneWorker):
    """Loops a DT8 .dat recording at wall-clock pace (for demo/tests)."""

    def __init__(
        self,
        pane: PaneState,
        stop: threading.Event,
        path: Path,
        speed: float = 1.0,
    ) -> None:
        super().__init__(pane, stop)
        self.path = path
        self.speed = speed

    def run(self) -> None:
        from evio.io.dat_file import DatFileSource

        cfg = self.pane.config
        window_us = int(cfg.window_ms * 1000)
        try:
            source = DatFileSource(
                str(self.path),
                window_length_us=window_us,
                step_us=window_us,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.pane.set_status(f"replay failed: {exc}")
            return
        words = source.event_words
        order = source.order
        self.pane.set_status(f"replay {self.path.name}")
        pace = cfg.window_s / self.speed
        while not self.stop_event.is_set():
            for rng in source.ranges():
                if self.stop_event.is_set():
                    return
                begin = time.monotonic()
                window_words = np.asarray(
                    words[order[rng.start_idx : rng.end_idx]],
                    dtype=np.uint32,
                )
                self._consume(
                    (window_words & _DT8_X_MASK).astype(np.int32),
                    (
                        (window_words >> _DT8_Y_SHIFT) & _DT8_X_MASK
                    ).astype(np.int32),
                )
                leftover = pace - (time.monotonic() - begin)
                if leftover > 0:
                    time.sleep(leftover)


@dataclass
class SyntheticScene:
    """Blinking dot grid whose defocus breathes over ``sweep_period_s``.

    Emulates an operator sweeping the focus ring through best focus:
    the score should peak twice per period, letting the whole UI be
    exercised (and benchmarked) with no hardware attached.
    """

    dots: int = 8
    blink_hz: float = 25.0
    events_per_window: int = 20_000
    max_blur_px: float = 6.0
    sweep_period_s: float = 8.0
    stuck_pixels: tuple[tuple[int, int], ...] = ()
    stuck_rate_per_window: int = 0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _centers: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        gx = np.linspace(SENSOR_W * 0.3, SENSOR_W * 0.7, self.dots)
        gy = np.linspace(SENSOR_H * 0.3, SENSOR_H * 0.7, self.dots)
        self._centers = np.stack(
            np.meshgrid(gx, gy), axis=-1,
        ).reshape(-1, 2)

    def window_events(
        self, t_s: float, window_s: float,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        phase = (t_s * self.blink_hz) % 1.0
        xs_parts: list[NDArray[np.int32]] = []
        ys_parts: list[NDArray[np.int32]] = []
        if phase < 0.5:  # stimulus ON half-cycle emits events
            blur = self.max_blur_px * 0.5 * (
                1.0 - np.cos(2.0 * np.pi * t_s / self.sweep_period_s)
            )
            picks = self._rng.integers(
                0, self._centers.shape[0], self.events_per_window,
            )
            pos = self._centers[picks] + self._rng.normal(
                scale=blur + 0.3, size=(self.events_per_window, 2),
            )
            xs_parts.append(
                np.clip(pos[:, 0], 0, SENSOR_W - 1).astype(np.int32),
            )
            ys_parts.append(
                np.clip(pos[:, 1], 0, SENSOR_H - 1).astype(np.int32),
            )
        for sx, sy in self.stuck_pixels:
            n_stuck = self.stuck_rate_per_window
            xs_parts.append(np.full(n_stuck, sx, dtype=np.int32))
            ys_parts.append(np.full(n_stuck, sy, dtype=np.int32))
        if not xs_parts:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty
        return np.concatenate(xs_parts), np.concatenate(ys_parts)


class SyntheticWorker(PaneWorker):
    """Paced synthetic scene — full UI demo with no hardware."""

    def __init__(
        self,
        pane: PaneState,
        stop: threading.Event,
        scene: SyntheticScene | None = None,
    ) -> None:
        super().__init__(pane, stop)
        self.scene = scene or SyntheticScene()

    def run(self) -> None:
        cfg = self.pane.config
        self.pane.set_status("synthetic")
        t_s = 0.0
        while not self.stop_event.is_set():
            begin = time.monotonic()
            xs, ys = self.scene.window_events(t_s, cfg.window_s)
            self._consume(xs, ys)
            t_s += cfg.window_s
            leftover = cfg.window_s - (time.monotonic() - begin)
            if leftover > 0:
                time.sleep(leftover)
