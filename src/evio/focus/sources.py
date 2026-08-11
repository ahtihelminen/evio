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
_REVIVE_BACKOFF_S = (2.0, 4.0, 8.0, 15.0)  # transient bus loss recovery
_DT8_X_MASK = 0x3FFF
_DT8_Y_SHIFT = 14
_DT8_P_SHIFT = 28
# Back-to-back opens of N>1 cameras leave some half-claimed and poll()
# throws seconds later (multicamera.py post_open_delay_s rationale).
_POST_OPEN_SETTLE_S = 0.25
_MAX_WINDOW_BACKLOG = 4  # drop-to-latest beyond 40 ms of stale windows
# Replay stuck-pixel seeding: pixels sustaining >= this rate over the
# seed pass get masked (OnlineHotMask convention — conservative).
_REPLAY_STUCK_EV_S = 1000.0

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
    dem_mask_dir: Path | None = None  # None = CWD-relative (discouraged)
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
    zoom: bool = False
    # Native-pixel ROI evidence image (what the metric scores) for the
    # zoom view — the operator judges focus by edge WIDTH, which the
    # downsampled overview cannot show.
    roi_view: NDArray[np.float32] | None = None


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
        self.zoom = False
        self.roi_view: NDArray[np.float32] | None = None

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
                zoom=self.zoom,
                roi_view=(
                    None if self.roi_view is None
                    else self.roi_view.copy()
                ),
            )

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def _set_roi(self, roi: tuple[int, int, int, int]) -> None:
        """Under self.lock: apply a new ROI; peak-hold resets because
        scores from different ROIs are not comparable."""
        if roi != self.roi:
            self.roi = roi
            self.history.reset_peak()

    def move_roi(self, dx: int, dy: int) -> None:
        with self.lock:
            self._set_roi(_clamp_roi(
                self.roi[0] + dx, self.roi[1] + dy,
                self.roi[2], self.roi[3],
            ))

    def resize_roi(self, delta: int) -> None:
        with self.lock:
            rx, ry, rw, rh = self.roi
            size = int(np.clip(rw + delta, 64, min(SENSOR_W, SENSOR_H)))
            cx, cy = rx + rw // 2, ry + rh // 2
            self._set_roi(_clamp_roi(
                cx - size // 2, cy - size // 2, size, size,
            ))

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
            self._set_roi(
                _clamp_roi(cx - rw // 2, cy - rh // 2, rw, rh),
            )

    def reset_peak(self) -> None:
        with self.lock:
            self.history.reset_peak()

    def toggle_zoom(self) -> None:
        with self.lock:
            self.zoom = not self.zoom
            if not self.zoom:
                self.roi_view = None


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
        self._zoom_skip = 0
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
        # One corrupt/out-of-range coordinate (replay corruption, driver
        # skew) must not kill the worker via the bincount reshape.
        valid = (
            (x_coords >= 0) & (x_coords < SENSOR_W)
            & (y_coords >= 0) & (y_coords < SENSOR_H)
        )
        if not valid.all():
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
        with self.pane.lock:
            roi = self.pane.roi
            zoom = self.pane.zoom
        result = self._scorer.score(
            x_coords, y_coords, roi=roi, stuck_mask=self.stuck_mask,
        )
        roi_view = None
        if zoom and self._scorer.acc_image is not None:
            self._zoom_skip = (self._zoom_skip + 1) % 3
            if self._zoom_skip == 0:  # ~33 Hz is plenty for the eye
                roi_view = self._scorer.acc_image.copy()
        if self.stuck_mask is not None and x_coords.size:
            # The view drives auto-ROI ('r') and display scaling — a
            # stuck pixel would otherwise be its brightest block.
            keep = ~self.stuck_mask[y_coords, x_coords]
            x_coords = x_coords[keep]
            y_coords = y_coords[keep]
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
        fresh_view = self._view_acc.copy()  # never alias the live acc
        with self.pane.lock:
            self.pane.history.push(result.hf)
            self.pane.last = result
            self.pane.view = fresh_view
            if roi_view is not None:
                self.pane.roi_view = roi_view
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
        for attempt in range(1 + len(_REVIVE_BACKOFF_S)):
            if self.stop_event.is_set():
                return
            if attempt:
                backoff = _REVIVE_BACKOFF_S[attempt - 1]
                self.pane.set_status(
                    f"reviving ({attempt}, wait {backoff:.0f}s)",
                )
                if self.stop_event.wait(backoff):
                    return
            source = Imx636CameraSource(
                serial=self.pane.serial,
                window_length_us=int(cfg.window_ms * 1000),
                dem_mask_dir=cfg.dem_mask_dir,
            )
            try:
                self._stream(source)
            except RuntimeError as exc:
                self.pane.set_status(f"USB dead: {exc}")
            except (OSError, ValueError) as exc:
                self.pane.set_status(f"worker died: {exc}")
                return  # not a USB drop — reviving would loop pointlessly
            else:
                return  # clean stop
        self.pane.set_status("dead (revive attempts exhausted)")

    def _stream(self, source: object) -> None:
        from evio.core.window import drain_window_backlog

        cfg = self.pane.config
        self.pane.set_status("opening")
        with _OPEN_LOCK:
            if self.stop_event.is_set():
                return
            source.__enter__()
            # Settle before the next camera opens — back-to-back opens
            # leave cameras half-claimed (poll() throws seconds later).
            time.sleep(_POST_OPEN_SETTLE_S)
        try:
            if self.stop_event.is_set():
                return
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
                batch, _, _ = drain_window_backlog(
                    source.window_queue, batch, _MAX_WINDOW_BACKLOG,
                )
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
        self.stuck_mask = _seed_replay_stuck_mask(words, order)
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


def _seed_replay_stuck_mask(
    words: np.ndarray, order: np.ndarray, seed_events: int = 2_000_000,
) -> NDArray[np.bool_] | None:
    """Offline hard rule: never analyze stuck pixels. One cheap pass
    over the head of the recording masks sustained >=1 kev/s pixels."""
    head = np.asarray(words[order[: min(seed_events, order.size)]],
                      dtype=np.uint32)
    if head.size == 0:
        return None
    xs = (head & _DT8_X_MASK).astype(np.intp)
    ys = ((head >> _DT8_Y_SHIFT) & _DT8_X_MASK).astype(np.intp)
    keep = (xs < SENSOR_W) & (ys < SENSOR_H)
    counts = np.bincount(
        ys[keep] * SENSOR_W + xs[keep], minlength=SENSOR_W * SENSOR_H,
    ).reshape(SENSOR_H, SENSOR_W)
    # Seed span is unknown in wall time; be conservative: only pixels
    # holding >0.1% of the head pass each (a real target never does).
    mask = counts > max(_REPLAY_STUCK_EV_S, 0.001 * head.size)
    return mask if mask.any() else None


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
