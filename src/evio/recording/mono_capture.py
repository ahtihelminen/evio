"""Background capture: mono-camera frames -> MonoFootageRecorder.

Bridges `evio.io.mono_camera.MonoCameraSource` (a blocking IDS Peak frame
grabber) and `MonoFootageRecorder`. The grab call blocks, so capture runs
on its own daemon thread, leaving the detection event loop untouched. The
appliance starts one of these per mono camera so footage records the
moment power comes up — no detection model is attached yet, we just keep
the frames.

`MonoCameraSource` imports the IDS Peak SDK, which is absent on dev
machines, so it is imported lazily inside the thread. Tests inject a
`source_factory` to drive the pump without hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from evio.recording.mono_footage_recorder import MonoFootageRecorder

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    import numpy as np

    SourceFactory = Callable[[], AbstractContextManager["FrameSource"]]

_log = logging.getLogger(__name__)


class FrameSource(Protocol):
    """The slice of `MonoCameraSource` the capture loop relies on."""

    width: int
    height: int

    def read(self, timeout_ms: int = ...) -> np.ndarray: ...


class MonoFootageCapture:
    def __init__(
        self,
        *,
        root: Path,
        camera_id: str = "mono-0",
        serial: str | None = None,
        fps: float | None = None,
        read_timeout_ms: int = 2000,
        on_frame: Callable[[], None] | None = None,
        source_factory: SourceFactory | None = None,
    ) -> None:
        self._root = Path(root)
        self._camera_id = camera_id
        self._serial = serial
        self._fps = fps
        self._read_timeout_ms = read_timeout_ms
        self._on_frame = on_frame
        self._source_factory = source_factory
        self._stop = threading.Event()
        # `_recording` gates writes only; the read loop keeps grabbing
        # frames so the IDS Peak source stays open and warm. Starts unset
        # (closed) to match RuntimeState.recording_enabled = False at boot.
        # Operator toggle (web UI /api/recording) flips this in lockstep
        # with the event-camera recorders so both stop / both resume.
        self._recording = threading.Event()
        self._thread: threading.Thread | None = None
        # Latest-frame slot (SN-491): the newest grabbed frame, kept
        # whether or not recording is on, so /mono.mjpg can show the camera
        # while nothing is being written. The sequence number lets a reader
        # tell a new frame from the one it already sent. Frames are stored
        # by reference: copying would cost a 20 MB memcpy per grab even
        # with nobody watching, and a preview can live with a torn frame.
        self._latest_lock = threading.Lock()
        self._latest: tuple[np.ndarray, int] | None = None

    def latest_frame(self) -> tuple[np.ndarray, int] | None:
        """Newest grabbed frame and its sequence number; None before the
        first grab.
        """
        with self._latest_lock:
            return self._latest

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._pump, name="mono-footage", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def is_alive(self) -> bool:
        """Whether the capture thread is still running (for soak monitoring)."""
        return self._thread is not None and self._thread.is_alive()

    def set_recording(self, enabled: bool) -> None:
        """Pause/resume writes without tearing down the camera source.

        Mirrors the event-camera workers' recording toggle: the SDK
        stream keeps running so resume is instant, but disk writes
        stop while disabled. Idempotent.
        """
        if enabled:
            self._recording.set()
        else:
            self._recording.clear()

    def _open_source(self) -> AbstractContextManager[FrameSource]:
        if self._source_factory is not None:
            return self._source_factory()
        # The IDS Peak SDK only exists on the appliance; import lazily so
        # this module stays importable (and testable) on dev machines.
        from evio.io.mono_camera import MonoCameraSource

        return MonoCameraSource(serial=self._serial, fps=self._fps)

    def _pump(self) -> None:
        # Hardware boundary on a background thread: any open/grab failure is
        # logged and ends capture rather than taking down the whole runner
        # (the event-camera mission keeps going). See the BLE001 per-file
        # ignore in pyproject.toml.
        try:
            with self._open_source() as source:
                recorder = MonoFootageRecorder(
                    root=self._root,
                    camera_id=self._camera_id,
                    width=source.width,
                    height=source.height,
                    fps=self._fps or 0.0,
                )
                self._record_loop(source, recorder)
        except Exception:
            _log.exception("mono capture stopped on error")

    def _record_loop(
        self, source: FrameSource, recorder: MonoFootageRecorder,
    ) -> None:
        try:
            while not self._stop.is_set():
                frame = source.read(timeout_ms=self._read_timeout_ms)
                with self._latest_lock:
                    seq = self._latest[1] + 1 if self._latest else 1
                    self._latest = (frame, seq)
                if self._recording.is_set():
                    recorder.write(frame, time.time_ns())
                    if self._on_frame is not None:
                        self._on_frame()
        finally:
            recorder.close()
