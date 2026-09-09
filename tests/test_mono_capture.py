from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from evio.recording import MonoFootageCapture, iter_manifest


class _FakeSource:
    """Stand-in for MonoCameraSource: a context manager that hands out a
    frame on each `read`, so the capture loop runs with no IDS hardware.
    """

    def __init__(
        self,
        width: int,
        height: int,
        on_read: Callable[[], None] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self._frame = np.full((height, width), 7, dtype=np.uint8)
        self._on_read = on_read

    def __enter__(self) -> _FakeSource:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, timeout_ms: int = 2000) -> np.ndarray:
        time.sleep(0.005)  # mimic a blocking grab so the loop yields
        if self._on_read is not None:
            self._on_read()
        return self._frame


def test_capture_writes_frames_and_finalises_on_stop(tmp_path: Path) -> None:
    saw_frame = threading.Event()
    count = {"n": 0}

    def on_frame() -> None:
        count["n"] += 1
        saw_frame.set()

    capture = MonoFootageCapture(
        root=tmp_path, camera_id="mono-0",
        on_frame=on_frame,
        source_factory=lambda: _FakeSource(4, 3),
    )
    capture.set_recording(True)  # write gate defaults closed; enable here
    capture.start()
    assert saw_frame.wait(2.0)
    capture.stop()

    assert count["n"] >= 1
    mraws = list(tmp_path.glob("mono-0/*/*.mraw"))
    assert len(mraws) == 1
    manifest = next(tmp_path.glob("mono-0/*/manifest.jsonl"))
    entries = list(iter_manifest(manifest))
    assert len(entries) == 1
    assert entries[0].capture["format"] == "mono-raw"
    assert entries[0].capture["width"] == 4
    assert entries[0].capture["height"] == 3


def test_capture_survives_source_open_failure(tmp_path: Path) -> None:
    def boom() -> _FakeSource:
        msg = "no camera"
        raise RuntimeError(msg)

    capture = MonoFootageCapture(root=tmp_path, source_factory=boom)
    capture.start()
    capture.stop()  # the failing open is logged; the runner is unaffected
    assert list(tmp_path.glob("**/*.mraw")) == []


def test_stop_without_start_is_safe(tmp_path: Path) -> None:
    capture = MonoFootageCapture(
        root=tmp_path, source_factory=lambda: _FakeSource(4, 3),
    )
    capture.stop()


def test_capture_writes_nothing_until_recording_enabled(tmp_path: Path) -> None:
    """The write gate defaults closed, matching RuntimeState.recording_enabled
    = False: a freshly constructed capture must produce no frames/files until
    set_recording(True) is called.
    """
    on_frame_count = {"n": 0}
    reading = threading.Event()

    def on_frame() -> None:
        on_frame_count["n"] += 1

    capture = MonoFootageCapture(
        root=tmp_path, camera_id="mono-0",
        on_frame=on_frame,
        source_factory=lambda: _FakeSource(4, 3, on_read=reading.set),
    )
    capture.start()
    assert reading.wait(2.0), "capture never started reading frames"
    time.sleep(0.05)
    assert on_frame_count["n"] == 0, "on_frame fired before recording enabled"
    assert list(tmp_path.glob("**/*.mraw")) == []

    capture.set_recording(True)
    deadline = time.monotonic() + 2.0
    while on_frame_count["n"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert on_frame_count["n"] > 0, "enabling recording should resume writes"
    capture.stop()
    assert list(tmp_path.glob("**/*.mraw"))


def test_read_loop_keeps_grabbing_while_gate_closed(tmp_path: Path) -> None:
    """source.read() must run every loop iteration regardless of the
    recording gate, so the IDS Peak source stays open and warm.
    """
    read_count = {"n": 0}

    def on_read() -> None:
        read_count["n"] += 1

    capture = MonoFootageCapture(
        root=tmp_path,
        source_factory=lambda: _FakeSource(4, 3, on_read=on_read),
    )
    capture.start()
    deadline = time.monotonic() + 2.0
    while read_count["n"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    first = read_count["n"]
    assert first >= 3, "source.read should run while the write gate is closed"

    while read_count["n"] < first + 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert read_count["n"] >= first + 3, "read loop stalled while recording off"
    capture.stop()


def test_set_recording_pauses_writes_but_not_grabs(tmp_path: Path) -> None:
    """set_recording(False) stops `on_frame` ticks and disk writes while
    keeping the SDK source open. Verifies unified-recording-toggle: mono
    pauses in lockstep with the event-camera workers.
    """
    count = {"n": 0}
    saw_frame = threading.Event()

    def on_frame() -> None:
        count["n"] += 1
        saw_frame.set()

    capture = MonoFootageCapture(
        root=tmp_path, camera_id="mono-0",
        on_frame=on_frame,
        source_factory=lambda: _FakeSource(4, 3),
    )
    capture.set_recording(True)  # write gate defaults closed; enable here
    capture.start()
    assert saw_frame.wait(2.0)
    capture.set_recording(False)
    time.sleep(0.05)
    pause_count = count["n"]
    time.sleep(0.05)
    assert count["n"] == pause_count, "writes leaked through pause"
    capture.set_recording(True)
    deadline = time.monotonic() + 2.0
    while count["n"] == pause_count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert count["n"] > pause_count, "resume did not re-enable writes"
    capture.stop()
