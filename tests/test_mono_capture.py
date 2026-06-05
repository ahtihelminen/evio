from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from evio.recording import MonoFootageCapture, iter_manifest


class _FakeSource:
    """Stand-in for MonoCameraSource: a context manager that hands out a
    frame on each `read`, so the capture loop runs with no IDS hardware.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._frame = np.full((height, width), 7, dtype=np.uint8)

    def __enter__(self) -> _FakeSource:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, timeout_ms: int = 2000) -> np.ndarray:
        time.sleep(0.005)  # mimic a blocking grab so the loop yields
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
