"""The U3V hardware test's footage checks, on a recording that rotates.

`MonoFootageRecorder` starts a new file when the UTC hour changes, and a new
day directory at midnight, so a 3 s recording can leave two files.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pytest

from evio.recording.mono_footage_recorder import MonoFootageRecorder
from tests.hardware.test_u3v_mono_recording import FPS, RECORD_S, check_footage

pytestmark = pytest.mark.t1

WIDTH, HEIGHT = 4, 3


def _record(root: Path, start: float) -> None:
    clock = {"t": start}
    recorder = MonoFootageRecorder(
        root=root, camera_id="mono-0", width=WIDTH, height=HEIGHT, fps=FPS,
        time_provider=lambda: clock["t"],
    )
    for i in range(int(FPS * RECORD_S)):
        clock["t"] = start + i / FPS
        frame = np.full((HEIGHT, WIDTH), i, dtype=np.uint8)
        recorder.write(frame, int(clock["t"] * 1e9))
    recorder.close()


@pytest.mark.parametrize(
    ("start", "files"),
    [
        ((2026, 6, 5, 10, 30, 0), 1),
        ((2026, 6, 5, 10, 59, 58), 2),
        ((2026, 6, 5, 23, 59, 58), 2),
    ],
    ids=["inside-an-hour", "across-an-hour", "across-midnight"],
)
def test_footage_checks_pass_on_a_recording(
    tmp_path: Path, start: tuple[int, ...], files: int,
) -> None:
    _record(tmp_path, float(calendar.timegm((*start, 0, 0, 0))))
    assert len(list(tmp_path.glob("mono-0/*/*.mraw"))) == files
    check_footage(tmp_path, WIDTH, HEIGHT, RECORD_S)
