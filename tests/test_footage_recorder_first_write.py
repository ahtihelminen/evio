"""Building a recorder compiles the encoder, so its first write does not.

A live IMX636 buffers about 120 ms; the first write of a fresh process used to
compile numba's `encode_dt8_pairs` for 200 ms or more, and the camera shed
events meanwhile. Compilation happens once per process, so this runs in a
fresh one and counts the compiled signatures instead of timing the write. The
child turns the suite's JIT-off setting back on: without a JIT there is
nothing to compile.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.t2

_CHILD = """
import sys
from pathlib import Path

import numpy as np

from evio.codec.dt8 import encode_dt8_pairs
from evio.core.events import EventsBatch
from evio.recording import FootageRecorder

rng = np.random.default_rng(0)
n = 20_000
batch = EventsBatch(
    timestamps=np.sort(rng.integers(0, 10_000, n)).astype(np.int64) + 1_000_000,
    x_coords=rng.integers(0, 1280, n).astype(np.int16),
    y_coords=rng.integers(0, 720, n).astype(np.int16),
    polarities=rng.integers(0, 2, n).astype(np.int8),
)
fresh = len(encode_dt8_pairs.signatures)
recorder = FootageRecorder(
    root=Path(sys.argv[1]), camera_id="cam", width=1280, height=720,
)
built = len(encode_dt8_pairs.signatures)
recorder.write(batch)
written = len(encode_dt8_pairs.signatures)
recorder.close()
print(fresh, built, written)
"""


def test_building_a_recorder_compiles_the_encoder_not_its_first_write(
    tmp_path: Path,
) -> None:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "NUMBA_DISABLE_JIT": "0"},
    )
    assert done.returncode == 0, done.stderr
    fresh, built, written = map(int, done.stdout.split())

    assert fresh == 0
    assert built >= 1
    assert written == built
