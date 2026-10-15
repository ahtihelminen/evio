"""Crash a recorder on demand, for the hardware tier's recovery tests.

`crash_recorder(root)` starts `recorder_child.py` in its own session, waits
until it has written its first batch, and SIGKILLs the whole process group
mid-write, so nothing it started outlives the call. What is left under the
returned directory is a recording the recorder never finalised.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

CAMERA_ID = "crashed"
_CHILD = Path(__file__).with_name("recorder_child.py")


def crash_recorder(root: Path) -> Path:
    """Kill a recorder writing under `root`; return its recording directory."""
    path = [str(_CHILD.parents[1]), os.environ.get("PYTHONPATH")]
    child = subprocess.Popen(
        [sys.executable, str(_CHILD), str(root)],
        stdout=subprocess.PIPE, text=True, start_new_session=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, path))},
    )
    with child.stdout:
        ready = child.stdout.readline()
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    assert ready == "recording\n", f"recorder exited before writing: {ready!r}"
    return root / CAMERA_ID
