"""Crash a recorder on demand, for the hardware tier's recovery tests.

`crash_recorder(root)` starts `recorder_child.py` in its own session, waits
until it has written its first batch, and SIGKILLs the whole process group
mid-write, so nothing it started outlives the call. What is left under the
returned directory is a recording the recorder never finalised.

`crash_recorder(root, "imx636", serial)` or `(root, "u3v", serial)` records
from that real camera instead, for `record_s` seconds before the kill. After
every write the child appends `<count> <last timestamp>` to `PROGRESS`, so
`written(root)` tells what the recorder had been handed when it died.

Differences from the real device:
- the default child feeds synthetic events, not a camera;
- SIGKILL of a process group is a crash of the recorder, not a power loss:
  the kernel still flushes what the child had written to the page cache.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

CAMERA_ID = "crashed"
PROGRESS = "progress.txt"
_CHILD = Path(__file__).with_name("recorder_child.py")


def crash_recorder(
    root: Path, *source: str, record_s: float = 0.0, timeout_s: float = 60.0,
) -> Path:
    """Kill a recorder writing under `root`; return its recording directory."""
    path = [str(_CHILD.parents[1]), os.environ.get("PYTHONPATH")]
    child = subprocess.Popen(
        [sys.executable, str(_CHILD), str(root), *source],
        stdout=subprocess.PIPE, text=True, start_new_session=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, path))},
    )
    with child.stdout:
        try:
            ready = ""
            if select.select([child.stdout], [], [], timeout_s)[0]:
                ready = child.stdout.readline()
            if ready == "recording\n":
                time.sleep(record_s)
        finally:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    assert ready == "recording\n", f"recorder exited before writing: {ready!r}"
    return root / CAMERA_ID


def written(root: Path) -> tuple[int, int]:
    """The event or frame count and last timestamp the child last reported."""
    lines = (root / PROGRESS).read_text().splitlines(keepends=True)
    last = next(line for line in reversed(lines) if line.endswith("\n"))
    count, last_ts = last.split()
    return int(count), int(last_ts)


def survivors(root: Path) -> str:
    """Processes whose command line still names `root`."""
    return subprocess.run(
        ["pgrep", "-f", str(root)], capture_output=True, text=True,
        check=False,
    ).stdout
