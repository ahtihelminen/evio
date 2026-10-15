"""A recorder that writes synthetic events until killed: `crash.py`'s child.

Prints `recording` once its first batch has landed, then never stops.
"""

from __future__ import annotations

import sys
from pathlib import Path

from support.crash import CAMERA_ID
from support.dat import events_at

from evio.recording import FootageRecorder


def main() -> None:
    recorder = FootageRecorder(
        root=Path(sys.argv[1]), camera_id=CAMERA_ID, width=1280, height=720,
    )
    start = 0
    recorder.write(events_at(list(range(start, start + 1000))))
    print("recording", flush=True)
    while True:
        start += 1000
        recorder.write(events_at(list(range(start, start + 1000))))


if __name__ == "__main__":
    main()
