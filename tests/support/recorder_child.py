"""A recorder that writes until killed: `crash.py`'s child.

`recorder_child.py <root>` feeds synthetic events; `<root> imx636 <serial>`
and `<root> u3v <serial>` record from that real camera. Prints `recording`
once its first batch has landed, then never stops. After every write it
appends the running event or frame count and the last timestamp to
`<root>/progress.txt`.

Differences from the real device:
- without a camera argument its events are synthetic, at a fixed rate,
  where a camera's rate follows the scene.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

from support.crash import CAMERA_ID, PROGRESS
from support.dat import events_at

from evio.recording import FootageRecorder, MonoFootageRecorder

U3V_FPS = 10.0


def _synthetic(root: Path) -> Iterator[tuple[int, int]]:
    recorder = FootageRecorder(
        root=root, camera_id=CAMERA_ID, width=1280, height=720,
    )
    start = 0
    while True:
        recorder.write(events_at(list(range(start, start + 1000))))
        start += 1000
        yield 1000, start - 1


def _imx636(root: Path, serial: str) -> Iterator[tuple[int, int]]:
    from evio.io.imx636_camera import Imx636CameraSource

    with Imx636CameraSource(serial=serial) as camera:
        recorder = FootageRecorder(
            root=root, camera_id=CAMERA_ID,
            width=camera.width, height=camera.height,
        )
        for batch in camera:
            if batch.timestamps.size:
                recorder.write(batch)
                yield batch.timestamps.size, int(batch.timestamps[-1])


def _u3v(root: Path, serial: str) -> Iterator[tuple[int, int]]:
    from evio.io.mono_backends.u3v import MonoCameraSource

    with MonoCameraSource(serial=serial, fps=U3V_FPS) as camera:
        recorder = MonoFootageRecorder(
            root=root, camera_id=CAMERA_ID,
            width=camera.width, height=camera.height, fps=U3V_FPS,
        )
        while True:
            frame = camera.read(timeout_ms=2000)
            ts_ns = time.time_ns()
            recorder.write(frame, ts_ns)
            yield 1, ts_ns


def main() -> None:
    root, *source = sys.argv[1:]
    sources = {"imx636": _imx636, "u3v": _u3v}
    writes = sources[source[0]](Path(root), *source[1:]) if source \
        else _synthetic(Path(root))
    total = 0
    with (Path(root) / PROGRESS).open("w") as progress:
        for count, last_ts in writes:
            total += count
            progress.write(f"{total} {last_ts}\n")
            progress.flush()
            if total == count:
                print("recording", flush=True)


if __name__ == "__main__":
    main()
