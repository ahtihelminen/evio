"""Ensures sibling test-support modules (e.g. `_cdp_client`) import cleanly
regardless of pytest's rootdir/import-mode resolution.

Also holds the `camera_process` fixture shared by
`test_camera_disconnect_state.py` and `test_run_ops_camera_state_api.py`.
It replaces the `_camera_process()` helper that used to be duplicated in
both files. That helper freed its `SharedMemory` in a `finally` that ran
on the `return` path, leaving the returned `CameraProcess.frame_view`
backed by freed memory (SN-405). This fixture keeps every segment it
creates alive until the test itself ends, then frees them all --
satisfying both "usable for the life of the test" and "no segment
survives the session".
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pytest

from evio.multiprocessing.multicamera import CameraProcess

sys.path.insert(0, str(Path(__file__).parent))

from support.hygiene import Snapshot, leaks

_AT_START: list[Snapshot] = []


class FakeProcess:
    """Stand-in for `multiprocessing.Process`: only `is_alive()` matters."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


@pytest.fixture
def camera_process() -> Callable[..., CameraProcess]:
    segments: list[SharedMemory] = []
    queues: list[Queue] = []

    def make(
        serial: str, *, alive: bool, batch_queue: Queue | None = None,
    ) -> CameraProcess:
        shm = SharedMemory(create=True, size=64)
        segments.append(shm)
        view = np.ndarray((4,), np.uint8, buffer=shm.buf)
        control_queue, information_queue = Queue(), Queue()
        queues.extend((control_queue, information_queue))
        return CameraProcess(
            serial=serial,
            shared_memory=shm,
            frame_view=view,
            process=FakeProcess(alive),
            control_queue=control_queue,
            information_queue=information_queue,
            batch_queue=batch_queue,
        )

    yield make

    # Each Queue owns semaphores. Left to the garbage collector they are
    # released from inside whatever finalizer happens to be running, which
    # is what "ResourceTracker called reentrantly" means -- 21 of those
    # warnings per full run, all from this fixture.
    for queue in queues:
        queue.close()
        queue.join_thread()
    for shm in segments:
        shm.close()
        shm.unlink()


def pytest_sessionstart() -> None:
    _AT_START.append(Snapshot.take())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """A green run that leaks is still lying about the system.

    Deliberately reports rather than cleans: reaping the evidence would
    remove the only clue to which test is responsible. Silent on an
    already-failing session, so a real failure is never masked by this.
    """
    if exitstatus != 0 or not _AT_START:
        return
    found = leaks(_AT_START[0], Snapshot.take())
    if found:
        session.exitstatus = 1
        print("\nsession left resources behind:")
        for item in found:
            print(f"  - {item}")
