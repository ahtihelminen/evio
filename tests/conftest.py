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

import importlib
import json
import os
import sys
from collections.abc import Callable, Iterator
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np
import pytest

from evio_ops.multiprocessing.multicamera import CameraProcess

sys.path.insert(0, str(Path(__file__).parent))

from support.hygiene import Snapshot, leaks

_AT_START: list[Snapshot] = []


class FakeProcess:
    """Stand-in for `multiprocessing.Process`: only `is_alive()` matters."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


_MANIFEST = Path(__file__).parent / "fixtures" / "tier_manifest.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tier-manifest", default=None,
        help="assert this tier executed at least its committed test count",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item],
) -> None:
    """A tier that quietly stops running tests must go red.

    Counts collected items, not passes: a skip is collected too, which is
    exactly how 62 browser tests disappear on a host with no Chrome while
    the run still reports green.
    """
    _require_browser_if_asked(items)
    requested = config.getoption("--tier-manifest")
    if not requested:
        return
    floors = json.loads(_MANIFEST.read_text())
    short = [
        f"{tier}: {found} < {floors[tier]}"
        for tier in requested.split(",")
        if (found := sum(1 for item in items if tier in item.keywords))
        < floors[tier]
    ]
    if short:
        session.exitstatus = 1
        message = (
            f"tiers below their committed floor in {_MANIFEST.name}: "
            + ", ".join(short)
        )
        raise pytest.UsageError(message)


def _require_browser_if_asked(items: list[pytest.Item]) -> None:
    """EVIO_REQUIRE_BROWSER=1 turns a missing Chrome into a failure.

    Without it the 62 browser tests skip in silence, so a host with no
    Chrome -- which is every OPS unit -- reports green with the whole
    renderer unverified.
    """
    if os.environ.get("EVIO_REQUIRE_BROWSER") != "1":
        return
    if not any("browser" in item.keywords for item in items):
        return
    from _cdp_client import find_chrome

    if find_chrome() is None:
        message = (
            "EVIO_REQUIRE_BROWSER=1 but no Chrome binary was found; the "
            "browser tier would have skipped in silence"
        )
        raise pytest.UsageError(message)


# What T1 means, enforced rather than promised. Tier membership was
# originally derived by grepping each file, and a grep is not transitive
# through imports: four files that launch a real Chrome through a helper
# read as pure. A test that reaches for one of these has to say so by
# being T2 -- the tier is a claim about behaviour, not about spelling.
_FORBIDDEN_IN_T1 = (
    ("subprocess", "Popen"),
    ("multiprocessing", "Process"),
    ("time", "sleep"),
)


@pytest.fixture(autouse=True)
def _t1_stays_pure(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "t1" not in request.keywords:
        return
    for module_name, attribute in _FORBIDDEN_IN_T1:
        module = importlib.import_module(module_name)

        def refuse(*_args: object, _seen: str = f"{module_name}.{attribute}",
                   **_kwargs: object) -> None:
            message = (
                f"a t1 test called {_seen}; move it to t2 or stop "
                f"reaching outside the process"
            )
            raise AssertionError(message)

        monkeypatch.setattr(module, attribute, refuse)


@pytest.fixture(autouse=True)
def _no_opencv_pool_outlives_its_test() -> Iterator[None]:
    """A live OpenCV thread pool must not leak into the next test's fork.

    Any `parallel_for` (e.g. the `cv2.resize` in `MediaManager`'s clip
    thread) starts OpenCV's process-global pthreads pool. A worker forked
    after that inherits the pool's bookkeeping but not its threads, and
    `camera_worker`'s first call, `cv2.setNumThreads(1)`, then waits on
    them forever (SN-566). Shrinking to one thread joins the real pool
    here; it is rebuilt lazily at the next `parallel_for`.
    """
    yield
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setNumThreads(threads)


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
