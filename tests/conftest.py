"""Suite-wide hooks: the tier floors, the t1 purity guard, and the session
leak gate. Also puts this directory on `sys.path` so the sibling
test-support packages (`fakes`, `support`) import cleanly regardless of
pytest's rootdir/import-mode resolution.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from support.hygiene import Snapshot, leaks

_AT_START: list[Snapshot] = []


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
    how a whole tier can disappear on a host missing a dependency while the
    run still reports green.
    """
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


# What T1 means, enforced rather than promised. Tier membership was
# originally derived by grepping each file, and a grep is not transitive
# through imports: a file that reaches a subprocess through a helper reads
# as pure. A test that reaches for one of these has to say so by
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

    Any `parallel_for` (e.g. a `cv2.resize`) starts OpenCV's process-global
    pthreads pool. A process forked after that inherits the pool's bookkeeping
    but not its threads, and its first `cv2.setNumThreads(1)` then waits on them
    forever (SN-566). Shrinking to one thread joins the real pool here; it is
    rebuilt lazily at the next `parallel_for`.
    """
    yield
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setNumThreads(threads)


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
