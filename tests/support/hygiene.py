"""What the suite must leave behind: nothing.

A green run that leaks a worker process, a thread or a file descriptor is
still telling the truth about its assertions and lying about the system.
The suite leaked 24 worker processes across eight runs on a dev machine
before anyone looked, and nothing went red.

This is deliberately a reporter of evidence, not a cleaner: a session-end
hook that tidied up would remove exactly the thing a reader needs to find
the test responsible.

File descriptors are deliberately NOT counted. Measured on this suite, a
green run ends with 39 more open descriptors than it started, and an lsof
dump shows almost all of them are shared libraries the interpreter dlopened
(torch's dylibs, libyaml, ICU data) plus /dev/null. Counting them would
measure module loading and call it a leak.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

from evio.ops.teardown import surviving_worker_pids

# CPython's own bookkeeping, not a test's leak: the resource tracker starts
# with the first multiprocessing resource and exits with its parent, and
# ThreadPoolExecutor's idle workers are joined by concurrent.futures' atexit
# hook. Counting either would make the gate cry wolf on every green run.
_RUNTIME_PROCESS = "multiprocessing.resource_tracker"
_RUNTIME_THREADS = ("ThreadPoolExecutor-",)


def _command(pid: int) -> str:
    listed = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True, text=True, check=False,
    )
    return listed.stdout.strip()

@dataclass(frozen=True)
class Snapshot:
    """Resources held by this process and its group at one moment."""

    pids: frozenset[int]
    threads: frozenset[str]

    @classmethod
    def take(cls) -> Snapshot:
        return cls(
            pids=frozenset(
                pid for pid in surviving_worker_pids()
                if _RUNTIME_PROCESS not in _command(pid)
            ),
            threads=frozenset(
                t.name for t in threading.enumerate()
                if not t.daemon and not t.name.startswith(_RUNTIME_THREADS)
            ),
        )


def leaks(start: Snapshot, end: Snapshot) -> list[str]:
    """Human-readable descriptions of what the session failed to give back."""
    found = []
    if new_pids := sorted(end.pids - start.pids):
        described = ", ".join(
            f"{pid} ({_command(pid)[:70]})" for pid in new_pids
        )
        found.append(
            f"{len(new_pids)} worker process(es) survived: {described}"
        )
    if new_threads := sorted(end.threads - start.threads):
        found.append(f"non-daemon thread(s) still running: {new_threads}")
    return found
