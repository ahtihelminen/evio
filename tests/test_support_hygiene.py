"""The leak report names what leaked and stays silent when nothing did."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tests.support.hygiene import Snapshot, leaks

pytestmark = pytest.mark.t2

LEAKED_NAME = "deliberately-leaked-thread"


@contextmanager
def leaked_thread() -> Iterator[None]:
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name=LEAKED_NAME)
    thread.start()
    try:
        yield
    finally:
        release.set()
        thread.join()


def test_a_deliberately_leaked_thread_is_named_in_the_report() -> None:
    start = Snapshot.take()
    with leaked_thread():
        report = leaks(start, Snapshot.take())
    assert len(report) == 1
    assert LEAKED_NAME in report[0]
    assert "thread" in report[0]


def test_the_report_is_empty_when_nothing_leaks() -> None:
    start = Snapshot.take()
    with leaked_thread():
        pass
    assert leaks(start, Snapshot.take()) == []


def test_a_new_worker_process_is_named_in_the_report() -> None:
    start = Snapshot(pids=frozenset({1}), threads=frozenset())
    end = Snapshot(pids=frozenset({1, 2**22 + 1}), threads=frozenset())
    report = leaks(start, end)
    assert len(report) == 1
    assert "worker process" in report[0]
    assert str(2**22 + 1) in report[0]


def test_a_daemon_thread_and_runtime_pool_threads_are_not_leaks() -> None:
    release = threading.Event()
    daemon = threading.Thread(target=release.wait, daemon=True)
    pool = threading.Thread(target=release.wait, name="ThreadPoolExecutor-0_0")
    start = Snapshot.take()
    daemon.start()
    pool.start()
    try:
        assert leaks(start, Snapshot.take()) == []
    finally:
        release.set()
        daemon.join()
        pool.join()
