"""Where a camera clock reset leaves the live part of a decoded buffer.

A stream can open with the tail of the previous session, still stamped with
the old clock, before the camera restarts its clock near zero (ops001,
2026-10-19: 5,994 events at 3.76 s, then 1.05 million events from 16 us).
Everything before the restart belongs to the old clock.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.t1

WINDOW_US = 10_000


def _reset_index(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: list[int],
    previous_last: int | None = None,
) -> int | None:
    monkeypatch.setitem(
        sys.modules, "imx636",
        sys.modules.get("imx636", types.ModuleType("imx636")),
    )
    from evio.io.imx636_camera import clock_reset_index
    return clock_reset_index(
        np.array(timestamps, np.int64), previous_last, WINDOW_US,
    )


def test_a_reset_inside_a_buffer_keeps_only_events_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = [3_764_000, 3_764_002, 3_786_226, 3_786_227]

    assert _reset_index(monkeypatch, [*stale, 16, 24, 26, 27]) == len(stale)


def test_a_buffer_that_opens_before_the_last_event_seen_is_a_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _reset_index(monkeypatch, [16, 24, 26], previous_last=3_786_227) == 0


def test_the_last_of_several_resets_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _reset_index(monkeypatch, [900_000, 5, 6, 700_000, 3, 4]) == 4


def test_ordered_events_have_no_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _reset_index(monkeypatch, [10, 11, 11, 500, 20_000]) is None
    assert _reset_index(monkeypatch, [20_001, 20_002], 20_000) is None


def test_a_step_back_within_one_window_is_not_a_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _reset_index(monkeypatch, [50_000, 49_000, 50_500]) is None
