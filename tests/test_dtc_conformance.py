"""The DTC conformance kit passes sound methods and catches broken ones."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import numpy as np
import pytest
from fakes.dtc_broken import TOYS, Centroid

from evio.dtc.conformance import check_method, synthetic_steps
from evio.dtc.manifest import Stage
from evio.dtc.registry import METHODS, method

pytestmark = pytest.mark.t1

SOUND = [
    "toy-centroid",
    "toy-joint",
    "toy-tracker",
    "toy-classifier",
    "toy-decision",
    "toy-fused",
    "toy-feedback-reader",
    "toy-trusts-own-feedback",
]
BROKEN = [
    ("toy-nondeterministic", "deterministic"),
    ("toy-forgets-reset", "reset_restores_fresh_state"),
    ("toy-shares-state", "no_cross_stream_leak"),
    ("toy-mutates-window", "inputs_not_mutated"),
    ("toy-crashes-on-empty", "empty_windows_handled"),
    ("toy-wrong-type", "output_well_typed"),
    ("toy-reads-clock", "reads_no_clock"),
    ("toy-needs-fact", "manifest_sane"),
    ("toy-reacts-to-unknown-cue", "unknown_cues_ignored"),
    ("toy-crashes-on-feedback", "priors_handled"),
]


@pytest.fixture
def toys() -> Iterator[None]:
    for name, (cls, stages, options) in TOYS.items():
        method(name, stages=stages, **options)(cls)
    yield
    for name in TOYS:
        del METHODS[name]


def failed_checks(name: str) -> set[str]:
    failures = check_method(name, facts={})
    return {failure.split(":")[0] for failure in failures}


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize("name", SOUND)
def test_sound_method_has_no_failures(name: str) -> None:
    assert check_method(name, facts={}) == []


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize(("name", "check"), BROKEN)
def test_broken_method_is_caught_by_its_check(name: str, check: str) -> None:
    assert check in failed_checks(name)


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize(
    ("name", "check"),
    [
        ("toy-forgets-reset", "reset_restores_fresh_state"),
        ("toy-mutates-window", "inputs_not_mutated"),
        ("toy-crashes-on-empty", "empty_windows_handled"),
        ("toy-reads-clock", "reads_no_clock"),
    ],
)
def test_single_violation_fails_only_its_check(name: str, check: str) -> None:
    assert failed_checks(name) == {check}


@pytest.mark.usefixtures("toys")
def test_facts_satisfy_a_required_fact() -> None:
    assert check_method("toy-needs-fact", facts={"deg_per_px": 0.1}) == []


@pytest.mark.usefixtures("toys")
def test_single_stream_has_nothing_to_leak_into() -> None:
    failures = check_method("toy-centroid", facts={}, streams=("only",))
    assert failures == []


def test_heavy_module_level_import_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("toy_heavy")
    module.torch = types.ModuleType("torch")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "toy_heavy", module)
    toy = type("Heavy", (Centroid,), {"__module__": "toy_heavy"})
    method("toy-heavy", stages=(Stage.DETECT,))(toy)
    try:
        assert failed_checks("toy-heavy") == {"imports_nothing_heavy"}
    finally:
        del METHODS["toy-heavy"]


def test_unknown_method_is_reported() -> None:
    assert check_method("no-such-method", facts={}) == [
        "unknown method 'no-such-method'"
    ]


def test_toys_are_not_left_registered() -> None:
    assert not set(TOYS) & set(METHODS)


def test_synthetic_steps_are_seeded_and_time_ordered() -> None:
    first = synthetic_steps(("a", "b"), 10_000, (128, 96), seed=3)
    again = synthetic_steps(("a", "b"), 10_000, (128, 96), seed=3)
    for one, two in zip(first, again, strict=True):
        for stream, window in one.windows.items():
            events = window.events
            assert np.array_equal(
                events.x_coords, two.windows[stream].events.x_coords
            )
            assert np.all(np.diff(events.timestamps) >= 0)
            assert window.t_start_us <= events.timestamps.min()


def test_synthetic_empty_steps_hold_no_events() -> None:
    steps = synthetic_steps(("a",), 10_000, (128, 96), empty=True)
    assert all(len(s.windows["a"].events.timestamps) == 0 for s in steps)
