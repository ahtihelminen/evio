"""A new test double must come from `tests/fakes/`, where it is pinned.

SN-541: `_FakeCam.get_biases_absolute()` echoed back whatever
`set_biases()` was handed, so a driver whose write scale is offsets and
whose read scale is absolute DAC codes looked perfectly coherent. Round
trip green, two reviews PASS, and a one-step nudge moved the sensor by the
whole factory baseline.

The convention alone does not hold. Over the nine commits after that was
written down, SN-491 fixed one constant-frame double and added another in
the same commit, one file over. So it is a test, and the allowlist of
doubles that predate `tests/fakes/` may only shrink.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = _ROOT / "tests"
_ALLOWLIST = _TESTS / "fixtures" / "adhoc_fakes_allowlist.json"

sys.path.insert(0, str(_TESTS))
from support.fakescan import scan  # noqa: E402


def _allowed() -> dict[str, set[str]]:
    payload = json.loads(_ALLOWLIST.read_text())["allowed"]
    return {name: set(classes) for name, classes in payload.items()}


def test_no_new_ad_hoc_double() -> None:
    allowed, found = _allowed(), scan(_TESTS)
    new = {
        name: sorted(classes - allowed.get(name, set()))
        for name, classes in found.items()
        if classes - allowed.get(name, set())
    }
    assert not new, (
        "these doubles are defined inline in a test file; put them in "
        f"tests/fakes/ with a test pinning what they model: {new}"
    )


def test_the_allowlist_may_only_shrink() -> None:
    """An entry that no longer exists is removed, not left to rot."""
    allowed, found = _allowed(), scan(_TESTS)
    stale = {
        name: sorted(classes - found.get(name, set()))
        for name, classes in allowed.items()
        if classes - found.get(name, set())
    }
    assert not stale, (
        "the allowlist names doubles that are gone; delete these entries so "
        f"the file cannot drift from the tree: {stale}"
    )


def test_the_rule_catches_the_two_doubles_it_exists_for() -> None:
    """A relation-only rule misses both; this is why the predicate is a union.

    `_FakeCam` reaches the code under test as `self.cam = _FakeCam()` and
    `_FakeU3vCamera` as `module.Camera = _FakeU3vCamera` -- neither is a
    monkeypatch, a `sys.modules` write or a factory argument.
    """
    found = scan(_TESTS)
    assert "_FakeCam" in found["test_bias_register_readback.py"]
    assert "_FakeU3vCamera" in found["test_mono_backends_contract.py"]
