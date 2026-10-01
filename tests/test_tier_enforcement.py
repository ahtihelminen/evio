"""Every test belongs to exactly one tier, and a tier cannot vanish quietly.

Tier membership used to be a property of how a file was spelled -- derived
by grepping each one for `subprocess`, `socket`, `sleep`. A grep is not
transitive through imports, so a file that reaches a subprocess through a
helper reads as a pure unit test, and a microbenchmark can be exiled to the
slow tier for containing the string `perf_counter`.

So the tier is declared, and both halves of the declaration are checked:
this file checks that every module declares one, and conftest's
`_t1_stays_pure` checks at run time that a T1 test does not reach for a
process, a subprocess or the clock.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = _ROOT / "tests"
# `hardware` is a tier of its own, not a slow T2: it needs a camera, and
# `make test`'s `-m "not hardware"` must deselect it cleanly.
_TIERS = {"t1", "t2", "hardware"}


def _module_marks(path: Path) -> list[str]:
    """The `pytest.mark.<name>`s in a module-level `pytestmark`."""
    for node in ast.parse(path.read_text()).body:
        if not (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "pytestmark"):
            continue
        value = node.value
        entries = value.elts if isinstance(value, ast.List) else [value]
        return [
            ast.unparse(entry).removeprefix("pytest.mark.").split("(")[0]
            for entry in entries
        ]
    return []


def test_every_test_module_declares_exactly_one_tier() -> None:
    offenders = {
        path.name: sorted(set(_module_marks(path)) & _TIERS)
        for path in sorted(_TESTS.glob("test_*.py"))
        if len(set(_module_marks(path)) & _TIERS) != 1
    }
    assert not offenders, (
        "every test module needs exactly one tier marker: " + str(offenders)
    )


def test_the_manifest_covers_every_declared_tier() -> None:
    path = _TESTS / "fixtures" / "tier_manifest.json"
    manifest = json.loads(path.read_text())
    declared = {
        mark
        for path in _TESTS.glob("test_*.py")
        for mark in _module_marks(path)
        if not mark.startswith("skipif")
    }
    missing = declared - set(manifest)
    assert not missing, f"tiers with no committed floor: {sorted(missing)}"
