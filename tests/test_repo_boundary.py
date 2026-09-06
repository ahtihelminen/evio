"""evioF must never reference evioR.

The split only holds if the dependency points one way: evioR depends on
evioF's `evio` package, never the reverse. A stray `evior` import here
would make the field appliance unbuildable on a mast.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TREES = ("src", "scripts", "tests")


def test_no_reference_to_evior() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}:{n}"
        for tree in _TREES
        for path in (_ROOT / tree).rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "evior" in line and path.name != "test_repo_boundary.py"
    ]
    assert not offenders, f"evioF references evioR: {offenders}"
