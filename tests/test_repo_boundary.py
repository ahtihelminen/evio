"""evioF must never reference evioR.

The split only holds if the dependency points one way: evioR depends on
evioF's `evio` package, never the reverse. A stray `evior` import here
would make the field appliance unbuildable on a mast.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

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


def test_no_research_module_reachable_from_evio_app() -> None:
    """Gate G3: importing the appliance must not pull evioR in transitively.

    The grep above catches a literal `evior` in this tree. It cannot catch
    `evio.app.x` importing an evioF module that itself reaches into evioR,
    so this walks the real import closure instead.
    """
    app_dir = _ROOT / "src" / "evio" / "app"
    modules = sorted(
        f"evio.app.{path.stem}" if path.stem != "__init__" else "evio.app"
        for path in app_dir.glob("*.py")
    )
    assert modules, "src/evio/app holds no modules — did the package move?"
    for name in modules:
        importlib.import_module(name)
    leaked = sorted(
        name for name in sys.modules
        if name == "evior" or name.startswith("evior.")
    )
    assert not leaked, f"evio.app pulled in evioR: {leaked}"
