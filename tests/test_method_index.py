"""The detection method index and the code must agree.

`docs/method-index.md` has one table row per method: `| name | kind | where |
description |`. Kind `dtc` is a method registered in `evio.dtc.registry`;
kind `detection` is a method module under `src/evio/detection`.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import evio.dtc.methods  # noqa: F401  (registers every shipped method)
from evio.dtc.registry import METHODS

pytestmark = pytest.mark.t1

ROOT =Path(__file__).resolve().parent.parent
INDEX = ROOT / "docs" / "method-index.md"
DETECTION_DIR = ROOT / "src" / "evio" / "detection"

# Modules in src/evio/detection that are plumbing, not detection methods.
NOT_METHODS = {
    "__init__.py",
    "emitter.py",
    "feature_functions.py",
    "grid.py",
    "types.py",
}

def shipped_methods() -> dict[str, type]:
    """Registered methods defined under src/evio/dtc/methods.

    Other tests register throwaway methods; those are not shipped.
    """
    shipped = ROOT / "src" / "evio" / "dtc" / "methods"
    return {
        name: cls
        for name, cls in METHODS.items()
        if Path(inspect.getsourcefile(cls) or "").parent == shipped
    }


ROW = re.compile(
    r"^\|\s*`(?P<name>[^`]+)`\s*\|\s*(?P<kind>dtc|detection)\s*\|"
    r"\s*`(?P<where>[^`]+)`\s*\|\s*(?P<desc>[^|]+?)\s*\|\s*$",
    re.MULTILINE,
)


def check_index(text: str) -> list[str]:
    """Every disagreement between index `text` and the code."""
    problems: list[str] = []
    methods = shipped_methods()
    entries = [m.groupdict() for m in ROW.finditer(text)]
    seen_dtc: set[str] = set()
    seen_modules: set[str] = set()
    for entry in entries:
        name, kind, where = entry["name"], entry["kind"], entry["where"]
        if not (ROOT / where).is_file():
            problems.append(f"{name}: path {where} does not exist")
        if kind == "dtc":
            seen_dtc.add(name)
            if name not in methods:
                problems.append(f"{name}: not registered in the dtc registry")
            else:
                source = Path(inspect.getsourcefile(methods[name]) or "")
                if source != ROOT / where:
                    problems.append(f"{name}: lives in {source}, not {where}")
        else:
            seen_modules.add(where)
            if not where.startswith("src/evio/detection/"):
                problems.append(f"{name}: {where} is outside detection/")
    problems.extend(
        f"{name}: registered dtc method has no index entry"
        for name in sorted(set(methods) - seen_dtc)
    )
    modules = {
        f"src/evio/detection/{p.name}"
        for p in DETECTION_DIR.glob("*.py")
        if p.name not in NOT_METHODS
    }
    problems.extend(
        f"{where}: detection module has no index entry"
        for where in sorted(modules - seen_modules)
    )
    return problems


def test_method_index_matches_code() -> None:
    assert check_index(INDEX.read_text()) == []


def test_method_index_fails_when_an_entry_is_removed() -> None:
    lines = INDEX.read_text().splitlines()
    for token in ("`rate-excess` | dtc", "`irstd` | detection"):
        kept = [line for line in lines if token not in line]
        assert len(kept) == len(lines) - 1
        problems = check_index("\n".join(kept))
        assert any("no index entry" in p for p in problems)


def test_method_index_fails_on_bogus_entries() -> None:
    text = INDEX.read_text()
    bogus_name = "\n| `ghost` | dtc | `src/evio/dtc/methods/oracle.py` | x |"
    assert any("not registered" in p for p in check_index(text + bogus_name))
    bogus_path = "\n| `ghost` | detection | `src/evio/detection/ghost.py` | x |"
    assert any("does not exist" in p for p in check_index(text + bogus_path))


def test_readme_links_the_method_index() -> None:
    assert INDEX.is_file()
    assert "(docs/method-index.md)" in (ROOT / "README.md").read_text()
