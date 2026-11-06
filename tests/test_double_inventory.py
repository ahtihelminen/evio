"""Every double is listed with its conformance test and its differences.

`docs/test-doubles.md` is the one list of what lives under `tests/fakes/`
and `tests/support/`. A double added without a row, or a row left behind
by a deleted double, fails here, as does a module whose docstring does not
say where it differs from the real device.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_LIST = _ROOT / "docs" / "test-doubles.md"
_HEADING = "Differences from the real device:"
_ROW = re.compile(r"^\| `(?P<module>tests/\S+\.py)` \| (?P<cell>.+?) \|$")
_GAP = "GAP"


def _modules() -> set[str]:
    return {
        path.relative_to(_ROOT).as_posix()
        for folder in ("fakes", "support")
        for path in (_ROOT / "tests" / folder).glob("*.py")
        if path.name != "__init__.py"
    }


def _rows() -> dict[str, str]:
    rows = (_ROW.match(line) for line in _LIST.read_text().splitlines())
    return {m["module"]: m["cell"] for m in rows if m}


def test_the_list_names_every_double_and_nothing_else() -> None:
    listed, found = set(_rows()), _modules()
    assert listed == found, (
        f"add to {_LIST.name}: {sorted(found - listed)}; "
        f"remove from it: {sorted(listed - found)}"
    )


def test_each_double_names_a_conformance_test_that_exists_or_a_gap() -> None:
    wrong = {
        module: cell
        for module, cell in _rows().items()
        if not cell.startswith(_GAP) and not (_ROOT / cell.strip("`")).is_file()
    }
    assert not wrong, f"neither an existing test path nor {_GAP}: {wrong}"


def test_each_double_documents_its_differences_from_the_real_device() -> None:
    missing = []
    for module in sorted(_modules()):
        doc = ast.get_docstring(ast.parse((_ROOT / module).read_text())) or ""
        _, found, after = doc.partition(_HEADING)
        if not (found and after.strip()):
            missing.append(module)
    assert not missing, (
        f"add a '{_HEADING}' section, with 'none known' if it is empty, "
        f"to the docstring of: {missing}"
    )
