"""Every package under `src/evio` has a coverage floor, and the floor holds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support.coverage_floor import manifest_mismatches, packages

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "evio"
_MANIFEST = json.loads(
    (_ROOT / "tests" / "fixtures" / "coverage_manifest.json").read_text(),
)


def test_coverage_manifest_names_every_package(tmp_path: Path) -> None:
    assert manifest_mismatches(packages(_SRC), _MANIFEST) == []

    for name in [*packages(_SRC), "newpkg"]:
        (tmp_path / name).mkdir()
    assert manifest_mismatches(packages(tmp_path), _MANIFEST) == [
        "newpkg: no entry in the manifest",
    ]
    assert manifest_mismatches(packages(_SRC), {**_MANIFEST, "gone": 1}) == [
        "gone: no such package",
    ]
