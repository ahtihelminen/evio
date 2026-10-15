"""Every package under `src/evio` has a coverage floor, and the floor holds."""

from __future__ import annotations

import json
from pathlib import Path

import coverage
import pytest
from support.coverage_floor import (
    below_floor,
    manifest_mismatches,
    measured,
    packages,
    shortfalls,
)

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


def test_coverage_floor_check_fails_below_floor_and_passes_at_it() -> None:
    floors = {"_why": "synthetic", "codec": 80, "core": 50}

    assert below_floor({"codec": 79.9, "core": 50.0}, floors) == [
        "codec: 79.9% < 80%",
    ]
    assert below_floor({"codec": 80.0, "core": 50.0}, floors) == []
    assert below_floor({"codec": 95.5, "core": 100.0}, floors) == []


def test_package_with_no_measured_data_reads_as_zero_and_below_floor(
    tmp_path: Path,
) -> None:
    for name in ("covered", "untouched"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "mod.py").write_text("x = 1\n")
    manifest = tmp_path / "floors.json"
    manifest.write_text(json.dumps({"covered": 50, "untouched": 40}))
    cov = coverage.Coverage(data_file=None)
    cov.get_data().add_lines({str(tmp_path / "covered" / "mod.py"): [1]})

    assert measured(cov, tmp_path, {"untouched"}) == {"untouched": 0.0}
    assert shortfalls(cov, tmp_path, manifest) == ["untouched: 0.0% < 40%"]
