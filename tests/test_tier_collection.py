"""Collection fails when a single test has none or more than one tier.

Runs a child pytest over a copy of this suite's conftest, so the collection
hook under test is the real one and the test files it judges are throwaways.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.t2

_TESTS = Path(__file__).parent


def _collect(tmp_path: Path, source: str) -> tuple[int, str]:
    suite = tmp_path / "tests"
    shutil.copytree(_TESTS / "support", suite / "support")
    shutil.copy(_TESTS / "conftest.py", suite)
    (suite / "test_probe.py").write_text(source)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    t1\n    t2\n    hardware\n"
    )
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
         "--collect-only", "-q", str(suite)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    return run.returncode, run.stdout + run.stderr


def test_a_test_carrying_two_tiers_fails_collection(tmp_path: Path) -> None:
    code, output = _collect(tmp_path, (
        "import pytest\n\n"
        "@pytest.mark.t1\n@pytest.mark.t2\n"
        "def test_doubly_tiered():\n    pass\n"
    ))
    assert code != 0
    assert "test_doubly_tiered" in output


def test_a_module_tier_plus_a_function_tier_fails_collection(
    tmp_path: Path,
) -> None:
    code, output = _collect(tmp_path, (
        "import pytest\n\npytestmark = pytest.mark.t1\n\n"
        "@pytest.mark.hardware\n"
        "def test_stacked():\n    pass\n"
    ))
    assert code != 0
    assert "test_stacked" in output


def test_a_test_with_no_tier_fails_collection(tmp_path: Path) -> None:
    code, output = _collect(tmp_path, "def test_untiered():\n    pass\n")
    assert code != 0
    assert "test_untiered" in output


def test_one_tier_per_test_collects_cleanly(tmp_path: Path) -> None:
    code, output = _collect(tmp_path, (
        "import pytest\n\n"
        "@pytest.mark.t1\ndef test_a():\n    pass\n\n"
        "@pytest.mark.t2\ndef test_b():\n    pass\n\n"
        "@pytest.mark.hardware\ndef test_c():\n    pass\n"
    ))
    assert code == 0, output
