"""The --tier-manifest gate reports a tier with no floor as a usage error.

Runs a child pytest over a copy of this suite's conftest whose manifest,
under tmp_path, has no entry for the requested tier.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.t2

_TESTS = Path(__file__).parent


def test_a_tier_missing_from_the_manifest_is_a_readable_failure(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "tests"
    shutil.copytree(_TESTS / "support", suite / "support")
    shutil.copy(_TESTS / "conftest.py", suite)
    (suite / "fixtures").mkdir()
    (suite / "fixtures" / "tier_manifest.json").write_text(
        json.dumps({"t1": 0}),
    )
    (suite / "test_one.py").write_text("def test_one():\n    pass\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
         "--tier-manifest=t2", str(suite)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )

    output = run.stdout + run.stderr
    assert run.returncode != 0
    assert "INTERNALERROR" not in output
    assert "t2" in output
    assert "tier_manifest.json" in output
