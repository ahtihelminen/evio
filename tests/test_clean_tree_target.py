"""`make check` must fail when the laptop run leaves files in the repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.t2

_ROOT = Path(__file__).resolve().parents[1]


def _dry_run(target: str) -> str:
    return subprocess.run(
        ["make", "-n", target],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_clean_tree_runs_the_laptop_suite_then_checks_git_status() -> None:
    plan = _dry_run("clean-tree")
    assert "pytest -q -m \"not hardware\"" in plan
    assert "git status --porcelain" in plan
    assert plan.index("pytest") < plan.index("git status --porcelain")


def test_check_runs_the_clean_tree_target() -> None:
    assert "git status --porcelain" in _dry_run("check")
