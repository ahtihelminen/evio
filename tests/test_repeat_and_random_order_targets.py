"""The laptop suite must be provably stable: repeated and in random order."""

from __future__ import annotations

import re
import subprocess
import tomllib
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


def test_test_repeat_runs_the_laptop_suite_and_compares_outcomes() -> None:
    plan = _dry_run("test-repeat")
    assert 'pytest -q -m "not hardware"' in plan
    assert "REPEAT" in (_ROOT / "Makefile").read_text()
    assert "diff" in plan


def test_test_repeat_states_n_in_makefile_and_readme() -> None:
    makefile = (_ROOT / "Makefile").read_text()
    n = re.search(r"^REPEAT\s*\?=\s*(\d+)", makefile, re.MULTILINE)
    assert n, "Makefile does not define REPEAT ?= <N>"
    assert int(n.group(1)) >= 2
    readme = (_ROOT / "README.md").read_text()
    assert f"{n.group(1)} times" in readme


def test_test_random_order_runs_the_suite_with_two_distinct_seeds() -> None:
    plan = _dry_run("test-random-order")
    seeds = re.findall(r"--random-order-seed=(\d+)", plan)
    assert len(seeds) == 2
    assert len(set(seeds)) == 2


def test_a_random_order_plugin_is_a_dev_dependency() -> None:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(dep.startswith("pytest-random-order") for dep in dev)


def test_the_readme_names_both_commands() -> None:
    readme = (_ROOT / "README.md").read_text()
    assert "make test-repeat" in readme
    assert "make test-random-order" in readme
