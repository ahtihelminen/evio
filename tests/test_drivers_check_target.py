"""`make drivers-check` must name each driver that fails to import."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.t2

_ROOT = Path(__file__).resolve().parents[1]


def _drivers_check(
    drivers: str, stub_dir: Path
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(stub_dir)}
    return subprocess.run(
        ["make", "drivers-check", f"DRIVERS={drivers}"],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_a_missing_driver_fails_and_is_named_in_stderr(tmp_path: Path) -> None:
    (tmp_path / "stub_present_driver.py").write_text("")
    result = _drivers_check(
        "stub_present_driver stub_missing_driver", tmp_path
    )
    assert result.returncode != 0
    assert "stub_missing_driver" in result.stderr
    assert "stub_present_driver" not in result.stderr


def test_the_target_exits_zero_when_all_drivers_import(tmp_path: Path) -> None:
    (tmp_path / "stub_driver_a.py").write_text("")
    (tmp_path / "stub_driver_b.py").write_text("")
    result = _drivers_check("stub_driver_a stub_driver_b", tmp_path)
    assert result.returncode == 0, result.stderr
