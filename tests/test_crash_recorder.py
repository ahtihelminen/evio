from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from support.crash import crash_recorder

pytestmark = pytest.mark.t2


def test_crash_leaves_an_unfinalised_recording_and_no_process(
    tmp_path: Path,
) -> None:
    directory = crash_recorder(tmp_path)

    assert directory.parent == tmp_path
    assert list(directory.glob("*/*.dat"))
    assert not list(directory.glob("*/manifest.jsonl"))
    survivors = subprocess.run(
        ["pgrep", "-f", str(tmp_path)], capture_output=True, text=True,
        check=False,
    )
    assert survivors.stdout == ""
