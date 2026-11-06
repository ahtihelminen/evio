"""The README's "Test tiers" section must agree with the Makefile.

The quick command for the edit loop and the broader one for before merge are
only useful if a reader can find them, so the README names each one and this
test fails when a named command is no longer a Makefile target.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_COMMANDS = ("make fast", "make check", "make test-hardware")


def _tier_section() -> str:
    readme = (_ROOT / "README.md").read_text()
    match = re.search(
        r"^## Test tiers\n(?P<body>.*?)(?=^## |\Z)",
        readme,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "README.md has no '## Test tiers' section"
    return match.group("body")


def _makefile_targets() -> set[str]:
    makefile = (_ROOT / "Makefile").read_text()
    return set(re.findall(r"^([A-Za-z0-9_-]+):(?!=)", makefile, re.MULTILINE))


def test_the_test_tiers_section_names_each_command() -> None:
    section = _tier_section()
    missing = [cmd for cmd in _COMMANDS if cmd not in section]
    assert not missing, f"README 'Test tiers' does not mention: {missing}"


def test_every_command_in_the_test_tiers_section_is_a_makefile_target() -> None:
    targets = _makefile_targets()
    missing = [
        cmd.removeprefix("make ")
        for cmd in _COMMANDS
        if cmd.removeprefix("make ") not in targets
    ]
    assert not missing, f"Makefile has no target named: {missing}"
