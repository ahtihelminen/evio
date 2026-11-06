"""Every top-level package under src/evio has a note in docs/modules.

A package added without documenting it is caught here, before merge. Only
the note's existence is checked, not its content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES = _ROOT / "src" / "evio"
_NOTES = _ROOT / "docs" / "modules"

# Package name -> note file, where the note is not simply `<name>.md`.
# `recording` is spelled out because docs/modules also holds
# `labelled-recording.md`, the format spec; the module note is `recording.md`.
_NOTE_FOR = {"recording": "recording.md"}


def _packages() -> list[str]:
    return sorted(
        path.name
        for path in _PACKAGES.iterdir()
        if (path / "__init__.py").is_file()
    )


def test_every_package_has_a_module_note() -> None:
    packages = _packages()
    assert packages, f"no packages found under {_PACKAGES}"
    missing = [
        f"{name} -> docs/modules/{_NOTE_FOR.get(name, name + '.md')}"
        for name in packages
        if not (_NOTES / _NOTE_FOR.get(name, f"{name}.md")).is_file()
    ]
    assert not missing, f"packages without a module note: {missing}"
