"""docs/driver-setup.md must agree with the Makefile and be linked from README.

The guide is the only place the driver build steps are written down, so this
test fails, naming the stale entry, when a target, clone directory, repository
URL or linked output it states no longer matches the Makefile.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "driver-setup.md"
_TARGETS = ("drivers", "build-externals", "drivers-check")
# section heading -> (Makefile dir variable, build target, driver module)
_DRIVERS = {
    "IMX636": ("IMX636_DIR", "build-imx636", "imx636"),
    "U3V": ("U3V_DIR", "build-u3v", "u3v"),
}


def _doc() -> str:
    assert _DOC.is_file(), "docs/driver-setup.md does not exist"
    return _DOC.read_text()


def _makefile() -> str:
    return (_ROOT / "Makefile").read_text()


def _section(name: str) -> str:
    match = re.search(
        rf"^## {re.escape(name)}\n(?P<body>.*?)(?=^## |\Z)",
        _doc(),
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"docs/driver-setup.md has no '## {name}' section"
    return match.group("body")


def _makefile_targets() -> set[str]:
    pattern = r"^([A-Za-z0-9_-]+):(?!=)"
    return set(re.findall(pattern, _makefile(), re.MULTILINE))


def _default_dir(variable: str) -> str:
    match = re.search(
        rf"^{variable}\s*\?=\s*\$\(EVIO_ROOT\)/(?P<dir>\S+)$",
        _makefile(),
        re.MULTILINE,
    )
    assert match, f"Makefile has no default for {variable}"
    return match.group("dir")


def _clone_url(module: str) -> str:
    match = re.search(
        rf"clone (?P<url>\S+/{module}\.git)", _makefile(), re.IGNORECASE
    )
    assert match, f"Makefile names no clone URL for {module}"
    return match.group("url")


def test_every_make_target_the_document_names_exists() -> None:
    doc = _doc()
    targets = _makefile_targets()
    for target in _TARGETS:
        assert f"make {target}" in doc, (
            f"driver-setup.md does not name `make {target}`"
        )
        assert target in targets, (
            f"driver-setup.md names make target '{target}' "
            "but the Makefile has none"
        )


@pytest.mark.parametrize("name", list(_DRIVERS))
def test_the_driver_section_names_its_makefile_facts(name: str) -> None:
    variable, build_target, module = _DRIVERS[name]
    section = _section(name)
    expected = {
        f"repository URL for {name}": _clone_url(module),
        f"clone directory for {name} ({variable} default)": _default_dir(
            variable
        ),
        f"build command for {name}": f"make {build_target}",
        f"linked output for {name}": f"{module}.pth",
    }
    stale = [what for what, text in expected.items() if text not in section]
    assert not stale, f"driver-setup.md is stale or missing: {stale}"
    assert build_target in _makefile_targets(), (
        f"driver-setup.md names make target '{build_target}' "
        "but the Makefile has none"
    )


def test_unknown_prerequisite_versions_are_marked_for_walkthrough() -> None:
    doc = _doc()
    assert "to be confirmed at the walkthrough" in doc, (
        "driver-setup.md must mark undeterminable prerequisite versions "
        "'to be confirmed at the walkthrough'"
    )
    for prerequisite in ("compiler", "cmake", "libusb"):
        line = next(
            (
                row
                for row in doc.splitlines()
                if prerequisite in row.lower() and row.startswith("|")
            ),
            None,
        )
        assert line, f"driver-setup.md lists no prerequisite '{prerequisite}'"
        assert "to be confirmed at the walkthrough" in line, (
            f"prerequisite '{prerequisite}' has no version and is not "
            "marked 'to be confirmed at the walkthrough'"
        )


def test_the_readme_links_to_the_driver_setup_guide() -> None:
    readme = (_ROOT / "README.md").read_text()
    assert "](docs/driver-setup.md)" in readme, (
        "README.md does not link to docs/driver-setup.md"
    )
