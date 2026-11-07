"""docs/driver-setup.md must agree with the Makefile and be linked from README.

The guide is the only place the driver build steps are written down, so this
test fails, naming the stale entry, when a target, clone directory, repository
URL or linked output it states no longer matches the Makefile.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.t2

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "driver-setup.md"
_TARGETS = ("drivers", "build-externals", "drivers-check", "drivers-prereqs")
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


def _steps() -> str:
    return _section("Steps")


def test_prerequisites_check_is_a_step_before_make_drivers() -> None:
    steps = _steps()
    check = steps.find("make drivers-prereqs")
    build = steps.find("make drivers\n")
    assert check != -1, "Steps does not list `make drivers-prereqs`"
    assert build != -1, "Steps does not list `make drivers`"
    assert check < build, (
        "`make drivers-prereqs` must come before `make drivers`"
    )
    assert "drivers-prereqs" in _makefile_targets()


def _make_help() -> str:
    make = shutil.which("make")
    assert make, "make is not installed"
    result = subprocess.run(
        [make, "-s", "help"], cwd=_ROOT, capture_output=True, text=True,
        check=True,
    )
    return result.stdout


def test_make_help_lists_drivers_prereqs() -> None:
    lines = [ln.strip() for ln in _make_help().splitlines()]
    assert any(ln.startswith("drivers-prereqs ") for ln in lines), (
        "make help does not list drivers-prereqs"
    )


# Prerequisite table row -> the command names, any one of which satisfies it.
_CLI_ROWS = {
    "git": ("git",),
    "uv": ("uv",),
    "make": ("make",),
    "C++ compiler": ("c++", "g++", "clang++"),
    "cmake": ("cmake",),
}
# Rows that name no command-line tool, so `command -v` cannot check them.
_NON_CLI_ROWS = {"Python", "libusb (with headers)"}


def _prerequisite_rows() -> list[str]:
    section = _section("Prerequisites")
    rows = [
        ln.split("|")[1].strip()
        for ln in section.splitlines()
        if ln.startswith("|")
    ]
    return [r for r in rows if r != "Prerequisite" and not set(r) <= {"-", " "}]


def _prereqs(path_tools: list[str], tmp_path: Path) -> tuple[int, str]:
    for name in path_tools:
        stub = tmp_path / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    make = shutil.which("make")
    assert make, "make is not installed"
    result = subprocess.run(
        [make, "-s", "drivers-prereqs"],
        cwd=_ROOT,
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    # make's own "make: *** [...] Error 1" line is not the target's output.
    stderr = "\n".join(
        ln for ln in result.stderr.splitlines() if not ln.startswith("make:")
    )
    return result.returncode, stderr


_ALL_TOOLS = [t for names in _CLI_ROWS.values() for t in names]


def test_every_prerequisite_row_is_classified() -> None:
    rows = _prerequisite_rows()
    unknown = [r for r in rows if r not in _CLI_ROWS and r not in _NON_CLI_ROWS]
    assert not unknown, (
        f"prerequisite rows with no check or exemption in this test: {unknown}"
    )


def test_drivers_prereqs_passes_with_every_tool_present(tmp_path: Path) -> None:
    code, stderr = _prereqs(_ALL_TOOLS, tmp_path)
    assert code == 0, stderr


@pytest.mark.parametrize("row", list(_CLI_ROWS))
def test_drivers_prereqs_names_the_missing_tool_and_only_it(
    row: str, tmp_path: Path
) -> None:
    missing = _CLI_ROWS[row]
    present = [t for t in _ALL_TOOLS if t not in missing]
    code, stderr = _prereqs(present, tmp_path)
    assert code != 0, f"passed with {row} missing"
    assert missing[0] in stderr, (
        f"stderr does not name {missing[0]}: {stderr!r}"
    )
    for tool in present:
        named = re.search(rf"(?<![\w+-]){re.escape(tool)}(?![\w+-])", stderr)
        assert not named, f"stderr names present tool {tool}: {stderr!r}"


def test_drivers_prereqs_names_every_missing_tool(tmp_path: Path) -> None:
    code, stderr = _prereqs([], tmp_path)
    assert code != 0
    for row, names in _CLI_ROWS.items():
        assert names[0] in stderr, f"{row} not named: {stderr!r}"


def test_one_c_plus_plus_compiler_is_enough(tmp_path: Path) -> None:
    others = [t for t in _ALL_TOOLS if t not in ("c++", "clang++")]
    code, stderr = _prereqs(others, tmp_path)
    assert code == 0, stderr


def test_the_readme_links_to_the_driver_setup_guide() -> None:
    readme = (_ROOT / "README.md").read_text()
    assert "](docs/driver-setup.md)" in readme, (
        "README.md does not link to docs/driver-setup.md"
    )
