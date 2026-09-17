"""The quality gate must be spelled the same way everywhere it is written.

Before this test there were four spellings — the Makefile, PROJECT.md's
Constraints, AGENTS.md and CLAUDE.md all disagreed, and none carried both
flags the gate actually needs. `--frozen` keeps `uv.lock` from being
rewritten for the host platform; `--extra dev` is where ruff and pytest
live, so without it the documented gate cannot run on a clean checkout.

The exemption matters as much as the rule: `uv sync` and `uv run` under
`install`/`sync` must NOT gain `--frozen`. PROJECT.md's 2026-09-10 Decisions
entry rejected that outright, measured — frozen there resolves the 9.5 GB
development environment onto a field unit instead of the lean 637 MB one.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = ("Makefile", "PROJECT.md", "AGENTS.md", "CLAUDE.md")
_TOOL = re.compile(r"uv run\b(?P<flags>[^\n`\"]*?)\b(?P<tool>ruff|pytest)\b")


def _gate_invocations(text: str) -> list[tuple[str, str]]:
    """Every `uv run ... ruff|pytest`, with `$(GATE)` expanded first."""
    expanded = text.replace("$(GATE)", "uv run --frozen --extra dev")
    return [
        (m.group("tool"), m.group("flags")) for m in _TOOL.finditer(expanded)
    ]


def test_every_documented_gate_carries_frozen_and_dev() -> None:
    offenders = [
        f"{name}: 'uv run{flags}{tool}' is missing {missing}"
        for name in _DOCS
        for tool, flags in _gate_invocations((_ROOT / name).read_text())
        if (missing := [
            flag for flag in ("--frozen", "--extra dev") if flag not in flags
        ])
    ]
    assert not offenders, (
        "the gate is spelled inconsistently: " + "; ".join(offenders)
    )


def test_the_makefile_gate_targets_use_the_shared_definition() -> None:
    makefile = (_ROOT / "Makefile").read_text()
    assert "GATE      := uv run --frozen --extra dev" in makefile
    for target in ("lint:", "test:", "test-hardware:"):
        body = makefile.split(f"\n{target}", 1)[1].split("\n\n", 1)[0]
        assert "$(GATE)" in body, f"{target} does not use $(GATE)"


def test_sync_and_install_are_left_unfrozen() -> None:
    """The 2026-09-10 Decisions entry rejected --frozen on sync, measured."""
    makefile = (_ROOT / "Makefile").read_text()
    sync_body = makefile.split("\nsync:", 1)[1].split("\n\n", 1)[0]
    assert "uv sync" in sync_body
    assert "--frozen" not in sync_body, (
        "adding --frozen to sync would resolve the 9.5 GB development "
        "environment onto a field unit; see PROJECT.md's Decisions log"
    )
