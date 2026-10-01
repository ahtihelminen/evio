"""evio is the base: it never reaches into the repos built on top of it.

`ops` (the OPS-unit appliance, package `evio_ops`) and `evior` (research
tooling) both depend on evio. The dependency only runs that way, so a
laptop with nothing but evio installed can import every module here.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
from pathlib import Path

import pytest

import evio

pytestmark = pytest.mark.t1

_ROOT = Path(__file__).resolve().parents[1]
_TREES = ("src", "scripts", "tests")
_DOWNSTREAM = re.compile(r"\bevio_ops\b|\bevior\b")


def test_no_reference_to_a_downstream_repo() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}:{n}"
        for tree in _TREES
        for path in (_ROOT / tree).rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _DOWNSTREAM.search(line) and path.name != "test_repo_boundary.py"
    ]
    assert not offenders, f"evio names a downstream package: {offenders}"


def test_no_downstream_module_reachable_from_evio() -> None:
    """The grep above sees literal names; this walks the real closure.

    Modules whose optional dependency (torch, the camera SDKs) is absent
    are skipped: what matters is what the importable ones pull in.
    """
    for info in pkgutil.walk_packages(evio.__path__, "evio."):
        try:
            importlib.import_module(info.name)
        except ImportError as error:
            if (error.name or "").startswith("evio"):
                raise
    leaked = sorted(
        name for name in sys.modules
        if name.split(".")[0] in ("evio_ops", "evior")
    )
    assert not leaked, f"importing evio pulled in: {leaked}"
