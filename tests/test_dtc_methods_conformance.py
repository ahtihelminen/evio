"""Every shipped method and gate fixture passes the conformance kit."""

from __future__ import annotations

import importlib

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.facts import EXTENT
from evio.dtc.registry import METHODS

pytestmark = pytest.mark.t1

importlib.import_module("evio.dtc.methods")
for fixture in ("point", "voxel_cnn", "joint", "frame"):
    importlib.import_module(f"fakes.dtc_{fixture}")

# These need a checkpoint to build; their own test modules cover them.
NEEDS_CONFIG = {"transformer"}


@pytest.mark.parametrize("name", sorted(set(METHODS) - NEEDS_CONFIG))
def test_registered_method_is_conformant(name: str) -> None:
    assert check_method(name, facts={EXTENT: (128, 96)}) == []
