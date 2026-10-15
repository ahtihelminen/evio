"""A plain `pytest -q` must never touch the hardware tier.

`make test` deselects it explicitly; the bare gate command relies on the
`-m "not hardware"` in `addopts`. This test runs in both and inspects the
session it belongs to, so it is the evidence rather than a reading of the
config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.t1

_HARDWARE = Path(__file__).parent / "hardware"


def test_the_default_run_collects_no_hardware_test(
    request: pytest.FixtureRequest,
) -> None:
    assert list(_HARDWARE.glob("test_*.py")), "no hardware test to deselect"
    collected = [
        item.nodeid
        for item in request.session.items
        if item.get_closest_marker("hardware")
    ]
    assert not collected, f"the default run collected {collected}"
