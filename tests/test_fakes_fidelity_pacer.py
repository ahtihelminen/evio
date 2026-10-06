"""Fidelity for the pacer doubles in `tests/fakes/pacer.py`.

`FakePacer` and `FakeStreamingPacer` stand in for `Pacer` and
`StreamingPacer` in the HUD tests. These tests fail when a fake and its
real class drift apart.
"""

from __future__ import annotations

import pytest
from fakes.pacer import FakePacer, FakeStreamingPacer

from evio.core.pacer import Pacer
from evio.core.streaming_pacer import StreamingPacer

pytestmark = pytest.mark.t1


def _public_properties(cls: type) -> set[str]:
    return {
        name for name, member in vars(cls).items()
        if isinstance(member, property) and not name.startswith("_")
    }


def test_fake_pacers_expose_every_public_property_of_the_real_classes() -> None:
    """Deliberate difference: the fakes hold each property as a plain
    settable field, where the real classes compute a read-only property
    from their pacing state."""
    for real, fake in (
        (Pacer, FakePacer()),
        (StreamingPacer, FakeStreamingPacer()),
    ):
        assert _public_properties(real) - set(dir(fake)) == set(), real
