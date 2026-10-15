"""Source facts: what a source advertises, what a method may require.

A fact is a named, optional value (`extent`, `deg_per_px`, ...); plugins may
add new kinds by using a new name. A method declares the names it needs and
the builder checks that the source's facts cover them (feature detection, not
inheritance). A method that declares nothing runs anywhere.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

Facts = Mapping[str, Any]

# (width, height) of the coordinate space the windows are in; a Window's own
# extent, when set, equals it.
EXTENT = "extent"
DEG_PER_PX = "deg_per_px"
# Interior x positions of vertical seams in that space, as a tuple of ints.
SEAMS = "seams"
TIME_SYNC = "time_sync"
POLARITY = "polarity"


def missing_facts(required: Collection[str], facts: Facts) -> tuple[str, ...]:
    """The required fact names the source does not advertise."""
    return tuple(name for name in required if name not in facts)
