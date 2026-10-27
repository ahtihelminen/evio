"""A method's manifest: what it implements, needs and costs.

The builder checks it before any data flows, so a mismatch fails at build,
not at window 10,000. It also drives discovery and selective install.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Collection
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from evio.dtc.errors import ResourceError

BOX = "box"
POINT = "point"


class Stage(IntEnum):
    """Stage kinds in chain order; a chain implements a prefix of them."""

    DETECT = 0
    TRACK = 1
    CLASSIFY = 2
    DECIDE = 3


class Compute(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


class Cost(StrEnum):
    """Rough per-window cost class, for scheduling and reports."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class Scope(StrEnum):
    """STREAM: one instance per stream. JOINT: one instance, all streams."""

    STREAM = "stream"
    JOINT = "joint"


@dataclass(frozen=True, slots=True)
class Manifest:
    name: str
    version: str
    stages: tuple[Stage, ...]
    summary: str = ""
    scope: Scope = Scope.STREAM
    requires: tuple[str, ...] = ()
    wants: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    compute: Compute = Compute.CPU
    cost: Cost = Cost.LIGHT
    cadence: int = 1
    evidence: tuple[str, ...] = ()
    # stage kinds whose output of the last step the method reads, from
    # `Step.feedback`; any kind, the method's own included
    feedback: tuple[Stage, ...] = ()
    geometry: tuple[str, ...] = (BOX,)

    def __post_init__(self) -> None:
        if not self.stages:
            msg = f"method {self.name!r} implements no stage"
            raise ValueError(msg)
        wanted = tuple(range(self.stages[0], self.stages[0] + len(self.stages)))
        if tuple(self.stages) != wanted:
            msg = f"method {self.name!r} stages must be contiguous"
            raise ValueError(msg)
        if len(self.stages) > 1 and self.stages[0] != Stage.DETECT:
            msg = f"fused method {self.name!r} must start at DETECT"
            raise ValueError(msg)
        if self.cadence < 1:
            msg = f"method {self.name!r} cadence must be >= 1"
            raise ValueError(msg)
        if not set(self.feedback) <= set(Stage):
            msg = f"method {self.name!r} feedback must name stage kinds"
            raise ValueError(msg)


def uncovered_geometry(
    needs: Collection[str], manifest: Manifest
) -> tuple[str, ...]:
    """The region kinds a scorer needs that the method does not emit."""
    return tuple(kind for kind in needs if kind not in manifest.geometry)


def check_dependencies(manifest: Manifest) -> None:
    """Raise `ResourceError` unless every declared module is importable.

    Uses `find_spec`, so nothing heavy is imported to answer.
    """
    missing = [
        name
        for name in manifest.dependencies
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        msg = f"method {manifest.name!r} is missing modules: {missing}"
        raise ResourceError(msg)
