"""Evidence: views of a track, requested by a stage, computed once.

A classifier or decision stage names the views it needs in its manifest
(`evidence=("patch_ring",)`). The chain builds each provider once per stream
and shares its output with every stage that asked, so patches and embeddings
no longer ride on the detection object. Providers are opt-in helpers, not
part of the method contract: a method that wants its own view builds it.

A provider's `compute` returns, for the tracks given, the view's value keyed
by track id. A provider holding state per track (a ring, a series) prunes it
against `Tracks.live_ids` and resets with the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar, Protocol, TypeVar

from evio.dtc.errors import ConfigError
from evio.dtc.protocols import Context
from evio.dtc.types import Step, Tracks


class EvidenceProvider(Protocol):
    name: ClassVar[str]

    def __init__(self, context: Context) -> None: ...

    def compute(self, step: Step, tracks: Tracks) -> Mapping[str, object]: ...

    def reset(self) -> None: ...


T = TypeVar("T", bound=type)

PROVIDERS: dict[str, type] = {}


def evidence_provider(name: str) -> Callable[[T], T]:
    """Register a provider class under the view name stages ask for."""

    def decorator(cls: T) -> T:
        if name in PROVIDERS:
            msg = f"evidence provider {name!r} already registered"
            raise ConfigError(msg)
        cls.name = name
        PROVIDERS[name] = cls
        return cls

    return decorator
