"""Method registry and the decorator that builds a manifest.

Registration is import-time. A toy method is a config dataclass, a class
with `step` and `reset`, and one decorator:

    @dataclass(frozen=True)
    class ThresholdConfig:
        min_events: int = 5

    @method("threshold", stages=(Stage.DETECT,), config=ThresholdConfig)
    class Threshold:
        def __init__(self, config: ThresholdConfig, context: Context): ...
        def step(self, step, priors) -> Candidates: ...
        def reset(self) -> None: ...

Config is data, validated against the dataclass; unknown keys are rejected.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from evio.dtc.errors import ConfigError
from evio.dtc.manifest import Manifest, Stage
from evio.dtc.protocols import Context

T = TypeVar("T", bound=type)

METHODS: dict[str, type] = {}


@dataclasses.dataclass(frozen=True)
class NoConfig:
    """Config of a method that has none."""


def parse_config[C](config_type: type[C], raw: Mapping[str, Any]) -> C:
    """Build `config_type` from data, rejecting unknown keys."""
    known = {f.name for f in dataclasses.fields(config_type)}
    unknown = set(raw) - known
    if unknown:
        msg = f"{config_type.__name__}: unknown keys {sorted(unknown)}"
        raise ConfigError(msg)
    try:
        return config_type(**raw)
    except (TypeError, ValueError) as error:
        msg = f"{config_type.__name__}: {error}"
        raise ConfigError(msg) from error


def method(
    name: str,
    *,
    stages: tuple[Stage, ...],
    config: type = NoConfig,
    version: str = "1",
    **manifest_fields: object,
) -> Callable[[T], T]:
    """Register a class as a method and attach its `manifest`."""

    def decorator(cls: T) -> T:
        if name in METHODS:
            msg = f"method {name!r} already registered"
            raise ConfigError(msg)
        cls.manifest = Manifest(
            name=name,
            version=version,
            stages=stages,
            summary=(cls.__doc__ or "").strip().split("\n")[0],
            **manifest_fields,
        )
        cls.config_type = config
        METHODS[name] = cls
        return cls

    return decorator


def create(
    name: str, raw_config: Mapping[str, Any], context: Context
) -> object:
    """Build one instance of a registered method."""
    if name not in METHODS:
        msg = f"unknown method {name!r}; known: {sorted(METHODS)}"
        raise ConfigError(msg)
    cls = METHODS[name]
    return cls(parse_config(cls.config_type, raw_config), context)


def list_methods() -> list[Manifest]:
    """Every registered method's manifest, by name."""
    return [METHODS[name].manifest for name in sorted(METHODS)]
