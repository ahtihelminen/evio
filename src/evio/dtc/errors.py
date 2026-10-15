"""Typed errors of the DTC pipeline contract.

`ConfigError` fails at build, `ResourceError` when a device or dependency is
missing, `StageFault` at runtime. Nothing here swallows an error; the fault
policy in the chain decides whether a `StageFault` is raised or surfaced.
"""

from __future__ import annotations


class DtcError(Exception):
    """Base of every error the DTC contract raises."""


class ConfigError(DtcError):
    """The pipeline spec, a method config or the source facts are invalid."""


class ResourceError(DtcError):
    """A device, checkpoint or optional dependency is missing."""


class StageFault(DtcError):
    """A stage raised while processing a step."""

    def __init__(
        self,
        stage: str,
        stream: str | None,
        step_index: int,
        cause: BaseException,
    ) -> None:
        super().__init__(
            f"stage {stage!r} stream {stream!r} step {step_index}: {cause!r}"
        )
        self.stage = stage
        self.stream = stream
        self.step_index = step_index
        self.cause = cause
