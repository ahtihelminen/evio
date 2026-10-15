"""The fault policy: the one place a stage's exception is caught.

Under RESEARCH the exception is re-raised as a `StageFault`. Under FIELD it
is turned into a surfaced `Fault` plus an empty, well-typed, DEGRADED output,
so the other streams and later stages keep running.
"""

from __future__ import annotations

from collections.abc import Callable

from evio.dtc.errors import StageFault
from evio.dtc.result import FaultPolicy
from evio.dtc.types import Fault, State, Status


def guarded[T](
    policy: FaultPolicy,
    stage: str,
    stream: str | None,
    step_index: int,
    call: Callable[[], T],
    fallback: Callable[[Status], T],
) -> tuple[T, Fault | None]:
    """Run `call`; on an exception, raise or return `fallback(degraded)`."""
    try:
        return call(), None
    except Exception as error:
        if policy is FaultPolicy.RESEARCH:
            raise StageFault(stage, stream, step_index, error) from error
        message = repr(error)
        degraded = Status(State.DEGRADED, detail=message)
        return fallback(degraded), Fault(stage, stream, step_index, message)
