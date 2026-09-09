"""Pipeline registry — name → factory for runtime swap.

The two pipelines on disk (`SimplifiedDetectionPipeline`,
`GmmOnlineBgDetectionPipeline`) are duck-typed interchangeable via the
`.update(batch).detections` surface. This module codifies that contract
as a `DetectionPipeline` Protocol and adds a name-keyed registry so the
control plane can `swap_pipeline gmm_online` without code edits.

Registration is import-time: each pipeline module imports `register`
and decorates a top-level factory. The package's `__init__.py` imports
both pipeline modules, so any code that imports
`evio.detection.pipeline` automatically populates `PIPELINES`. There
are no circular-import concerns because the registry imports nothing
from the pipeline modules — the wiring goes the other way.

A swap mid-flight resets calibration. `SimplifiedDetectionPipeline`
loses ~1 s to its 100-window warmup (the deployed default, set by this
module's `_build_simplified` factory — not `SimplifiedPipelineConfig`'s
own, unused-in-deployment 10-window dataclass default);
`GmmOnlineBgDetectionPipeline` loses its 3 s skip. The control-plane
status surface reports
`pipeline_state: "calibrating"` during that window so the dev viewer
can grey out outputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from evio.core.events import EventsBatch
from evio.detection.types import Detection


@runtime_checkable
class DetectionResult(Protocol):
    detections: list[Detection]


class DetectionPipeline(Protocol):
    def update(self, batch: EventsBatch) -> DetectionResult: ...


PipelineFactory = Callable[[dict[str, Any]], DetectionPipeline]

PIPELINES: dict[str, PipelineFactory] = {}


def register(name: str) -> Callable[[PipelineFactory], PipelineFactory]:
    def decorator(factory: PipelineFactory) -> PipelineFactory:
        if name in PIPELINES:
            msg = f"pipeline {name!r} already registered"
            raise ValueError(msg)
        PIPELINES[name] = factory
        return factory
    return decorator


def build(name: str, config: dict[str, Any] | None = None) -> DetectionPipeline:
    if name not in PIPELINES:
        known = sorted(PIPELINES)
        msg = f"unknown pipeline {name!r}; known: {known}"
        raise KeyError(msg)
    return PIPELINES[name](config or {})


def list_pipelines() -> list[str]:
    return sorted(PIPELINES)
