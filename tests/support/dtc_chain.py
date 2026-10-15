"""Spec and step builders for the chain tests. Sensor time only."""

from __future__ import annotations

from evio.dtc.spec import CandidateFilter, PipelineSpec, StageSpec
from evio.dtc.types import Step
from tests.support import dtc

STEP_US = 10_000
STREAMS = ("cam0", "cam1")
CENTER = (40, 30)


def spec_of(
    *methods: str | tuple[str, dict],
    candidate_filter: CandidateFilter | None = None,
) -> PipelineSpec:
    return PipelineSpec(
        tuple(
            StageSpec(m) if isinstance(m, str) else StageSpec(*m)
            for m in methods
        ),
        candidate_filter or CandidateFilter(),
    )


def step_at(
    index: int, *present: str, streams: tuple[str, ...] = STREAMS
) -> Step:
    """A step whose `present` streams hold a blob and the rest are empty."""
    start = index * STEP_US
    windows = [
        dtc.window(
            stream,
            dtc.blob_events(start, start + STEP_US, *CENTER, seed=index)
            if stream in present
            else dtc.empty_events(),
            start,
            start + STEP_US,
        )
        for stream in streams
    ]
    return dtc.step_of(index, STEP_US, *windows)
