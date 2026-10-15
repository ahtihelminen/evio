"""The pipeline spec: data that names methods and their configs.

A spec is plain data so it can be hashed for provenance, stored beside
results and rebuilt. Candidate filtering (top-k by score, then IoU dedupe) is
part of the spec because the chain owns it once, not each method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from evio.dtc.errors import ConfigError


@dataclass(frozen=True, slots=True)
class StageSpec:
    method: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateFilter:
    """Keep the `top_k` best candidates, then drop boxes overlapping a better
    one by more than `iou`. None disables a step.

    Applied within one stream key, since regions of different coordinate
    spaces never compare; candidates without a box skip the IoU step."""

    top_k: int | None = 20
    iou: float | None = 0.3

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 0:
            msg = "top_k must not be negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    stages: tuple[StageSpec, ...]
    candidate_filter: CandidateFilter = CandidateFilter()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PipelineSpec:
        unknown = set(raw) - {"stages", "candidate_filter"}
        if unknown:
            msg = f"unknown pipeline spec keys: {sorted(unknown)}"
            raise ConfigError(msg)
        try:
            stages = tuple(
                StageSpec(s["method"], dict(s.get("config", {})))
                for s in raw["stages"]
            )
            filter_ = CandidateFilter(**raw.get("candidate_filter", {}))
        except (KeyError, TypeError) as error:
            msg = f"invalid pipeline spec: {error!r}"
            raise ConfigError(msg) from error
        return cls(stages, filter_)
