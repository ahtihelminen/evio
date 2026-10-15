"""Candidate filtering, owned once by the chain: top-k, then IoU dedupe.

Pure functions on `Candidate`. Filtering is per stream key, because regions
of different coordinate spaces never compare, and candidates without a box
skip the IoU step. The order is deterministic: streams in order of first
appearance, best score first within a stream (ties keep input order).
"""

from __future__ import annotations

from collections.abc import Sequence

from evio.dtc.spec import CandidateFilter
from evio.dtc.types import Box, Candidate, StreamKey


def iou(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def filter_candidates(
    items: Sequence[Candidate], rule: CandidateFilter
) -> tuple[Candidate, ...]:
    by_stream: dict[StreamKey, list[Candidate]] = {}
    for candidate in items:
        by_stream.setdefault(candidate.stream, []).append(candidate)
    kept: list[Candidate] = []
    for group in by_stream.values():
        ranked = sorted(group, key=lambda c: -c.score)[: rule.top_k]
        kept += ranked if rule.iou is None else _dedupe(ranked, rule.iou)
    return tuple(kept)


def _dedupe(ranked: list[Candidate], threshold: float) -> list[Candidate]:
    """Greedy: keep a box unless a better kept box overlaps it too much."""
    kept: list[Candidate] = []
    for candidate in ranked:
        box = candidate.region.box
        if box is None or all(
            other.region.box is None or iou(box, other.region.box) <= threshold
            for other in kept
        ):
            kept.append(candidate)
    return kept
