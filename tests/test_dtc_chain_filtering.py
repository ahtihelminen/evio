"""Candidate filtering: top-k by score, then greedy IoU dedupe."""

from __future__ import annotations

import pytest

from evio.dtc.filtering import filter_candidates, iou
from evio.dtc.spec import CandidateFilter
from evio.dtc.types import Box, Candidate, Region

pytestmark = pytest.mark.t1


def boxed(stream: str, x0: float, score: float) -> Candidate:
    return Candidate(stream, Region((x0, 0, x0 + 10, 10)), score)


def pointed(stream: str, score: float) -> Candidate:
    return Candidate(stream, Region(point=(5.0, 5.0)), score)


def scores(items: tuple[Candidate, ...]) -> list[float]:
    return [c.score for c in items]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        ((0, 0, 10, 10), (20, 0, 30, 10), 0.0),
        ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),
        ((0, 0, 10, 10), (5, 0, 15, 10), 50 / 150),
    ],
)
def test_iou(a: Box, b: Box, expected: float) -> None:
    assert iou(a, b) == pytest.approx(expected)


def test_top_k_keeps_the_best_scores_best_first() -> None:
    items = [boxed("a", 100 * i, s) for i, s in enumerate((1, 5, 3, 4))]
    kept = filter_candidates(items, CandidateFilter(top_k=2, iou=None))
    assert scores(kept) == [5, 4]


def test_overlapping_lower_scores_are_dropped_greedily() -> None:
    items = [boxed("a", 0, 3), boxed("a", 1, 9), boxed("a", 50, 1)]
    kept = filter_candidates(items, CandidateFilter(top_k=None, iou=0.3))
    assert scores(kept) == [9, 1]


def test_top_k_applies_before_the_dedupe() -> None:
    items = [boxed("a", 0, 9), boxed("a", 1, 8), boxed("a", 50, 7)]
    kept = filter_candidates(items, CandidateFilter(top_k=2, iou=0.3))
    assert scores(kept) == [9]


def test_none_disables_each_step() -> None:
    items = [boxed("a", 0, 9), boxed("a", 0, 8)]
    assert len(filter_candidates(items, CandidateFilter(None, None))) == 2


def test_streams_are_filtered_independently() -> None:
    items = [boxed("a", 0, 9), boxed("b", 0, 8), boxed("a", 0, 7)]
    kept = filter_candidates(items, CandidateFilter(top_k=1, iou=0.3))
    assert [(c.stream, c.score) for c in kept] == [("a", 9), ("b", 8)]


def test_identical_boxes_in_different_streams_are_not_duplicates() -> None:
    items = [boxed("a", 0, 9), boxed("b", 0, 8)]
    kept = filter_candidates(items, CandidateFilter(top_k=None, iou=0.1))
    assert len(kept) == 2


def test_candidates_without_a_box_skip_the_iou_step() -> None:
    items = [pointed("a", 9), pointed("a", 8), boxed("a", 0, 7)]
    kept = filter_candidates(items, CandidateFilter(top_k=None, iou=0.0))
    assert scores(kept) == [9, 8, 7]


def test_boxless_candidates_still_count_towards_top_k() -> None:
    items = [pointed("a", 9), boxed("a", 0, 8), pointed("a", 7)]
    kept = filter_candidates(items, CandidateFilter(top_k=2, iou=0.3))
    assert scores(kept) == [9, 8]


def test_equal_scores_keep_input_order_and_are_deterministic() -> None:
    items = [boxed("a", 100 * i, 5) for i in range(4)]
    rule = CandidateFilter(top_k=3, iou=0.3)
    kept = filter_candidates(items, rule)
    assert kept == tuple(items[:3])
    assert filter_candidates(items, rule) == kept


def test_the_input_is_not_modified() -> None:
    items = [boxed("a", 0, 1), boxed("a", 100, 9)]
    before = list(items)
    filter_candidates(items, CandidateFilter())
    assert items == before
