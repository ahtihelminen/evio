"""Dedup behaviour: keep highest m_coh per IoU cluster."""

from __future__ import annotations

from evio.detection.dedupe import deduplicate_by_iou, top_k_by_mcoh
from evio.detection.types import Detection


def _det(x0: int, y0: int, x1: int, y1: int, m_coh: float) -> Detection:
    return Detection(
        x0=x0, y0=y0, x1=x1, y1=y1,
        vx_px_s=0.0, vy_px_s=0.0,
        m_coh=m_coh, best_psr=1.0, event_count=100,
        bg_class=0, row_start=0, row_end=1, col_start=0, col_end=1,
        z_score=3.0, camera_id="cam",
    )


def test_empty_in_empty_out() -> None:
    assert deduplicate_by_iou([]) == []


def test_single_detection_passes_through() -> None:
    d = _det(10, 10, 50, 50, m_coh=0.8)
    assert deduplicate_by_iou([d]) == [d]


def test_overlapping_keeps_highest_mcoh() -> None:
    a = _det(0, 0, 100, 100, m_coh=0.5)
    b = _det(10, 10, 110, 110, m_coh=0.9)
    c = _det(20, 20, 120, 120, m_coh=0.7)
    survivors = deduplicate_by_iou([a, b, c], iou_threshold=0.3)
    assert survivors == [b]


def test_disjoint_detections_all_survive() -> None:
    a = _det(0, 0, 50, 50, m_coh=0.8)
    b = _det(200, 200, 250, 250, m_coh=0.6)
    survivors = deduplicate_by_iou([a, b], iou_threshold=0.3)
    assert set(survivors) == {a, b}


def test_top_k_keeps_highest_mcoh() -> None:
    dets = [
        _det(0, 0, 10, 10, m_coh=v) for v in (0.1, 0.9, 0.5, 0.8, 0.3)
    ]
    kept = top_k_by_mcoh(dets, k=2)
    assert [d.m_coh for d in kept] == [0.9, 0.8]


def test_top_k_zero_is_identity() -> None:
    dets = [_det(0, 0, 1, 1, m_coh=0.5)]
    assert top_k_by_mcoh(dets, k=0) == dets


def test_top_k_larger_than_input_returns_all() -> None:
    dets = [_det(0, 0, 1, 1, m_coh=0.5)]
    assert top_k_by_mcoh(dets, k=5) == dets


def test_iou_threshold_controls_aggressiveness() -> None:
    a = _det(0, 0, 100, 100, m_coh=0.9)
    b = _det(50, 50, 150, 150, m_coh=0.5)
    loose = deduplicate_by_iou([a, b], iou_threshold=0.5)
    strict = deduplicate_by_iou([a, b], iou_threshold=0.1)
    assert len(loose) == 2
    assert len(strict) == 1
