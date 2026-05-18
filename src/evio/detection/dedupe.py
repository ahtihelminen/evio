"""Per-window deduplication + ranking of pipeline detections.

The TBD stage in `SimplifiedDetectionPipeline` emits one Detection per
anomalous grid tile. A drone large enough to occupy several tiles produces
several candidate Detections in the same window. We have two reducers:

- `deduplicate_by_iou` — greedy NMS by m_coh; keeps the highest-confidence
  detection per IoU cluster.
- `top_k_by_mcoh` — keep at most K detections per window, highest m_coh
  wins. Useful when the pipeline emits many spurious candidates with
  non-overlapping bboxes (so IoU dedup doesn't help).

Both are pure and order the survivors highest-m_coh first.
"""

from __future__ import annotations

from evio.detection.types import Detection


def deduplicate_by_iou(
    detections: list[Detection], iou_threshold: float = 0.3,
) -> list[Detection]:
    """Greedy non-max suppression: highest m_coh wins per IoU cluster."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: -d.m_coh)
    kept: list[Detection] = []
    for det in sorted_dets:
        if any(_iou(det, k) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def top_k_by_mcoh(
    detections: list[Detection], k: int,
) -> list[Detection]:
    """Keep at most `k` detections per window, ranked by m_coh descending."""
    if k <= 0 or len(detections) <= k:
        return list(detections)
    return sorted(detections, key=lambda d: -d.m_coh)[:k]


def _iou(a: Detection, b: Detection) -> float:
    inter_x0 = max(a.x0, b.x0)
    inter_y0 = max(a.y0, b.y0)
    inter_x1 = min(a.x1, b.x1)
    inter_y1 = min(a.y1, b.y1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(0, a.x1 - a.x0) * max(0, a.y1 - a.y0)
    area_b = max(0, b.x1 - b.x0) * max(0, b.y1 - b.y0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
