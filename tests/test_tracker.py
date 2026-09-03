

def test_eiou_metric_and_far_target_association() -> None:
    from evio.detection.tracker import (
        TrackManager,
        TrackManagerConfig,
        _diou,
        _eiou,
        _iou,
    )

    # Two 2-px boxes one pixel apart: IoU zeroes, EIoU keeps a graded
    # score (centre term dominates at tiny scales: ~-0.31).
    a, b = (100, 100, 102, 102), (103, 100, 105, 102)
    assert _iou(a, b) == 0.0
    assert -0.5 < _eiou(a, b) < 0.0
    # Identical boxes: EIoU == IoU == 1.
    assert _eiou(a, a) == 1.0
    # Approaching target (box doubles, centred): EIoU's size penalty
    # PUNISHES the growth (regression-loss artifact) — DIoU does not.
    small, big = (100, 100, 110, 110), (95, 95, 115, 115)
    assert _eiou(small, big) < _iou(small, big)
    assert _diou(small, big) == _iou(small, big)  # centred: no penalty

    from evio.detection.types import Detection

    def det(x0: int, y0: int, x1: int, y1: int) -> Detection:
        return Detection(
            x0=x0, y0=y0, x1=x1, y1=y1, vx_px_s=0.0, vy_px_s=0.0,
            m_coh=100.0, best_psr=1.0, event_count=10, bg_class=0,
            row_start=0, row_end=1, col_start=0, col_end=1,
        )

    # A glint-jumping 32-px far target (+40 px/window: IoU = 0 between
    # windows): the IoU tracker fragments — never confirms; EIoU holds one
    # identity to confirmation.
    # Stationary far target with glint jitter: detections alternate +-40 px
    # around a point (KF-consistent with the vx=0 observations). Successive
    # boxes are disjoint (40 px jump, 32 px box) so IoU-association
    # fragments; the distance-aware metrics hold one identity.
    positions = [(200 + (40 if i % 2 else 0), 300) for i in range(6)]
    for metric, expect_confirmed in (("iou", 0), ("eiou", 1), ("diou", 1)):
        tracker = TrackManager(TrackManagerConfig(
            match_metric=metric, confirm_after_hits=5, max_miss_frames=8,
        ))
        confirmed_ids = set()
        for i, (x, y) in enumerate(positions):
            out = tracker.update([det(x, y, x + 32, y + 32)],
                                 now_us=(i + 1) * 10_000)
            confirmed_ids |= {d.track_id for d in out if d.track_id}
        assert len(confirmed_ids) == expect_confirmed, (
            f"{metric}: {confirmed_ids}"
        )
