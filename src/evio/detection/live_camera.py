"""Torch-free per-camera detection, runnable inside a camera worker.

Splits the detect -> track -> patch/tokenize stage of the live pipeline
away from the transformer classifier. The classifier is shared and stays
in the consumer's main process; everything
here is pure numpy, so it can run N-way parallel in the per-camera worker
processes, which already hold each camera's ``EventsBatch`` locally.

:meth:`CameraDetector.update` ingests one ``EventsBatch`` and returns a
:class:`CameraDetectionResult`: the boxes to draw, the recorder object-ids
to mark, and — for tracks whose 16-frame ring just filled — the stacked
patches (and, on the v3 path, the tokenized events) the transformer scores.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from evio.core.events import EventsBatch
from evio.detection.dedupe import deduplicate_by_iou, top_k_by_mcoh
from evio.detection.event_tokens import DEFAULT_E_MAX, tokenize_events_in_track
from evio.detection.pipeline import build
from evio.detection.reid import extract_event_patch
from evio.detection.tracker import TrackManager, TrackManagerConfig

if TYPE_CHECKING:
    from evio.detection.pipeline import DetectionPipeline
    from evio.detection.types import Detection

_log = logging.getLogger(__name__)

# v2 transformer fixed input geometry (must match the D_mae8 corpus).
SEQ_LEN = 16
PATCH_SIZE = 32

# v3 raw-event branch: per-track event tokens are sub-sampled to this many.
EVENT_MAX = DEFAULT_E_MAX

# Drop a track's ring once it has been unseen this many windows (~3 s at a
# 10 ms window). Tolerates brief occlusion misses while keeping the
# per-camera bookkeeping bounded.
_STALE_TICKS = 300

_Bbox = tuple[int, int, int, int]


@dataclass(slots=True)
class TrackBox:
    bbox: _Bbox
    track_id: str


@dataclass(slots=True)
class CameraDetectionResult:
    """One tick of detection output, shipped from worker to main.

    ``patches``/``events``/``mask`` are populated only for ``ready_ids`` —
    tracks whose patch ring just reached ``SEQ_LEN``. ``events``/``mask``
    are ``None`` on the v2 path (no raw-event branch) and when v3 history
    is still warming up.
    """

    boxes: list[TrackBox]
    ready_ids: list[str]
    patches: np.ndarray | None
    events: np.ndarray | None
    mask: np.ndarray | None
    object_ids: list[str]
    live_ids: frozenset[str]


class CameraDetector:
    """GMM-online detector -> tracker -> per-track patch/event ring buffers.

    Pure numpy and holds no torch, so it is safe to construct inside a
    camera worker process. The shared transformer classifier consumes the
    patches/events this emits; it lives in the main process.
    """

    def __init__(
        self,
        *,
        serial: str,
        sensor_width: int = 1280,
        sensor_height: int = 720,
        window_us: int = 10_000,
        top_k: int = 20,
        is_v3: bool = False,
    ) -> None:
        self._serial = serial
        self._window_us = window_us
        self._top_k = top_k
        self._is_v3 = is_v3
        # The registry's only entry, and the same pipeline the SAPIENT path
        # runs. This branch is not reachable from the app today -- nothing
        # passes `detect_enabled=True` -- but it must name a pipeline that
        # exists, and naming a different one from the rest of the system is
        # how `gmm_online` came to look like the live background model.
        self._pipeline: DetectionPipeline = build(
            "simplified",
            {
                "width": sensor_width,
                "height": sensor_height,
                "window_us": window_us,
            },
        )
        self._tracker = TrackManager(TrackManagerConfig(reid_enabled=True))
        self._rings: dict[str, deque[np.ndarray]] = {}
        self._bbox_rings: dict[str, deque[tuple[_Bbox, int]]] = {}
        self._batch_ring: deque[EventsBatch] = deque(maxlen=SEQ_LEN)
        self._seen_tick: dict[str, int] = {}
        self._tick = 0

    def update(self, batch: EventsBatch) -> CameraDetectionResult:
        self._tick += 1
        if self._is_v3:
            self._batch_ring.append(batch)
        candidates = self._pipeline.update(batch).detections
        ranked = top_k_by_mcoh(candidates, k=self._top_k)
        deduped = deduplicate_by_iou(ranked, iou_threshold=0.3)
        with_patch = [
            replace(
                det,
                camera_id=self._serial,
                patch=extract_event_patch(
                    batch, (det.x0, det.y0, det.x1, det.y1),
                ),
            )
            for det in deduped
        ]
        tracked = self._tracker.update(
            with_patch, now_us=int(batch.timestamps[-1]),
        )

        ts_mid_us = int(
            (int(batch.timestamps[0]) + int(batch.timestamps[-1])) // 2,
        )
        for det in tracked:
            self._seen_tick[det.track_id] = self._tick
            if det.patch is not None:
                self._rings.setdefault(
                    det.track_id, deque(maxlen=SEQ_LEN),
                ).append(det.patch)
                if self._is_v3:
                    self._bbox_rings.setdefault(
                        det.track_id, deque(maxlen=SEQ_LEN),
                    ).append(((det.x0, det.y0, det.x1, det.y1), ts_mid_us))

        ready, patches, events, mask = self._classify_inputs(tracked)
        boxes = [
            TrackBox((det.x0, det.y0, det.x1, det.y1), det.track_id)
            for det in tracked
        ]
        object_ids = [det.track_id for det in tracked if det.track_id]
        self._prune()
        return CameraDetectionResult(
            boxes=boxes,
            ready_ids=ready,
            patches=patches,
            events=events,
            mask=mask,
            object_ids=object_ids,
            live_ids=frozenset(self._seen_tick),
        )

    def _classify_inputs(
        self, tracked: list[Detection],
    ) -> tuple[
        list[str], np.ndarray | None, np.ndarray | None, np.ndarray | None,
    ]:
        """Stack the patches (and v3 events) for tracks ready to classify.

        A track is ready when it was seen this tick and its patch ring is
        full. Returns empty/``None`` when nothing is ready, or when the v3
        event history is not built yet (so the caller skips this tick).
        """
        ready = [
            tid for tid in dict.fromkeys(d.track_id for d in tracked)
            if len(self._rings.get(tid, ())) == SEQ_LEN
        ]
        if not ready:
            return [], None, None, None
        patches = np.stack(
            [np.stack(tuple(self._rings[tid]), axis=0) for tid in ready],
            axis=0,
        ).astype(np.uint8)
        if not self._is_v3:
            return ready, patches, None, None
        events, mask = self._build_event_batch(ready)
        if events is None:
            return [], None, None, None
        return ready, patches, events, mask

    def _build_event_batch(
        self, ready: list[str],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Concat the recent batches and tokenize per ready track.

        Returns ``(events, mask)`` shaped ``(B, E_max, 4)`` / ``(B, E_max)``
        or ``(None, None)`` if there isn't enough event history yet (the
        camera ring is still warming up).
        """
        if not self._batch_ring:
            return None, None
        merged = _concat_batches(tuple(self._batch_ring))
        all_events = np.zeros((len(ready), EVENT_MAX, 4), dtype=np.float32)
        all_mask = np.zeros((len(ready), EVENT_MAX), dtype=np.bool_)
        for i, tid in enumerate(ready):
            ring = self._bbox_rings.get(tid)
            if ring is None or len(ring) != SEQ_LEN:
                continue
            bboxes = [item[0] for item in ring]
            timestamps_us = [item[1] for item in ring]
            ev, m = tokenize_events_in_track(
                merged,
                bboxes=bboxes,
                timestamps_us=timestamps_us,
                window_us=self._window_us,
                e_max=EVENT_MAX,
            )
            all_events[i] = ev
            all_mask[i] = m
        return all_events, all_mask

    def _prune(self) -> None:
        stale = [
            tid for tid, tick in self._seen_tick.items()
            if self._tick - tick > _STALE_TICKS
        ]
        for tid in stale:
            self._seen_tick.pop(tid, None)
            self._rings.pop(tid, None)
            self._bbox_rings.pop(tid, None)


def safe_update(
    detector: CameraDetector, batch: EventsBatch,
) -> CameraDetectionResult | None:
    """Run one detection tick, swallowing any failure.

    Worker-side detection is best-effort: a failure must disable detection
    for this camera, never crash the streaming/recording worker.
    """
    try:
        return detector.update(batch)
    except Exception:
        _log.exception("worker-side detection failed; disabling for camera")
        return None


def _concat_batches(batches: tuple[EventsBatch, ...]) -> EventsBatch:
    """Concat consecutive EventsBatches into one (timestamps stay sorted)."""
    if len(batches) == 1:
        return batches[0]
    return EventsBatch(
        timestamps=np.concatenate([b.timestamps for b in batches]),
        x_coords=np.concatenate([b.x_coords for b in batches]),
        y_coords=np.concatenate([b.y_coords for b in batches]),
        polarities=np.concatenate([b.polarities for b in batches]),
    )


__all__ = [
    "EVENT_MAX",
    "PATCH_SIZE",
    "SEQ_LEN",
    "CameraDetectionResult",
    "CameraDetector",
    "TrackBox",
    "safe_update",
]
