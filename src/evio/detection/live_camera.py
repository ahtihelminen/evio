"""Per-camera live front-end: events -> {track_id: (patch, sensor bbox)}.

Lifts the per-camera detect -> patch -> track chain from
``scripts/run_evio_sapient.py`` so the panorama field test can feed
``evio.detection.live_overlay.LiveVerdictOverlay``. Patches are built with
``extract_event_patch`` (the corpus contract), so the v2 transformer sees
in-distribution input. The live event source is each camera worker's
``EventsBatch`` (wired at field bring-up); given a batch this class is pure
and unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from evio.core.events import EventsBatch
from evio.detection.dedupe import deduplicate_by_iou, top_k_by_mcoh
from evio.detection.grid import GridConfig
from evio.detection.pipeline import ClassSpecificConfig, DualBackgroundConfig
from evio.detection.pipeline.simplified_pipeline import (
    SimplifiedDetectionPipeline,
    SimplifiedPipelineConfig,
)
from evio.detection.reid import extract_event_patch
from evio.detection.tracker import TrackManager, TrackManagerConfig
from evio.rendering.panorama import sensor_box_to_view

Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CameraDetections:
    """One camera worker's per-track output, shipped to the main process.

    ``patches`` and ``boxes`` are keyed by the same (globally-unique) track
    ids; ``boxes`` are sensor-pixel (x0, y0, x1, y1). ``serial`` lets the
    main process find the camera's panorama strip slot.
    """

    serial: str
    patches: dict[str, np.ndarray]
    boxes: dict[str, Box]


class LiveCameraDetector:
    """Runs detect -> patch -> track on one camera's event batches."""

    def __init__(
        self,
        pipeline: SimplifiedDetectionPipeline,
        tracker: TrackManager,
        *,
        top_k: int = 1,
        iou_threshold: float = 0.3,
    ) -> None:
        self._pipeline = pipeline
        self._tracker = tracker
        self._top_k = top_k
        self._iou_threshold = iou_threshold

    def process(
        self, batch: EventsBatch,
    ) -> tuple[dict[str, np.ndarray], dict[str, Box]]:
        """Detect, extract patches, track; return per-track patch + bbox.

        Keys are confirmed-track ids; ``patches`` values are (32,32,3) uint8
        event-density crops (the corpus contract); ``boxes`` values are
        sensor-pixel (x0, y0, x1, y1). An empty batch yields empty dicts.
        """
        if batch.timestamps.size == 0:
            return {}, {}
        result = self._pipeline.update(batch)
        stamped = [
            replace(
                d, patch=extract_event_patch(batch, (d.x0, d.y0, d.x1, d.y1)),
            )
            for d in result.detections
        ]
        ranked = top_k_by_mcoh(stamped, k=self._top_k)
        deduped = deduplicate_by_iou(ranked, iou_threshold=self._iou_threshold)
        now_us = int(batch.timestamps[-1])
        tracked = self._tracker.update(deduped, now_us=now_us)
        patches = {
            d.track_id: d.patch
            for d in tracked
            if d.track_id and d.patch is not None
        }
        boxes = {
            d.track_id: (d.x0, d.y0, d.x1, d.y1)
            for d in tracked
            if d.track_id
        }
        return patches, boxes


def build_camera_detector(
    width: int,
    height: int,
    window_us: int,
    *,
    reid: bool = False,
    top_k: int = 1,
) -> LiveCameraDetector:
    """Construct a detector with the run_evio_sapient field configuration."""
    grid = GridConfig(
        width=width, height=height, rows=72, cols=128, time_bins=1,
        window_us=window_us,
    )
    class_cfg = ClassSpecificConfig(
        z_threshold=10, update_threshold=2, calibration_windows=10,
        count_weight=10, multi_weight=10, coactivity_weight=100,
        count_std_floor=1, multi_std_floor=1,
    )
    background = DualBackgroundConfig(
        sky=class_cfg, not_sky=class_cfg,
        alpha=5, ema_alpha=None, model_update_weight=0.1,
    )
    config = SimplifiedPipelineConfig(
        calibration_time_us=int(1e6), background=background,
    )
    pipeline = SimplifiedDetectionPipeline(grid, config)
    tracker = TrackManager(TrackManagerConfig(reid_enabled=reid))
    return LiveCameraDetector(pipeline, tracker, top_k=top_k)


def merge_camera_detections(
    results: list[CameraDetections],
    ordered_serials: list[str],
    rotated_serials: set[str],
    pan_offset: int,
    strip_width: int,
    view_width: int,
) -> tuple[dict[str, np.ndarray], dict[str, Box]]:
    """Map every camera's sensor boxes into shared panorama-view pixels.

    ``ordered_serials`` is the strip slot order (see
    ``panorama_serial_order``). A track whose box falls off the current view
    is dropped (both box and patch) so the overlay never tries to draw or
    classify something it can't show.
    """
    slot_of = {serial: i for i, serial in enumerate(ordered_serials)}
    patches: dict[str, np.ndarray] = {}
    view_boxes: dict[str, Box] = {}
    for result in results:
        slot = slot_of.get(result.serial)
        if slot is None:
            continue
        rotated = result.serial in rotated_serials
        for track_id, box in result.boxes.items():
            view_box = sensor_box_to_view(
                box, slot, pan_offset, strip_width, view_width,
                rotated=rotated,
            )
            if view_box is None:
                continue
            view_boxes[track_id] = view_box
            patches[track_id] = result.patches[track_id]
    return patches, view_boxes


__all__ = [
    "CameraDetections",
    "LiveCameraDetector",
    "build_camera_detector",
    "merge_camera_detections",
]
