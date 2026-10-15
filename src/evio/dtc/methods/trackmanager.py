"""The deployed Kalman `TrackManager`, as a per-stream tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from evio.detection.tracker import TrackManager, TrackManagerConfig, _Track
from evio.detection.types import Detection
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import (
    Box,
    Candidate,
    Candidates,
    Region,
    Step,
    Track,
    Tracks,
)

POINT_HALF_SIZE_PX = 1


@dataclass(frozen=True)
class TrackManagerOptions:
    """Main `TrackManagerConfig` fields with its defaults.

    The sensor size defaults to the EXTENT fact. `deterministic_ids` numbers
    tracks `<stream>-<n>` in spawn order so replay is reproducible; off, the
    tracker's ULID passes through (the SAPIENT object id).
    """

    match_iou_threshold: float = 0.2
    match_metric: str = "iou"
    match_eiou_threshold: float = -0.3
    max_miss_frames: int = 8
    process_noise_pos_px_per_s: float = 5.0
    process_noise_vel_px_per_s2: float = 200.0
    obs_noise_pos_px: float = 8.0
    obs_noise_vel_px_per_s: float = 100.0
    bbox_smoothing: float = 0.3
    min_confidence_factor: float = 0.1
    confirm_after_hits: int = 3
    spawn_min_mcoh: float = 0.0
    mahalanobis_matching_enabled: bool = False
    velocity_coherence_enabled: bool = False
    persistence_gate_enabled: bool = False
    spawn_density_enabled: bool = False
    spatial_dormant_enabled: bool = False
    adaptive_gate_enabled: bool = False
    sensor_width_px: int | None = None
    sensor_height_px: int | None = None
    deterministic_ids: bool = True


def _float_box(box: tuple[int, int, int, int]) -> Box:
    x0, y0, x1, y1 = box
    return (float(x0), float(y0), float(x1), float(y1))


def _detection(candidate: Candidate) -> Detection:
    x0, y0, x1, y1 = candidate.region.box or (
        *(c - POINT_HALF_SIZE_PX for c in candidate.region.center),
        *(c + POINT_HALF_SIZE_PX for c in candidate.region.center),
    )
    vx, vy = candidate.region.velocity_px_s or (0.0, 0.0)
    features = candidate.features
    return Detection(
        x0=round(x0),
        y0=round(y0),
        x1=round(x1),
        y1=round(y1),
        vx_px_s=vx,
        vy_px_s=vy,
        m_coh=candidate.score,
        best_psr=features.get("best_psr", 0.0),
        event_count=int(features.get("event_count", 0)),
        bg_class=int(features.get("bg_class", 0)),
        row_start=0,
        row_end=0,
        col_start=0,
        col_end=0,
        z_score=features.get("z_score", 0.0),
    )


class NumberedTrackManager(TrackManager):
    """A `TrackManager` numbering its tracks `<stream>-<n>` in spawn order.

    The stock ids are ULIDs, which read the clock and differ between runs.
    """

    def __init__(self, stream: str, config: TrackManagerConfig) -> None:
        super().__init__(config)
        self._stream = stream
        self._spawned = 0

    def _spawn(
        self, det: Detection, now_us: int, track_id: str | None = None
    ) -> _Track:
        if track_id is None:
            track_id = f"{self._stream}-{self._spawned}"
            self._spawned += 1
        return super()._spawn(det, now_us, track_id)


@method(
    "trackmanager",
    stages=(Stage.TRACK,),
    config=TrackManagerOptions,
)
class TrackManagerTracker:
    """Kalman tracks with confirmation, coasting and optional re-id."""

    def __init__(self, config: TrackManagerOptions, context: Context) -> None:
        self.config = config
        self.stream = str(context.stream)
        width, height = context.facts[EXTENT]
        fields = asdict(config)
        del fields["deterministic_ids"]
        fields["sensor_width_px"] = config.sensor_width_px or width
        fields["sensor_height_px"] = config.sensor_height_px or height
        self.manager_config = TrackManagerConfig(**fields)
        self.reset()

    def reset(self) -> None:
        self._manager = (
            NumberedTrackManager(self.stream, self.manager_config)
            if self.config.deterministic_ids
            else TrackManager(self.manager_config)
        )

    def step(self, step: Step, candidates: Candidates) -> Tracks:
        detections = self._manager.update(
            [_detection(candidate) for candidate in candidates.items],
            now_us=step.t_end_us,
        )
        return Tracks(
            tuple(self._track(detection) for detection in detections),
            frozenset(self._manager.live_track_ids()),
            tuple(
                Region(box=_float_box(box))
                for box in self._manager.live_boxes(confirmed_only=True)
            ),
        )

    def _track(self, detection: Detection) -> Track:
        box = (detection.x0, detection.y0, detection.x1, detection.y1)
        return Track(
            self.stream,
            detection.track_id,
            Region(
                box=_float_box(box),
                velocity_px_s=(detection.vx_px_s, detection.vy_px_s),
            ),
            detection.m_coh,
            self._manager.track_features(detection.track_id) or {},
        )
