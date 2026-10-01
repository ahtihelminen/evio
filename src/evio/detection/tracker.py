"""Constant-velocity Kalman tracker for evio detections.

`TrackManager.update(detections, now_us)` is what callers run once per
event window after dedupe. Returns the same detections annotated with
stable per-track `track_id`s (ULIDs reusable as SAPIENT `object_id`s) and
a `m_coh` scaled by Kalman fit quality.

State per track: `[cx, cy, vx, vy]` in pixels and pixels/second. The
Detection dataclass already carries vx/vy from the pipeline's TBD stage,
so the observation matrix is identity. Constant-velocity transition with
configurable process + observation noise.

Matching is greedy by IoU between the detection bbox and a predicted-
forward track bbox (track stores the last-observed half-width/half-height
to reconstruct the bbox at the predicted centroid). Unmatched detections
spawn fresh tracks; unmatched tracks age out after `max_miss_frames`
consecutive misses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from evio.core.ids import new_ulid
from evio.detection.persistence import PersistenceGateConfig, PersistenceMap
from evio.detection.reid import (
    EmbeddingBuffer,
    SignatureBuffer,
    cosine_similarity,
    normalize_patch,
    unit_normalize,
)
from evio.detection.spawn_density import SpawnDensityConfig, SpawnDensityMap

if TYPE_CHECKING:
    from evio.detection.types import Detection

# 4D chi-square 95th percentile ≈ 9.49; below this, fit is "good".
_CHI2_4DOF_95 = 9.49
# 4D chi-square 95th percentile = 9.49; 99% = 13.28; 99.9% = 18.47.
_DEFAULT_MAHAL_GATE = 13.28


@dataclass(slots=True)
class _Track:
    track_id: str
    x: NDArray[np.float64]  # [cx, cy, vx, vy]
    cov: NDArray[np.float64]  # 4x4
    half_width: float
    half_height: float
    last_seen_us: int
    age_frames: int = 1
    miss_count: int = 0
    signature: SignatureBuffer = field(default_factory=SignatureBuffer)
    embedding: EmbeddingBuffer = field(default_factory=EmbeddingBuffer)
    # Velocity-coherence accumulator used by the F1 confirmation rule.
    # `speed_samples` stores observed speeds (px/s); `path_history` is the
    # last N centroids for the straightness measure.
    speed_samples: list[float] = field(default_factory=list)
    path_history: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class _DormantTrack:
    """A confirmed track whose Kalman state has timed out. Held in the
    re-ID dormant pool so an occluded-then-reappearing drone can resurrect
    its original track_id instead of getting a fresh one.

    ``signature`` (raw-patch) and ``embedding`` (learned v4 ReID) are the
    appearance features frozen at eviction; both are populated only when
    ``reid_enabled`` is set. The F1.5 spatial-only dormant pool uses
    ``last_state`` + ``last_cov`` for a motion-extrapolated Mahalanobis match
    (no appearance), so both can be ``None`` for purely classical reuse.
    """

    track_id: str
    signature: NDArray[np.float32] | None
    embedding: NDArray[np.float32] | None
    last_seen_us: int
    last_bbox: tuple[int, int, int, int]
    last_state: NDArray[np.float64]  # [cx, cy, vx, vy] at eviction time
    last_cov: NDArray[np.float64]    # 4x4 covariance at eviction time
    half_width: float                # for re-spawn bbox sizing
    half_height: float


@dataclass(slots=True)
class TrackManagerConfig:
    match_iou_threshold: float = 0.2
    # "eiou" association scores centre/size proximity even for disjoint
    # boxes — built for far (1-2 px) and fast-approaching targets where
    # plain IoU zeroes out and the track dies at confirmation. EIoU is in
    # (-1, 1]; for SMALL boxes the enclosing-box normalisation makes the
    # centre term dominate (two 2-px boxes one pixel apart already score
    # ~-0.31), so the threshold is NEGATIVE: -0.3 associates pipeline-scale
    # (32 px) boxes across ~45 px single-window jumps while rejecting
    # cross-target distances.
    match_metric: str = "iou"  # "iou" | "eiou" | "diou" (diou recommended)
    match_eiou_threshold: float = -0.3
    max_miss_frames: int = 8
    process_noise_pos_px_per_s: float = 5.0
    process_noise_vel_px_per_s2: float = 200.0
    obs_noise_pos_px: float = 8.0
    obs_noise_vel_px_per_s: float = 100.0
    bbox_smoothing: float = 0.3  # EMA factor for half-extents
    min_confidence_factor: float = 0.1
    # SORT-style track confirmation: spawn a tentative track on first sight
    # but only emit detections from it after it has been re-observed this
    # many times. Filters noise spikes that briefly form fake tracks.
    confirm_after_hits: int = 3
    # Unmatched detections only spawn a new track if their m_coh meets
    # this floor. Set well above the pipeline's noise floor to stop
    # transient high-z-score blobs from fragmenting an existing track.
    # 0 = spawn anything (legacy behaviour).
    spawn_min_mcoh: float = 0.0
    # Signature-based re-identification: when a confirmed track times out,
    # move it to a dormant pool instead of deleting; if an unmatched
    # detection's patch matches a dormant signature, resurrect the original
    # track_id. Opt-in — set reid_enabled=True and populate Detection.patch.
    reid_enabled: bool = False
    # Cosine similarity is sharply concentrated on event-density patches
    # (typical same-drone cosine is 0.92-0.99 on FRED 84), so the default
    # is well above 0.5; tune empirically per dataset.
    reid_similarity_threshold: float = 0.9
    reid_dormant_timeout_s: float = 10.0
    reid_signature_buffer: int = 10
    # Phase-2 learned re-ID: when set, the appearance match uses the v4
    # transformer's L2-normed embedding (``Detection.embedding``) instead of
    # the raw-patch signature, falling back to the patch on any detection
    # that carries no embedding. The learned embedding lives in a different
    # space than the grayscale patch, so it gets its own cosine floor (a
    # contrastive embedding's same-id cosine sits lower than the patch's).
    reid_use_embedding: bool = False
    reid_embedding_threshold: float = 0.5

    # ---- F1: persistence-gate + Mahalanobis-first matching ----
    # Enable the persistent-cluster suppression map.  Detections in cells
    # that fire in >`persistence_hit_rate_threshold` of recent windows
    # AND with mean speed below `persistence_max_velocity_px_s` get
    # dropped before the tracker. Off by default; flip on for F1.
    persistence_gate_enabled: bool = False
    # Use Mahalanobis-distance gating instead of greedy IoU.  Cosine
    # similarity becomes a tiebreaker (only invoked when ≥2 tracks gate
    # the same detection).
    mahalanobis_matching_enabled: bool = False
    # 4-D chi-square cutoff for the Mahalanobis gate.  Default = 99% CI;
    # plan calibration range is 9.5 - 16.3.
    mahalanobis_chi2_cutoff: float = _DEFAULT_MAHAL_GATE
    # Velocity-coherence confirmation rule (replaces plain hit count).
    # A tentative track promotes to confirmed once it has been seen
    # `confirm_after_hits` times AND the velocity-coherence check passes.
    # Set False to keep SORT's hit-count-only behaviour.
    velocity_coherence_enabled: bool = False
    velocity_min_confirm_px_s: float = 25.0
    velocity_cv_max: float = 0.6
    velocity_straightness_min: float = 0.3
    velocity_history_len: int = 8
    # F1 persistence-gate parameters (passed straight through to the
    # ``PersistenceMap`` helper).
    persistence_grid_rows: int = 64
    persistence_grid_cols: int = 64
    persistence_window_history: int = 200
    persistence_hit_rate_threshold: float = 0.30
    persistence_max_velocity_px_s: float = 5.0
    persistence_warmup_windows: int = 30
    # F1 sensor geometry (defaults match the IMX636 used across evio).
    sensor_width_px: int = 1280
    sensor_height_px: int = 720

    # ---- F1.5: spawn-density suppressor (rain / sky-blob filter) ----
    # Per-cell counter of NEW track_id spawns in a rolling time window.
    # A region producing many short-lived spawn/die cycles is almost
    # certainly noise — suppress further spawns there until the rate
    # drops. Existing tracks passing through the region are unaffected.
    spawn_density_enabled: bool = False
    spawn_density_grid_rows: int = 64
    spawn_density_grid_cols: int = 64
    spawn_density_window_s: float = 5.0
    spawn_density_max_spawns: int = 4
    spawn_density_cooldown_s: float = 5.0

    # ---- F1.5: spatial-only dormant pool (slowdown / dir-change fix) ----
    # When a confirmed track dies, save its last Kalman state + covariance.
    # An unmatched detection that lands within a motion-extrapolated
    # Mahalanobis ellipse of any dormant track resurrects the original
    # track_id. No appearance lookup — pure motion continuity, so it's
    # safe even without trained signatures.
    spatial_dormant_enabled: bool = False
    spatial_dormant_timeout_s: float = 2.0
    # χ²(4) cutoff for the dormant-extrapolated state.  Generous because
    # the covariance has already been inflated by the dormant interval.
    spatial_dormant_chi2_cutoff: float = 25.0

    # ---- F1.5: adaptive Mahalanobis gate (preempts fragmentation) ----
    # Loosen the χ² cutoff for confirmed tracks that have recently slowed.
    # `widen_factor` multiplies the base cutoff; e.g. 2.0 doubles the gate
    # so a slowing drone is still matched. Only applies once the track is
    # past `min_age_frames` AND moving slower than `low_speed_px_s`.
    adaptive_gate_enabled: bool = False
    adaptive_gate_min_age_frames: int = 5
    adaptive_gate_low_speed_px_s: float = 50.0
    adaptive_gate_widen_factor: float = 2.0


class TrackManager:
    """Per-camera Kalman tracker. Stateful across calls."""

    def __init__(self, config: TrackManagerConfig | None = None) -> None:
        self._cfg = config or TrackManagerConfig()
        self._tracks: dict[str, _Track] = {}
        self._dormant: dict[str, _DormantTrack] = {}
        self._last_us: int | None = None
        self._persistence = PersistenceMap(
            PersistenceGateConfig(
                grid_rows=self._cfg.persistence_grid_rows,
                grid_cols=self._cfg.persistence_grid_cols,
                window_history=self._cfg.persistence_window_history,
                hit_rate_threshold=self._cfg.persistence_hit_rate_threshold,
                max_velocity_px_s=self._cfg.persistence_max_velocity_px_s,
                warmup_windows=self._cfg.persistence_warmup_windows,
                enabled=self._cfg.persistence_gate_enabled,
            ),
            width=self._cfg.sensor_width_px,
            height=self._cfg.sensor_height_px,
        )
        self._persistence_drops = 0
        self._spawn_density = SpawnDensityMap(
            SpawnDensityConfig(
                grid_rows=self._cfg.spawn_density_grid_rows,
                grid_cols=self._cfg.spawn_density_grid_cols,
                window_s=self._cfg.spawn_density_window_s,
                max_spawns=self._cfg.spawn_density_max_spawns,
                cooldown_s=self._cfg.spawn_density_cooldown_s,
                enabled=self._cfg.spawn_density_enabled,
            ),
            width=self._cfg.sensor_width_px,
            height=self._cfg.sensor_height_px,
        )
        self._spawn_density_drops = 0
        self._dormant_resurrections = 0

    @property
    def dormant_track_count(self) -> int:
        return len(self._dormant)

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def persistence_map(self) -> PersistenceMap:
        return self._persistence

    @property
    def persistence_drops(self) -> int:
        return self._persistence_drops

    @property
    def spawn_density_map(self) -> SpawnDensityMap:
        return self._spawn_density

    @property
    def spawn_density_drops(self) -> int:
        return self._spawn_density_drops

    @property
    def dormant_resurrections(self) -> int:
        return self._dormant_resurrections

    def live_track_ids(self) -> set[str]:
        """IDs of every live track, INCLUDING ones coasting through misses.

        Verdict-gate state must be pruned against this set — pruning against
        a single window's emitted detections deletes SPRT/latch state on
        every missed window, defeating the gate's dropout-survival property.
        """
        return set(self._tracks)

    def track_features(self, track_id: str) -> dict[str, float] | None:
        """Kinematic feature snapshot for a live track (None if unknown).

        Read-only motion features for the fusion verdict gate
        (`evio.detection.kinematic_llr`) and its offline trainer; combined with
        the detection's quality scores via ``assemble_features``. No effect on
        tracking behaviour.
        """
        track = self._tracks.get(track_id)
        if track is None:
            return None
        samples = track.speed_samples
        mean_speed, cv = _speed_stats(samples) if samples else (0.0, 0.0)
        if not math.isfinite(cv):
            cv = 0.0
        return {
            "speed_mean": mean_speed,
            "speed_cv": cv,
            "straightness": _straightness(track.path_history),
            "vel_mag": math.hypot(float(track.x[2]), float(track.x[3])),
            "age_frames": float(track.age_frames),
            "miss_count": float(track.miss_count),
            "bbox_diag": math.hypot(
                2.0 * track.half_width, 2.0 * track.half_height,
            ),
            "pos_cov_trace": float(track.cov[0, 0] + track.cov[1, 1]),
        }

    def live_boxes(
        self, *, confirmed_only: bool = True,
    ) -> list[tuple[int, int, int, int]]:
        """Current bbox of each live track — the R1 keep-hysteresis ROI set.

        Read-only; lets the consumer-side R1 gate (`evio.detection.r1_gate`,
        0b) rescue a weak candidate that overlaps a track the tracker already
        holds. ``confirmed_only`` restricts to tracks past
        ``confirm_after_hits``.
        """
        boxes: list[tuple[int, int, int, int]] = []
        for track in self._tracks.values():
            if (
                confirmed_only
                and track.age_frames < self._cfg.confirm_after_hits
            ):
                continue
            cx, cy = float(track.x[0]), float(track.x[1])
            boxes.append((
                int(cx - track.half_width), int(cy - track.half_height),
                int(cx + track.half_width), int(cy + track.half_height),
            ))
        return boxes

    def update(
        self, detections: list[Detection], now_us: int,
    ) -> list[Detection]:
        observations = [
            (
                (d.x0 + d.x1) / 2.0,
                (d.y0 + d.y1) / 2.0,
                math.hypot(d.vx_px_s, d.vy_px_s),
            )
            for d in detections
        ]
        self._persistence.observe(observations)
        if self._cfg.persistence_gate_enabled:
            kept = self._persistence.filter(detections)
            self._persistence_drops += len(detections) - len(kept)
            detections = kept

        dt = self._dt(now_us)
        self._predict_all(dt)
        matches, unmatched_dets, unmatched_tracks = self._match(detections)

        annotated: list[Detection] = []
        for det, track in matches:
            mahal = self._update_track_kalman(track, det, now_us)
            if track.age_frames < self._cfg.confirm_after_hits:
                continue  # tentative — wait for confirmation
            if (
                self._cfg.velocity_coherence_enabled
                and not _passes_velocity_coherence(track, self._cfg)
            ):
                continue  # incoherent motion — likely clutter
            factor = self._confidence_factor(mahal)
            annotated.append(replace(
                det,
                track_id=track.track_id,
                m_coh=max(
                    self._cfg.min_confidence_factor,
                    min(1.0, det.m_coh * factor),
                ),
            ))

        for det in unmatched_dets:
            new_track = self._resurrect_or_spawn(det, now_us)
            if new_track is None:
                continue
            if new_track.age_frames >= self._cfg.confirm_after_hits:
                annotated.append(
                    replace(det, track_id=new_track.track_id),
                )

        for track in unmatched_tracks:
            track.miss_count += 1

        self._evict_stale(now_us)
        self._last_us = now_us
        return annotated

    def _dt(self, now_us: int) -> float:
        if self._last_us is None:
            return 0.0
        return max(0.0, (now_us - self._last_us) / 1_000_000.0)

    def _predict_all(self, dt: float) -> None:
        if dt <= 0.0:
            return
        f = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
        )
        q = self._process_noise(dt)
        for track in self._tracks.values():
            track.x = f @ track.x
            track.cov = f @ track.cov @ f.T + q

    def _process_noise(self, dt: float) -> NDArray[np.float64]:
        sig_p = self._cfg.process_noise_pos_px_per_s
        sig_v = self._cfg.process_noise_vel_px_per_s2
        return np.diag([
            (sig_p * dt) ** 2,
            (sig_p * dt) ** 2,
            (sig_v * dt) ** 2,
            (sig_v * dt) ** 2,
        ])

    def _match(
        self, detections: list[Detection],
    ) -> tuple[
        list[tuple[Detection, _Track]],
        list[Detection],
        list[_Track],
    ]:
        unmatched_tracks = list(self._tracks.values())
        matches: list[tuple[Detection, _Track]] = []
        unmatched_dets: list[Detection] = []
        for det in detections:
            if self._cfg.mahalanobis_matching_enabled:
                best_track = self._match_by_mahalanobis(det, unmatched_tracks)
            else:
                best_track = self._match_by_iou(det, unmatched_tracks)
            if best_track is None:
                # Drone may have darted past the Kalman prediction (event-
                # camera detections are jittery). Fall back to appearance
                # similarity against the remaining active tracks before
                # spawning a fresh ID.
                best_track = self._match_by_signature(det, unmatched_tracks)
            if best_track is None:
                unmatched_dets.append(det)
            else:
                matches.append((det, best_track))
                unmatched_tracks.remove(best_track)
        return matches, unmatched_dets, unmatched_tracks

    def _match_by_mahalanobis(
        self, det: Detection, candidates: list[_Track],
    ) -> _Track | None:
        """Mahalanobis-distance gate; cosine tiebreaker when ≥2 tracks gate."""
        if not candidates:
            return None
        z = np.array([
            (det.x0 + det.x1) / 2.0, (det.y0 + det.y1) / 2.0,
            det.vx_px_s, det.vy_px_s,
        ])
        r = np.diag([
            self._cfg.obs_noise_pos_px ** 2,
            self._cfg.obs_noise_pos_px ** 2,
            self._cfg.obs_noise_vel_px_per_s ** 2,
            self._cfg.obs_noise_vel_px_per_s ** 2,
        ])
        gated: list[tuple[float, _Track]] = []
        best_dist = float("inf")
        best_track: _Track | None = None
        for track in candidates:
            innov = z - track.x
            s = track.cov + r
            try:
                s_inv = np.linalg.inv(s)
            except np.linalg.LinAlgError:
                continue
            dist = float(innov @ s_inv @ innov)
            cutoff = self._effective_chi2_cutoff(track)
            if dist <= cutoff:
                gated.append((dist, track))
            if dist < best_dist and dist <= cutoff:
                best_dist = dist
                best_track = track
        if len(gated) <= 1:
            return best_track
        return self._tiebreak_by_signature(det, gated) or gated[0][1]

    def _effective_chi2_cutoff(self, track: _Track) -> float:
        """χ² cutoff widened for confirmed-but-slowing tracks (Fix 3).

        When a drone slows or changes direction, the constant-velocity
        prediction goes wrong → the next real detection lands outside
        the tight χ² ellipse → track fragments. We widen the gate for
        tracks past their confirmation threshold AND moving slower than
        ``adaptive_gate_low_speed_px_s`` so the drone keeps its ID.
        Young/uncommitted tracks keep the tight gate so noise can't
        glue onto them.
        """
        base = self._cfg.mahalanobis_chi2_cutoff
        if not self._cfg.adaptive_gate_enabled:
            return base
        if track.age_frames < self._cfg.adaptive_gate_min_age_frames:
            return base
        speed = math.hypot(float(track.x[2]), float(track.x[3]))
        if speed >= self._cfg.adaptive_gate_low_speed_px_s:
            return base
        return base * self._cfg.adaptive_gate_widen_factor

    def _det_feature(
        self, det: Detection,
    ) -> tuple[NDArray[np.float32] | None, bool]:
        """Appearance feature for matching + which space it lives in.

        Prefers the learned v4 embedding when ``reid_use_embedding`` is set
        and the detection carries one; otherwise falls back to the raw-patch
        signature so re-id keeps working through the classify-ring warm-up.
        The bool says ``is_embedding`` so the caller picks the matching cosine
        floor and compares against the matching track buffer.
        """
        if self._cfg.reid_use_embedding and det.embedding is not None:
            return unit_normalize(det.embedding), True
        if det.patch is not None:
            return normalize_patch(det.patch), False
        return None, False

    def _reid_threshold(self, *, is_embedding: bool) -> float:
        return (
            self._cfg.reid_embedding_threshold if is_embedding
            else self._cfg.reid_similarity_threshold
        )

    def _best_match(
        self,
        det_vec: NDArray[np.float32],
        candidates: list[_Track],
        *,
        is_embedding: bool,
    ) -> tuple[_Track | None, float]:
        """Most appearance-similar candidate + cosine, in the chosen space."""
        best_sim = -1.0
        best_track: _Track | None = None
        for track in candidates:
            buf = track.embedding if is_embedding else track.signature
            track_vec = buf.signature()
            if track_vec is None:
                continue
            sim = cosine_similarity(det_vec, track_vec)
            if sim > best_sim:
                best_sim = sim
                best_track = track
        return best_track, best_sim

    def _tiebreak_by_signature(
        self,
        det: Detection,
        gated: list[tuple[float, _Track]],
    ) -> _Track | None:
        """Appearance tiebreaker, only invoked when ≥2 tracks gate."""
        if not self._cfg.reid_enabled:
            return None
        det_vec, is_embedding = self._det_feature(det)
        if det_vec is None:
            return None
        best_track, _ = self._best_match(
            det_vec, [t for _, t in gated], is_embedding=is_embedding,
        )
        return best_track

    def _match_by_iou(
        self, det: Detection, candidates: list[_Track],
    ) -> _Track | None:
        metric = {"eiou": _eiou, "diou": _diou}.get(
            self._cfg.match_metric, _iou,
        )
        best_score = (
            self._cfg.match_eiou_threshold
            if self._cfg.match_metric in ("eiou", "diou")
            else self._cfg.match_iou_threshold
        )
        best_track: _Track | None = None
        det_box = (det.x0, det.y0, det.x1, det.y1)
        for track in candidates:
            score = metric(det_box, self._predicted_bbox(track))
            if score > best_score:
                best_score = score
                best_track = track
        return best_track

    def _match_by_signature(
        self, det: Detection, candidates: list[_Track],
    ) -> _Track | None:
        if not self._cfg.reid_enabled or not candidates:
            return None
        det_vec, is_embedding = self._det_feature(det)
        if det_vec is None:
            return None
        best_track, best_sim = self._best_match(
            det_vec, candidates, is_embedding=is_embedding,
        )
        threshold = self._reid_threshold(is_embedding=is_embedding)
        return best_track if best_sim > threshold else None

    def _predicted_bbox(
        self, track: _Track,
    ) -> tuple[float, float, float, float]:
        cx, cy = float(track.x[0]), float(track.x[1])
        return (
            cx - track.half_width, cy - track.half_height,
            cx + track.half_width, cy + track.half_height,
        )

    def _update_track_kalman(
        self, track: _Track, det: Detection, now_us: int,
    ) -> float:
        z = np.array([
            (det.x0 + det.x1) / 2.0,
            (det.y0 + det.y1) / 2.0,
            det.vx_px_s,
            det.vy_px_s,
        ])
        r = np.diag([
            self._cfg.obs_noise_pos_px ** 2,
            self._cfg.obs_noise_pos_px ** 2,
            self._cfg.obs_noise_vel_px_per_s ** 2,
            self._cfg.obs_noise_vel_px_per_s ** 2,
        ])
        innovation = z - track.x
        s = track.cov + r
        s_inv = np.linalg.inv(s)
        gain = track.cov @ s_inv
        track.x = track.x + gain @ innovation
        track.cov = (np.eye(4) - gain) @ track.cov
        mahal = float(innovation @ s_inv @ innovation)

        alpha = self._cfg.bbox_smoothing
        new_hw = (det.x1 - det.x0) / 2.0
        new_hh = (det.y1 - det.y0) / 2.0
        track.half_width = (1 - alpha) * track.half_width + alpha * new_hw
        track.half_height = (1 - alpha) * track.half_height + alpha * new_hh
        track.last_seen_us = now_us
        track.age_frames += 1
        track.miss_count = 0
        self._add_appearance(track, det)
        # F1 velocity-coherence accumulator: keep N most-recent speeds +
        # centroids so the confirmation gate sees a stable history.
        speed = math.hypot(det.vx_px_s, det.vy_px_s)
        track.speed_samples.append(speed)
        track.path_history.append((float(track.x[0]), float(track.x[1])))
        if len(track.speed_samples) > self._cfg.velocity_history_len:
            track.speed_samples.pop(0)
        if len(track.path_history) > self._cfg.velocity_history_len:
            track.path_history.pop(0)
        return mahal

    def _spawn(
        self, det: Detection, now_us: int, track_id: str | None = None,
    ) -> _Track:
        cx = (det.x0 + det.x1) / 2.0
        cy = (det.y0 + det.y1) / 2.0
        x = np.array([cx, cy, det.vx_px_s, det.vy_px_s])
        cov = np.diag([
            (2 * self._cfg.obs_noise_pos_px) ** 2,
            (2 * self._cfg.obs_noise_pos_px) ** 2,
            (2 * self._cfg.obs_noise_vel_px_per_s) ** 2,
            (2 * self._cfg.obs_noise_vel_px_per_s) ** 2,
        ])
        track = _Track(
            track_id=track_id or new_ulid(),
            x=x, cov=cov,
            half_width=(det.x1 - det.x0) / 2.0,
            half_height=(det.y1 - det.y0) / 2.0,
            last_seen_us=now_us,
            signature=SignatureBuffer(
                capacity=self._cfg.reid_signature_buffer,
            ),
            embedding=EmbeddingBuffer(
                capacity=self._cfg.reid_signature_buffer,
            ),
        )
        self._add_appearance(track, det)
        self._tracks[track.track_id] = track
        return track

    def _add_appearance(self, track: _Track, det: Detection) -> None:
        """Append this detection's appearance to the track's re-id buffers."""
        if not self._cfg.reid_enabled:
            return
        if det.patch is not None:
            track.signature.add(det.patch)
        if det.embedding is not None:
            track.embedding.add(det.embedding)

    def _resurrect_or_spawn(
        self, det: Detection, now_us: int,
    ) -> _Track | None:
        """Try to reuse a dormant track_id before spawning a fresh one.

        Returns None when the caller should skip (m_coh below spawn floor,
        or spawn-density gate suppresses this region).
        """
        if det.m_coh < self._cfg.spawn_min_mcoh:
            return None
        cx = (det.x0 + det.x1) / 2.0
        cy = (det.y0 + det.y1) / 2.0

        # Fix 2: spatial-only dormant resurrection (no appearance).
        if self._cfg.spatial_dormant_enabled and self._dormant:
            resurrected = self._match_dormant_spatial(det, now_us)
            if resurrected is not None:
                return resurrected

        # Appearance-based Re-ID: learned embedding when present, else patch.
        if self._cfg.reid_enabled and self._dormant:
            resurrected = self._match_dormant_appearance(det, now_us)
            if resurrected is not None:
                return resurrected

        # Fix 1: spawn-density gate suppresses noisy regions.
        if self._spawn_density.should_suppress(cx, cy, now_us):
            self._spawn_density_drops += 1
            return None

        track = self._spawn(det, now_us)
        self._spawn_density.register_spawn(cx, cy, now_us)
        return track

    def _match_dormant_spatial(
        self, det: Detection, now_us: int,
    ) -> _Track | None:
        """Resurrect a dormant track when motion extrapolation matches.

        For each dormant track, extrapolate its last Kalman state forward
        by (now - last_seen) and inflate the covariance by process noise
        over that interval. Mahalanobis distance from the detection to
        this extrapolated state, gated by ``spatial_dormant_chi2_cutoff``.
        The covariance inflation does the right thing automatically: a
        track that died right before the detection has a tight gate; one
        that died a second ago has a generous one.
        """
        timeout_us = int(self._cfg.spatial_dormant_timeout_s * 1_000_000)
        z = np.array([
            (det.x0 + det.x1) / 2.0, (det.y0 + det.y1) / 2.0,
            det.vx_px_s, det.vy_px_s,
        ])
        r = np.diag([
            self._cfg.obs_noise_pos_px ** 2,
            self._cfg.obs_noise_pos_px ** 2,
            self._cfg.obs_noise_vel_px_per_s ** 2,
            self._cfg.obs_noise_vel_px_per_s ** 2,
        ])
        best_tid: str | None = None
        best_dist = self._cfg.spatial_dormant_chi2_cutoff
        for tid, dormant in self._dormant.items():
            dt_us = now_us - dormant.last_seen_us
            if dt_us > timeout_us or dt_us < 0:
                continue
            dt = dt_us / 1_000_000.0
            f = np.array(
                [[1.0, 0.0, dt, 0.0],
                 [0.0, 1.0, 0.0, dt],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]],
            )
            x_pred = f @ dormant.last_state
            cov_pred = f @ dormant.last_cov @ f.T + self._process_noise(dt)
            innov = z - x_pred
            s = cov_pred + r
            try:
                s_inv = np.linalg.inv(s)
            except np.linalg.LinAlgError:
                continue
            dist = float(innov @ s_inv @ innov)
            if dist < best_dist:
                best_dist = dist
                best_tid = tid
        if best_tid is None:
            return None
        dormant = self._dormant.pop(best_tid)
        track = self._spawn(det, now_us, track_id=dormant.track_id)
        track.age_frames = max(
            track.age_frames, self._cfg.confirm_after_hits,
        )
        self._dormant_resurrections += 1
        return track

    def _match_dormant_appearance(
        self, det: Detection, now_us: int,
    ) -> _Track | None:
        """Resurrect a dormant track by appearance: embedding or patch."""
        det_vec, is_embedding = self._det_feature(det)
        if det_vec is None:
            return None
        timeout_us = int(self._cfg.reid_dormant_timeout_s * 1_000_000)
        best_sim = self._reid_threshold(is_embedding=is_embedding)
        best_tid: str | None = None
        for tid, dormant in self._dormant.items():
            dvec = dormant.embedding if is_embedding else dormant.signature
            if dvec is None or now_us - dormant.last_seen_us > timeout_us:
                continue
            sim = cosine_similarity(det_vec, dvec)
            if sim > best_sim:
                best_sim = sim
                best_tid = tid
        if best_tid is None:
            return None
        dormant = self._dormant.pop(best_tid)
        track = self._spawn(det, now_us, track_id=dormant.track_id)
        track.age_frames = max(
            track.age_frames, self._cfg.confirm_after_hits,
        )
        self._dormant_resurrections += 1
        return track

    def _evict_stale(self, now_us: int) -> None:
        # Active tracks: tentative ones (any miss before confirmation) die
        # immediately; confirmed ones past `max_miss_frames` go dormant if
        # either re-id mechanism is enabled, else just die.
        dormant_active = (
            self._cfg.reid_enabled or self._cfg.spatial_dormant_enabled
        )
        stale_active = [
            tid for tid, t in self._tracks.items()
            if t.miss_count > self._cfg.max_miss_frames
            or (
                t.miss_count > 0
                and t.age_frames < self._cfg.confirm_after_hits
            )
        ]
        for tid in stale_active:
            track = self._tracks.pop(tid)
            if (
                dormant_active
                and track.age_frames >= self._cfg.confirm_after_hits
            ):
                reid = self._cfg.reid_enabled
                self._dormant[tid] = _DormantTrack(
                    track_id=tid,
                    signature=track.signature.signature() if reid else None,
                    embedding=track.embedding.signature() if reid else None,
                    last_seen_us=track.last_seen_us,
                    last_bbox=(
                        int(track.x[0] - track.half_width),
                        int(track.x[1] - track.half_height),
                        int(track.x[0] + track.half_width),
                        int(track.x[1] + track.half_height),
                    ),
                    last_state=track.x.copy(),
                    last_cov=track.cov.copy(),
                    half_width=track.half_width,
                    half_height=track.half_height,
                )
        # Dormant tracks themselves age out — use the larger of any
        # enabled mechanism's timeout so neither gets prematurely cut.
        if self._dormant:
            candidates = []
            if self._cfg.reid_enabled:
                candidates.append(self._cfg.reid_dormant_timeout_s)
            if self._cfg.spatial_dormant_enabled:
                candidates.append(self._cfg.spatial_dormant_timeout_s)
            timeout_s = max(candidates) if candidates else 0.0
            timeout_us = int(timeout_s * 1_000_000)
            stale_dormant = [
                tid for tid, d in self._dormant.items()
                if now_us - d.last_seen_us > timeout_us
            ]
            for tid in stale_dormant:
                del self._dormant[tid]

    def _confidence_factor(self, mahal: float) -> float:
        """Mahalanobis → multiplier in [min, 1].

        Low residual ≈ Kalman thinks this matches → factor 1. Increasing
        residual smoothly decays the factor; at 3x chi-square critical
        value the factor is ~0.2 → clearly bad fit.
        """
        if not math.isfinite(mahal) or mahal < 0:
            return self._cfg.min_confidence_factor
        return max(
            self._cfg.min_confidence_factor,
            math.exp(-mahal / (2 * _CHI2_4DOF_95)),
        )


def _speed_stats(samples: list[float]) -> tuple[float, float]:
    """Mean speed and coefficient of variation over a speed ring."""
    speeds = np.asarray(samples, dtype=np.float64)
    mean_speed = float(speeds.mean())
    cv = float(speeds.std() / mean_speed) if mean_speed > 0 else float("inf")
    return mean_speed, cv


def _straightness(path: list[tuple[float, float]]) -> float:
    """Endpoint distance / path length in [0, 1]; 0 for a degenerate path."""
    if len(path) < 2:
        return 0.0
    p0 = np.asarray(path[0], dtype=np.float64)
    pn = np.asarray(path[-1], dtype=np.float64)
    endpoint_dist = float(np.linalg.norm(pn - p0))
    travelled = float(sum(
        np.linalg.norm(np.subtract(path[i], path[i - 1]))
        for i in range(1, len(path))
    ))
    return endpoint_dist / travelled if travelled > 0.0 else 0.0


def _passes_velocity_coherence(
    track: _Track, cfg: TrackManagerConfig,
) -> bool:
    """F1 velocity-coherence rule (replaces SORT's plain hit count)."""
    samples = track.speed_samples
    if len(samples) < cfg.confirm_after_hits:
        return False
    mean_speed, cv = _speed_stats(samples)
    if mean_speed < cfg.velocity_min_confirm_px_s:
        return False
    if cv > cfg.velocity_cv_max:
        return False
    straightness = _straightness(track.path_history)
    return straightness >= cfg.velocity_straightness_min


def _eiou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Efficient-IoU similarity in (-1, 1] (Zhang 2021, as a metric).

    IoU minus three penalties — centre distance, width mismatch, height
    mismatch — each normalised by the smallest enclosing box. Unlike plain
    IoU it degrades GRACEFULLY for the small/changing-size targets this
    tracker exists for: two 2-px far-drone boxes one pixel apart score
    IoU 0 (association death) but EIoU ~ -0.05 (still matchable), and an
    approaching drone whose box doubles between windows keeps credit for
    the centred position instead of being punished twice via the union.
    """
    iou = _iou(a, b)
    # smallest enclosing box
    cw = max(a[2], b[2]) - min(a[0], b[0])
    ch = max(a[3], b[3]) - min(a[1], b[1])
    if cw <= 0.0 or ch <= 0.0:
        return iou
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    centre = ((ax - bx) ** 2 + (ay - by) ** 2) / (cw**2 + ch**2)
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    size = (wa - wb) ** 2 / cw**2 + (ha - hb) ** 2 / ch**2
    return iou - centre - size


def _diou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Distance-IoU similarity in (-1, 1] — the ASSOCIATION variant.

    IoU minus the centre penalty ONLY. EIoU's extra width/height terms
    exist to sharpen bbox-REGRESSION gradients; as a matching metric they
    punish a centred, size-changing target (approaching drone: box doubles
    -> EIoU drops below plain IoU). DIoU keeps the graceful disjoint-box
    grading without that regression artifact.
    """
    iou = _iou(a, b)
    cw = max(a[2], b[2]) - min(a[0], b[0])
    ch = max(a[3], b[3]) - min(a[1], b[1])
    if cw <= 0.0 or ch <= 0.0:
        return iou
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    centre = ((ax - bx) ** 2 + (ay - by) ** 2) / (cw**2 + ch**2)
    return iou - centre


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    inter_x0 = max(a[0], b[0])
    inter_y0 = max(a[1], b[1])
    inter_x1 = min(a[2], b[2])
    inter_y1 = min(a[3], b[3])
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


__all__ = ["TrackManager", "TrackManagerConfig"]
