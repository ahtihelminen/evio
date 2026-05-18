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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sapient.messages import new_ulid

if TYPE_CHECKING:
    from evio.detection.types import Detection

# 4D chi-square 95th percentile ≈ 9.49; below this, fit is "good".
_CHI2_4DOF_95 = 9.49


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


@dataclass(slots=True)
class TrackManagerConfig:
    match_iou_threshold: float = 0.2
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


class TrackManager:
    """Per-camera Kalman tracker. Stateful across calls."""

    def __init__(self, config: TrackManagerConfig | None = None) -> None:
        self._cfg = config or TrackManagerConfig()
        self._tracks: dict[str, _Track] = {}
        self._last_us: int | None = None

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(
        self, detections: list[Detection], now_us: int,
    ) -> list[Detection]:
        dt = self._dt(now_us)
        self._predict_all(dt)
        matches, unmatched_dets, unmatched_tracks = self._match(detections)

        annotated: list[Detection] = []
        for det, track in matches:
            mahal = self._update_track_kalman(track, det, now_us)
            if track.age_frames < self._cfg.confirm_after_hits:
                continue  # tentative — wait for confirmation
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
            new_track = self._spawn(det, now_us)
            if new_track.age_frames >= self._cfg.confirm_after_hits:
                annotated.append(
                    replace(det, track_id=new_track.track_id),
                )

        for track in unmatched_tracks:
            track.miss_count += 1

        self._evict_stale()
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
            best_iou = self._cfg.match_iou_threshold
            best_track: _Track | None = None
            det_box = (det.x0, det.y0, det.x1, det.y1)
            for track in unmatched_tracks:
                track_box = self._predicted_bbox(track)
                iou = _iou(det_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is None:
                unmatched_dets.append(det)
            else:
                matches.append((det, best_track))
                unmatched_tracks.remove(best_track)
        return matches, unmatched_dets, unmatched_tracks

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
        return mahal

    def _spawn(self, det: Detection, now_us: int) -> _Track:
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
            track_id=new_ulid(),
            x=x, cov=cov,
            half_width=(det.x1 - det.x0) / 2.0,
            half_height=(det.y1 - det.y0) / 2.0,
            last_seen_us=now_us,
        )
        self._tracks[track.track_id] = track
        return track

    def _evict_stale(self) -> None:
        stale = [
            tid for tid, t in self._tracks.items()
            if t.miss_count > self._cfg.max_miss_frames
            or (
                t.miss_count > 0
                and t.age_frames < self._cfg.confirm_after_hits
            )
        ]
        for tid in stale:
            del self._tracks[tid]

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
