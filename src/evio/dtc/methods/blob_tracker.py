"""Track blob centres and rank tracks by the evidence behind them.

Association is in the image plane (a blob near where a track is predicted to
be), which costs a few distance checks per step. A track's score is how
strongly it was seen over the last few steps, so a target that is the best
blob in some steps and merely a good one in others still outranks a clutter
blob that flashes once. Positions and velocities are in pixels per step; the
kinematic features are in pixels per second.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

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

_Z_CAP = 30.0


@dataclass(frozen=True)
class BlobTrackerConfig:
    """`gains` are the (position, velocity) gains of the residual."""

    confirm_hits: int = 4
    window: int = 6
    max_misses: int = 6
    coast_steps: int = 3
    gate_px: float = 30.0
    spawn_z: float = 8.0
    gains: tuple[float, float] = (0.6, 0.25)
    max_tracks: int = 30
    reacquire_steps: int = 50
    reacquire_speed_px: float = 12.0
    reacquire_hits: int = 4
    miss_gate_px: float = 10.0
    merge_px: float = 30.0
    confirmed_first: bool = True
    velocity_decay: float = 1.0
    history_len: int = 8


@dataclass(slots=True)
class _Track:
    track_id: str
    centre: np.ndarray
    velocity: np.ndarray  # pixels per step
    size: np.ndarray
    seen: deque[float]  # z per step, 0 when missed
    path: deque[np.ndarray]  # recent centres
    anchor: np.ndarray  # where a blob last followed it
    anchor_step: int
    misses: int = 0
    observations: int = 1
    needs: int = 4  # hits in the window before it is shown
    shown: bool = False
    born: int = 0
    parent: _Track | None = None  # the lost track this one stands in for

    @property
    def predicted(self) -> np.ndarray:
        return self.centre + self.velocity

    @property
    def hits(self) -> int:
        return sum(z > 0 for z in self.seen)

    @property
    def score(self) -> float:
        return sum(self.seen) / self.seen.maxlen

    def box_at(self, centre: np.ndarray) -> Box:
        (x, y), (w, h) = centre, self.size
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def features(self, dt_s: float) -> dict[str, float]:
        """The `TrackManager.track_features` subset a raster tracker has."""
        hops = [math.dist(a, b) for a, b in pairwise(self.path)]
        speeds = np.array(hops or [0.0]) / dt_s
        mean_speed = float(speeds.mean())
        travelled = sum(hops)
        return {
            "speed_mean": mean_speed,
            "speed_cv": float(speeds.std() / mean_speed) if mean_speed else 0.0,
            "straightness": (
                math.dist(self.path[0], self.path[-1]) / travelled
                if travelled
                else 0.0
            ),
            "vel_mag": math.hypot(*self.velocity) / dt_s,
            "age_frames": float(self.observations),
            "miss_count": float(self.misses),
            "bbox_diag": math.hypot(*self.size),
        }


@method(
    "blob-tracker",
    stages=(Stage.TRACK,),
    config=BlobTrackerConfig,
)
class BlobTracker:
    """Image-plane tracks of blobs, ranked by recent z, with re-acquisition."""

    def __init__(self, config: BlobTrackerConfig, context: Context) -> None:
        self.config = config
        self.stream = str(context.stream)
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._lost: list[_Track] = []
        self._created = 0
        self._step = 0

    def step(self, step: Step, candidates: Candidates) -> Tracks:
        """Take this step's blobs; the confirmed tracks, best first."""
        config = self.config
        blobs = candidates.items
        dt_s = step.duration_us / 1e6
        self._step += 1
        followed, used = self._associate(blobs)
        for track in self._tracks:
            if id(track) not in followed:
                track.centre = track.predicted
                track.velocity = track.velocity * config.velocity_decay
                track.seen.append(0.0)
                track.path.append(track.centre)
                track.misses += 1
        self._retire()
        self._merge_duplicates()
        self._spawn(blobs, used)
        shown = [
            t
            for t in self._tracks
            if t.hits >= t.needs and t.misses <= config.coast_steps
        ]
        for track in shown:
            track.shown = True
        shown.sort(key=lambda track: -track.score)
        return Tracks(
            tuple(self._emit(track, dt_s) for track in shown),
            frozenset(self._key(track) for track in self._tracks),
            tuple(
                Region(track.box_at(track.predicted)) for track in self._tracks
            ),
        )

    def _key(self, track: _Track) -> str:
        return f"{self.stream}-{track.track_id}"

    def _emit(self, track: _Track, dt_s: float) -> Track:
        vx, vy = track.velocity / dt_s
        return Track(
            self.stream,
            self._key(track),
            Region(track.box_at(track.centre), velocity_px_s=(vx, vy)),
            track.score,
            track.features(dt_s),
        )

    def _retire(self) -> None:
        """Drop tracks that missed too long; remember the ones that showed."""
        config = self.config
        alive = []
        for track in self._tracks:
            if track.misses <= config.max_misses:
                alive.append(track)
            elif track.shown and config.reacquire_steps:
                self._lost.append(track)
            elif track.parent is not None:
                self._lost.append(track.parent)
        self._tracks = alive
        self._lost = [
            t
            for t in self._lost
            if self._step - t.anchor_step <= config.reacquire_steps
        ]

    def _merge_duplicates(self) -> None:
        """Where two tracks sit on one spot, the older one stays."""
        if not self.config.merge_px:
            return
        kept: list[_Track] = []
        for track in sorted(self._tracks, key=lambda t: t.born):
            if all(
                math.dist(track.centre, other.centre) > self.config.merge_px
                for other in kept
            ):
                kept.append(track)
        self._tracks = kept

    def _associate(
        self, blobs: tuple[Candidate, ...]
    ) -> tuple[set[int], set[int]]:
        """Greedy nearest-first pairing of tracks and blobs inside the gate.

        Returns the ids of the tracks that followed a blob, and the indices
        of the blobs they followed.
        """
        config = self.config
        pairs = sorted(
            (
                config.confirmed_first and not t.shown,
                math.dist(blob.region.center, t.predicted),
                i,
                j,
            )
            for i, t in enumerate(self._tracks)
            for j, blob in enumerate(blobs)
        )
        followed: set[int] = set()
        used: set[int] = set()
        for _, distance, i, j in pairs:
            track = self._tracks[i]
            gate = (
                config.gate_px
                + config.miss_gate_px * track.misses
                + math.hypot(*track.velocity)
            )
            if id(track) in followed or j in used or distance > gate:
                continue
            followed.add(id(track))
            used.add(j)
            self._follow(track, blobs[j])
        return followed, used

    def _follow(self, track: _Track, blob: Candidate) -> None:
        gain_position, gain_velocity = self.config.gains
        centre = np.array(blob.region.center)
        residual = centre - track.predicted
        track.centre = track.predicted + gain_position * residual
        track.velocity = track.velocity + gain_velocity * residual
        track.size = 0.7 * track.size + 0.3 * _size(blob)
        track.seen.append(min(blob.score, _Z_CAP))
        track.path.append(track.centre)
        track.misses = 0
        track.observations += 1
        track.anchor = centre
        track.anchor_step = self._step

    def _spawn(self, blobs: tuple[Candidate, ...], used: set[int]) -> None:
        config = self.config
        free = [
            j
            for j, blob in enumerate(blobs)
            if j not in used and blob.score >= config.spawn_z
        ]
        for lost in sorted(self._lost, key=lambda t: -t.anchor_step):
            j = self._nearest_in_reach(lost, blobs, free)
            if j is not None:
                free.remove(j)
                self._lost.remove(lost)
                self._add(
                    blobs[j],
                    lost.track_id,
                    config.reacquire_hits,
                    parent=lost,
                )
        for j in free:
            self._created += 1
            self._add(blobs[j], f"t{self._created}", config.confirm_hits)

    def _nearest_in_reach(
        self, lost: _Track, blobs: tuple[Candidate, ...], free: list[int]
    ) -> int | None:
        """The free blob closest to where a lost track may have gone."""
        gap = self._step - lost.anchor_step
        reach = self.config.gate_px + self.config.reacquire_speed_px * gap
        near = {j: math.dist(blobs[j].region.center, lost.anchor) for j in free}
        near = {j: d for j, d in near.items() if d <= reach}
        return min(near, key=near.get) if near else None

    def _add(
        self,
        blob: Candidate,
        track_id: str,
        needs: int,
        parent: _Track | None = None,
    ) -> None:
        config = self.config
        if len(self._tracks) >= config.max_tracks:
            self._tracks.remove(min(self._tracks, key=lambda t: t.score))
        centre = np.array(blob.region.center)
        self._tracks.append(
            _Track(
                track_id,
                centre,
                np.zeros(2),
                _size(blob),
                deque([min(blob.score, _Z_CAP)], maxlen=config.window),
                deque([centre], maxlen=config.history_len),
                centre,
                self._step,
                needs=needs,
                born=self._step,
                parent=parent,
            )
        )


def _size(blob: Candidate) -> np.ndarray:
    x0, y0, x1, y1 = blob.region.box
    return np.array([x1 - x0, y1 - y0])
