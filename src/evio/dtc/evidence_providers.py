"""The built-in evidence providers (see `evio.dtc.evidence`).

Every provider reads the window of the stream a track belongs to and returns
one value per track id. A point-only region gets a small fixed box around
the point. Nothing here reads a clock, and the heavy renderer behind the
patch ring is imported when first needed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from math import ceil, floor

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.dtc.evidence import evidence_provider
from evio.dtc.protocols import Context
from evio.dtc.types import Box, Region, Step, Track, Tracks, Window

ROI_MARGIN_PX = 4
POINT_HALF_SIZE_PX = 8
RING_LENGTH = 16
PATCH_SIZE = (32, 32)
SERIES_BIN_US = 500


def track_box(region: Region) -> Box:
    """The region's box, or a small box around its point."""
    if region.box is not None:
        return region.box
    x, y = region.center
    half = POINT_HALF_SIZE_PX
    return (x - half, y - half, x + half, y + half)


def events_in_box(window: Window, box: Box, margin: int) -> EventsBatch:
    """A copy of the window's events inside `box` grown by `margin`."""
    x0, y0, x1, y1 = box
    batch = window.events
    mask = (
        (batch.x_coords >= x0 - margin)
        & (batch.x_coords <= x1 + margin)
        & (batch.y_coords >= y0 - margin)
        & (batch.y_coords <= y1 + margin)
    )
    return EventsBatch(
        timestamps=batch.timestamps[mask],
        x_coords=batch.x_coords[mask],
        y_coords=batch.y_coords[mask],
        polarities=batch.polarities[mask],
    )


def track_events(step: Step, track: Track) -> EventsBatch:
    window = step.windows[track.stream]
    return events_in_box(window, track_box(track.region), ROI_MARGIN_PX)


class _Stateless:
    def __init__(self, context: Context) -> None:
        pass

    def reset(self) -> None:
        pass


@evidence_provider("kinematics")
class Kinematics(_Stateless):
    """Each track's feature dict, copied."""

    def compute(self, step: Step, tracks: Tracks) -> Mapping[str, object]:
        return {t.id: dict(t.features) for t in tracks.items}


@evidence_provider("roi_events")
class RoiEvents(_Stateless):
    """The window's events in each track's box plus a margin."""

    def compute(self, step: Step, tracks: Tracks) -> Mapping[str, object]:
        return {t.id: track_events(step, t) for t in tracks.items}


@evidence_provider("event_series")
class EventSeries(_Stateless):
    """Polarity-split event counts per 0.5 ms bin over the step.

    Bins start at the window start, so a track's series lines up with its
    `t_start_us` whatever the step length.
    """

    def compute(self, step: Step, tracks: Tracks) -> Mapping[str, object]:
        return {t.id: self._series(step, t) for t in tracks.items}

    @staticmethod
    def _series(step: Step, track: Track) -> dict[str, object]:
        window = step.windows[track.stream]
        batch = track_events(step, track)
        n_bins = ceil((window.t_end_us - window.t_start_us) / SERIES_BIN_US)
        bins = (batch.timestamps - window.t_start_us) // SERIES_BIN_US
        inside = (bins >= 0) & (bins < n_bins)
        on = batch.polarities > 0
        return {
            "t_start_us": window.t_start_us,
            "bin_us": SERIES_BIN_US,
            "on": np.bincount(bins[inside & on], minlength=n_bins),
            "off": np.bincount(bins[inside & ~on], minlength=n_bins),
        }


@evidence_provider("patch_ring")
class PatchRing:
    """The last 16 event-density patches per track, oldest first.

    A track's value is a (16, 32, 32, 3) uint8 array, present only once the
    ring is full. Rings of tracks no longer live are dropped.
    """

    def __init__(self, context: Context) -> None:
        self._rings: dict[str, deque[NDArray[np.uint8]]] = {}

    def compute(self, step: Step, tracks: Tracks) -> Mapping[str, object]:
        from evio.detection.reid import extract_event_patch

        self._rings = {
            id_: ring
            for id_, ring in self._rings.items()
            if id_ in tracks.live_ids
        }
        for track in tracks.items:
            x0, y0, x1, y1 = track_box(track.region)
            patch = extract_event_patch(
                step.windows[track.stream].events,
                (floor(x0), floor(y0), ceil(x1), ceil(y1)),
                PATCH_SIZE,
            )
            ring = self._rings.setdefault(track.id, deque(maxlen=RING_LENGTH))
            ring.append(patch)
        return {
            t.id: np.stack(self._rings[t.id])
            for t in tracks.items
            if len(self._rings[t.id]) == RING_LENGTH
        }

    def reset(self) -> None:
        self._rings.clear()
