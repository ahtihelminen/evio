"""IRSTD-style local-contrast re-scorer on the event-density patch (P5).

MPCM (multiscale patch-based contrast measure, Wei 2016) adapted to the
32x32 ReID event patch: the polarity-MAGNITUDE map ``|gray - 127| / 128``
counts ON and OFF energy alike (no polarity cancellation), then each cell's
contrast is the minimum over opposite-neighbour difference *products* — a
compact blob exceeds its background on all sides so every product is
positive, while an edge (cloud boundary, tree line) exceeds on one side only
and zeroes out. Division-free, so sparse all-baseline neighbourhoods can't
blow up the way LCM's background-mean ratio does.

The four ``IRSTD_FEATURES`` feed a :class:`~evio.detection.kinematic_llr.
KinematicLLR` instance (the generic standardised-logistic arm) fit by
``scripts/train_evidence_llrs.py``; the arm rides the fusion bus additively
and never gates detection/tracking — which is what makes the
``single_pixel_PD`` non-regression argument structural.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SCALES: tuple[int, ...] = (2, 4, 8)  # cell edge, px, on the 32x32 patch
_BORDER_PX = 4

IRSTD_FEATURES: tuple[str, ...] = (
    "irstd_mpcm",         # max multiscale patch contrast (edge-suppressing)
    "irstd_log_energy",   # log1p(total event energy in the patch)
    "irstd_fill",         # fraction of pixels carrying any energy
    "irstd_border_frac",  # energy share in the outer ring (clutter leakage)
)


def patch_energy_map(patch: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BGR event patch → polarity-magnitude map in [0, 1].

    Grey 127 = no events; ON (255) and OFF (0) both map to high energy, so a
    mixed-polarity rotor/track signature is energy, not cancellation.
    """
    grey = patch.mean(axis=2) if patch.ndim == 3 else patch.astype(np.float32)
    return (np.abs(grey.astype(np.float32) - 127.0) / 128.0).astype(
        np.float32,
    )


def _cell_means(
    energy: NDArray[np.float32], scale: int,
) -> NDArray[np.float32]:
    """Non-overlapping ``scale``x``scale`` cell means over the patch."""
    h, w = energy.shape
    ch, cw = h // scale, w // scale
    trimmed = energy[: ch * scale, : cw * scale]
    return trimmed.reshape(ch, scale, cw, scale).mean(axis=(1, 3))


# Neighbour offsets ordered so that i and i+4 are opposite pairs.
_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1), (0, 1),
    (1, 1), (1, 0), (1, -1), (0, -1),
)


def mpcm_score(
    energy: NDArray[np.float32], *, scales: tuple[int, ...] = _SCALES,
) -> tuple[float, int]:
    """Division-free multiscale patch contrast → (max score, argmax scale).

    Per interior cell: ``d_i = centre - neighbour_i``; the positional score is
    ``min over the 4 opposite pairs of (d_i · d_{i+4})`` clipped at 0.
    """
    best = 0.0
    best_scale = scales[0]
    for scale in scales:
        cells = _cell_means(energy, scale)
        if cells.shape[0] < 3 or cells.shape[1] < 3:
            continue
        centre = cells[1:-1, 1:-1]
        diffs = [
            centre - cells[1 + dy: cells.shape[0] - 1 + dy,
                           1 + dx: cells.shape[1] - 1 + dx]
            for dy, dx in _NEIGHBOURS
        ]
        pair_products = np.stack(
            [diffs[i] * diffs[i + 4] for i in range(4)], axis=0,
        )
        positional = np.clip(pair_products.min(axis=0), 0.0, None)
        peak = float(positional.max()) if positional.size else 0.0
        if peak > best:
            best = peak
            best_scale = scale
    return best, best_scale


def irstd_features(
    patch: NDArray[np.uint8] | None,
) -> dict[str, float]:
    """The 4-feature dict for the IRSTD LLR arm; ``{}`` when patch is None."""
    if patch is None:
        return {}
    energy = patch_energy_map(patch)
    total = float(energy.sum())
    mpcm, _scale = mpcm_score(energy)
    border = energy.copy()
    border[_BORDER_PX:-_BORDER_PX, _BORDER_PX:-_BORDER_PX] = 0.0
    return {
        "irstd_mpcm": mpcm,
        "irstd_log_energy": float(np.log1p(total)),
        "irstd_fill": float((energy > 1e-3).mean()),
        "irstd_border_frac": float(border.sum()) / max(total, 1e-6),
    }


__all__ = [
    "IRSTD_FEATURES",
    "irstd_features",
    "mpcm_score",
    "patch_energy_map",
]
