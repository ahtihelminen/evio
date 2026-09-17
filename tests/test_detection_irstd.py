"""IRSTD MPCM re-scorer: edge suppression, polarity handling, features."""

from __future__ import annotations

import time

import numpy as np
import pytest

from evio.detection.irstd import (
    IRSTD_FEATURES,
    irstd_features,
    mpcm_score,
    patch_energy_map,
)

pytestmark = pytest.mark.t1


def _baseline_patch() -> np.ndarray:
    return np.full((32, 32, 3), 127, dtype=np.uint8)


def _blob_patch(size: int = 5, value: int = 255) -> np.ndarray:
    patch = _baseline_patch()
    lo = 16 - size // 2
    patch[lo: lo + size, lo: lo + size] = value
    return patch


def _edge_patch() -> np.ndarray:
    """Half-plane edge with the same bright value as the blob."""
    patch = _baseline_patch()
    patch[:, 16:] = 255
    return patch


def test_blob_beats_edge() -> None:
    """Edge suppression is the load-bearing MPCM property."""
    blob = irstd_features(_blob_patch())["irstd_mpcm"]
    edge = irstd_features(_edge_patch())["irstd_mpcm"]
    assert blob > 0.05
    assert edge < blob * 0.2


def test_mixed_polarity_target_scores() -> None:
    """Adjacent ON/OFF pixels are ENERGY (magnitude map, no cancellation).

    The target sits inside one scale-4 cell (16:20) — MPCM needs the blob
    within a cell, and this test isolates polarity handling from that.
    """
    patch = _baseline_patch()
    patch[16:20, 16:18] = 255  # ON half
    patch[16:20, 18:20] = 0    # OFF half
    feats = irstd_features(patch)
    assert feats["irstd_mpcm"] > 0.05
    assert feats["irstd_log_energy"] > 0.0


def test_all_baseline_patch_is_neutral_and_finite() -> None:
    feats = irstd_features(_baseline_patch())
    assert set(feats) == set(IRSTD_FEATURES)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["irstd_mpcm"] == 0.0
    assert feats["irstd_fill"] == 0.0


def test_none_patch_returns_empty() -> None:
    assert irstd_features(None) == {}


def test_blob_beats_salt_noise_at_equal_energy() -> None:
    blob = _blob_patch(size=5)
    n_on = int((blob != 127).sum() // 3)
    rng = np.random.default_rng(0)
    salt = _baseline_patch()
    ys, xs = np.unravel_index(
        rng.choice(32 * 32, size=n_on, replace=False), (32, 32),
    )
    salt[ys, xs] = 255
    assert (
        irstd_features(blob)["irstd_mpcm"]
        > irstd_features(salt)["irstd_mpcm"]
    )


def test_border_frac_flags_clutter_leakage() -> None:
    centered = irstd_features(_blob_patch())["irstd_border_frac"]
    corner = _baseline_patch()
    corner[0:5, 0:5] = 255
    cornered = irstd_features(corner)["irstd_border_frac"]
    assert cornered > centered


def test_energy_map_polarity_symmetry() -> None:
    on = _blob_patch(value=255)
    off = _blob_patch(value=0)
    assert np.allclose(patch_energy_map(on), patch_energy_map(off), atol=0.01)


def test_single_pixel_target_scores_at_fine_scale() -> None:
    """A 1-2 px far target must not be invisible to the scorer."""
    patch = _baseline_patch()
    patch[16, 16] = 255
    patch[16, 17] = 255
    mpcm, scale = mpcm_score(patch_energy_map(patch))
    assert mpcm > 0.0
    assert scale == 2  # found at the finest cell scale


def test_cost_budget() -> None:
    patch = _blob_patch()
    irstd_features(patch)  # warm
    t0 = time.perf_counter()
    for _ in range(100):
        irstd_features(patch)
    per_call_us = (time.perf_counter() - t0) * 1e4
    assert per_call_us < 500.0  # loose guard against O(n^2) rewrites
