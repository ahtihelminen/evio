"""Equivalence test for the fused feature-matrix numba kernel.

``_compute_feature_matrix`` used to run five independent whole-array
numpy passes (see ``basic_features.py``). This test pins the fused
``@njit`` replacement to the numpy reference implementation, which stays
in ``classes.py`` under `_compute_feature_matrix_reference` as the
readable spec the kernel must match bit-for-bit.

It also documents the SN-431 dead-branch bug rather than fixing it:
``cell_polarity_skewness`` reads ``neg_mask = events.polarities < 0``
(`basic_features.py:139`), so unsigned {0, 1} polarity input can never
populate ``neg_mask`` and the skewness column collapses to {0.0, 1.0}.
Only signed (+-1) input exercises the real branch. Both conventions are
covered here, and the kernel is required to reproduce the buggy {0, 1}
collapse exactly, not paper over it.
"""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.background.classes import (
    _compute_feature_matrix,
    _compute_feature_matrix_reference,
)
from evio.detection.grid import GridConfig

pytestmark = pytest.mark.t1

_GRID = GridConfig(
    width=97, height=71, rows=7, cols=9, time_bins=1, window_us=10_000
)


def _make_events(
    rng: np.random.Generator, n_events: int, *, signed_polarity: bool
) -> EventsBatch:
    x_coords = rng.integers(0, _GRID.width, size=n_events, dtype=np.int16)
    y_coords = rng.integers(0, _GRID.height, size=n_events, dtype=np.int16)
    timestamps = np.sort(
        rng.integers(0, 10_000, size=n_events).astype(np.int64)
    )
    if signed_polarity:
        polarities = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=n_events
        )
    else:
        polarities = rng.choice(
            np.array([0, 1], dtype=np.int8), size=n_events
        )
    return EventsBatch(
        timestamps=timestamps,
        x_coords=x_coords,
        y_coords=y_coords,
        polarities=polarities,
    )


@pytest.mark.parametrize("n_events", [2_000, 10_000, 30_000])
@pytest.mark.parametrize("signed_polarity", [True, False])
def test_kernel_matches_numpy_reference_bit_for_bit(
    n_events: int, signed_polarity: bool
) -> None:
    rng = np.random.default_rng(seed=(n_events, signed_polarity))
    events = _make_events(rng, n_events, signed_polarity=signed_polarity)

    kernel_matrix = _compute_feature_matrix(events, _GRID)
    reference_matrix = _compute_feature_matrix_reference(events, _GRID)

    assert kernel_matrix.shape == (_GRID.cells, 5)
    assert kernel_matrix.dtype == reference_matrix.dtype
    max_abs_diff = np.max(np.abs(kernel_matrix - reference_matrix))
    assert max_abs_diff == 0.0


def test_unsigned_polarity_input_hits_the_dead_neg_mask_branch() -> None:
    """Documents SN-431: {0, 1} input can never populate `neg_mask`.

    With unsigned input every active cell's skewness collapses to
    exactly 1.0 (no cell ever sees a negative polarity), so this branch
    of the equivalence test above passes trivially -- it is the signed
    variant that actually exercises `neg_mask`.
    """
    rng = np.random.default_rng(seed=1234)
    events = _make_events(rng, 10_000, signed_polarity=False)

    reference_matrix = _compute_feature_matrix_reference(events, _GRID)
    skewness = reference_matrix[:, 4]
    active = reference_matrix[:, 0] > 0

    assert np.all(np.isin(skewness, [0.0, 1.0]))
    assert np.all(skewness[active] == 1.0)


def test_signed_polarity_input_exercises_the_real_neg_mask_branch() -> None:
    """Signed (+-1) input is what actually reaches `neg_mask`."""
    rng = np.random.default_rng(seed=5678)
    events = _make_events(rng, 10_000, signed_polarity=True)

    reference_matrix = _compute_feature_matrix_reference(events, _GRID)
    skewness = reference_matrix[:, 4]

    assert np.any(~np.isin(skewness, [0.0, 1.0]))
