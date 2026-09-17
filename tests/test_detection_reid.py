"""Unit tests for the Re-ID primitives."""

from __future__ import annotations

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.reid import (
    SignatureBuffer,
    cosine_similarity,
    extract_event_patch,
    normalize_patch,
)

pytestmark = pytest.mark.t1


def _batch(n: int = 256) -> EventsBatch:
    rng = np.random.default_rng(0)
    return EventsBatch(
        timestamps=np.arange(n, dtype=np.int64) * 1000,
        x_coords=rng.integers(0, 128, n, dtype=np.int16),
        y_coords=rng.integers(0, 72, n, dtype=np.int16),
        polarities=rng.integers(0, 2, n, dtype=np.int8),
    )


def test_extract_returns_fixed_shape() -> None:
    patch = extract_event_patch(_batch(), bbox=(10, 10, 50, 50))
    assert patch.shape == (32, 32, 3)
    assert patch.dtype == np.uint8


def test_extract_handles_zero_size_bbox() -> None:
    patch = extract_event_patch(_batch(), bbox=(20, 20, 20, 20))
    assert patch.shape == (32, 32, 3)
    assert patch.sum() == 0


def test_normalize_returns_unit_vector() -> None:
    patch = extract_event_patch(_batch(), bbox=(0, 0, 64, 64))
    vec = normalize_patch(patch)
    assert vec.shape == (32 * 32,)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_normalize_all_zero_patch_is_zero_vector() -> None:
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    vec = normalize_patch(blank)
    assert np.allclose(vec, 0.0)


def test_cosine_self_similarity_is_one() -> None:
    patch = extract_event_patch(_batch(), bbox=(0, 0, 64, 64))
    vec = normalize_patch(patch)
    assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)


def test_cosine_shape_mismatch_raises() -> None:
    a = np.zeros(32, dtype=np.float32)
    b = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity(a, b)


def test_signature_buffer_empty_returns_none() -> None:
    buf = SignatureBuffer()
    assert buf.signature() is None
    assert len(buf) == 0


def test_signature_buffer_mean_is_normalized() -> None:
    buf = SignatureBuffer(capacity=4)
    patch = extract_event_patch(_batch(), bbox=(0, 0, 64, 64))
    buf.add(patch)
    buf.add(patch)
    sig = buf.signature()
    assert sig is not None
    assert np.linalg.norm(sig) == pytest.approx(1.0, abs=1e-5)


def test_signature_buffer_drops_oldest() -> None:
    buf = SignatureBuffer(capacity=2)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        patch = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        buf.add(patch)
    assert len(buf) == 2


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        SignatureBuffer(capacity=0)
