"""Unit tests for the temporal sequence transformer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from evio.detection.temporal_transformer import (
    TemporalTrackTransformer,
    TemporalTransformerBundle,
    TemporalTransformerConfig,
    count_parameters,
    load_temporal_transformer,
)


def _rand_uint8(
    batch: int, seq: int, size: int = 32, seed: int = 0,
    in_channels: int = 3,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.integers(
            0, 256, (batch, seq, size, size, in_channels), dtype=np.uint8,
        ),
    )


def test_forward_shape_and_dtype() -> None:
    model = TemporalTrackTransformer(TemporalTransformerConfig()).eval()
    x = _rand_uint8(batch=2, seq=16)
    out = model(x)
    assert out["drone_logits"].shape == (2, 16)
    assert out["embeddings"].shape == (2, 16, 128)
    assert out["validity"].shape == (2,)
    assert all(v.dtype == torch.float32 for v in out.values())


def test_forward_accepts_uint8_or_float32() -> None:
    """Same patches as uint8 vs float32 in [0, 1] produce identical outputs."""
    model = TemporalTrackTransformer(TemporalTransformerConfig()).eval()
    x_u8 = _rand_uint8(batch=2, seq=16)
    x_f32 = x_u8.to(torch.float32) / 255.0
    with torch.no_grad():
        out_u8 = model(x_u8)
        out_f32 = model(x_f32)
    for key in out_u8:
        assert torch.allclose(out_u8[key], out_f32[key], atol=1e-5)


def test_embeddings_are_l2_unit_norm() -> None:
    model = TemporalTrackTransformer(TemporalTransformerConfig()).eval()
    with torch.no_grad():
        out = model(_rand_uint8(batch=3, seq=16))
    norms = torch.linalg.norm(out["embeddings"], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_validity_sigmoid_range() -> None:
    model = TemporalTrackTransformer(TemporalTransformerConfig()).eval()
    with torch.no_grad():
        out = model(_rand_uint8(batch=4, seq=16))
    probs = torch.sigmoid(out["validity"])
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_gradient_sanity() -> None:
    """Combined three-head loss back-propagates without NaN grads."""
    torch.manual_seed(0)
    config = TemporalTransformerConfig()
    model = TemporalTrackTransformer(config)
    x = _rand_uint8(batch=2, seq=16, seed=1)
    out = model(x)
    drone_targets = torch.zeros_like(out["drone_logits"])
    validity_targets = torch.ones_like(out["validity"])
    embedding_targets = torch.nn.functional.normalize(
        torch.randn_like(out["embeddings"]), dim=-1,
    )
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            out["drone_logits"], drone_targets,
        )
        + 0.5 * torch.nn.functional.mse_loss(
            out["embeddings"], embedding_targets,
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            out["validity"], validity_targets,
        )
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameters received gradients"
    assert all(torch.isfinite(g).all() for g in grads)


def test_save_load_state_dict_roundtrip(tmp_path: Path) -> None:
    """A model loaded from state_dict produces identical eval outputs."""
    torch.manual_seed(0)
    config = TemporalTransformerConfig()
    model = TemporalTrackTransformer(config).eval()
    x = _rand_uint8(batch=2, seq=16, seed=2)
    with torch.no_grad():
        before = model(x)
    ckpt = tmp_path / "tt.pt"
    torch.save(model.state_dict(), ckpt)
    fresh = TemporalTrackTransformer(config).eval()
    fresh.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True),
    )
    with torch.no_grad():
        after = fresh(x)
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-6)


def test_load_temporal_transformer_via_bundle(tmp_path: Path) -> None:
    """`load_temporal_transformer` reads meta JSON and matches model(x)."""
    torch.manual_seed(0)
    config = TemporalTransformerConfig()
    model = TemporalTrackTransformer(config).eval()
    ckpt = tmp_path / "tt.pt"
    torch.save(model.state_dict(), ckpt)
    (tmp_path / "tt_meta.json").write_text(json.dumps({
        "config": {
            "seq_len": config.seq_len,
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "embed_dim": config.embed_dim,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "ffn_dim": config.ffn_dim,
            "embedding_dim": config.embedding_dim,
            "dropout": config.dropout,
        },
    }))

    bundle = load_temporal_transformer(ckpt, device=torch.device("cpu"))
    assert isinstance(bundle, TemporalTransformerBundle)
    assert bundle.config == config

    patches = _rand_uint8(batch=2, seq=16, seed=3).numpy()
    via_bundle = bundle.predict(patches)
    with torch.no_grad():
        via_model = model(torch.from_numpy(patches))
    for key in via_bundle:
        assert np.allclose(via_bundle[key], via_model[key].numpy(), atol=1e-5)


def test_param_count_under_1m() -> None:
    model = TemporalTrackTransformer(TemporalTransformerConfig())
    n_params = count_parameters(model)
    assert n_params < 1_000_000, f"param budget exceeded: {n_params}"


@pytest.mark.parametrize("seq_len", [8, 32])
def test_seq_len_variation(seq_len: int) -> None:
    """The model works at construction-time T values other than 16."""
    config = TemporalTransformerConfig(seq_len=seq_len)
    model = TemporalTrackTransformer(config).eval()
    x = _rand_uint8(batch=2, seq=seq_len, seed=seq_len)
    with torch.no_grad():
        out = model(x)
    assert out["drone_logits"].shape == (2, seq_len)
    assert out["embeddings"].shape == (2, seq_len, config.embedding_dim)
    assert out["validity"].shape == (2,)


@pytest.mark.parametrize("in_channels", [3, 16])
def test_in_channels_variation(in_channels: int) -> None:
    """Voxel-grid input (in_channels=16) is wire-compatible."""
    config = TemporalTransformerConfig(in_channels=in_channels)
    model = TemporalTrackTransformer(config).eval()
    x = _rand_uint8(batch=2, seq=16, in_channels=in_channels, seed=in_channels)
    with torch.no_grad():
        out = model(x)
    assert out["drone_logits"].shape == (2, 16)
    assert out["embeddings"].shape == (2, 16, config.embedding_dim)
    assert out["validity"].shape == (2,)


def test_wrong_channel_count_raises() -> None:
    model = TemporalTrackTransformer(
        TemporalTransformerConfig(in_channels=3),
    ).eval()
    x = _rand_uint8(batch=1, seq=4, in_channels=16)
    with pytest.raises(ValueError, match="expected"):
        model(x)
