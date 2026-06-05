"""v4 streaming/causal temporal transformer tests.

Covers:
* `config.causal=True` builds a `_StreamingEncoder` whose state dict
  is compatible with the bidirectional `nn.TransformerEncoder`.
* The causal mask blocks future positions and gives different drone
  logits than the bidirectional model on identical input.
* The static (L, L) mask is shared across the batch.
* The K/V cache produces equivalent output to the full forward at
  every non-CLS position (CLS staleness is the documented streaming
  semantic and doesn't affect the drone head).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from evio.detection.temporal_transformer_v3 import (
    TemporalTrackTransformerV3,
    TemporalTransformerV3Config,
    _StreamingEncoder,
)


def _rand_inputs(
    batch: int, seq: int, e_max: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    patches = torch.from_numpy(
        rng.integers(0, 256, (batch, seq, 32, 32, 3), dtype=np.uint8),
    )
    events = torch.from_numpy(
        (rng.random((batch, e_max, 4), dtype=np.float32) * 2 - 1)
        .astype(np.float32),
    )
    mask = torch.from_numpy(rng.random((batch, e_max)) > 0.3)
    return patches, events, mask


def test_streaming_encoder_matches_nn_transformer_encoder() -> None:
    """Without mask: _StreamingEncoder is math-equivalent to PyTorch's."""
    torch.manual_seed(0)
    d, h, ffn, n = 128, 8, 512, 6
    ref = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=ffn, dropout=0.0,
            batch_first=True, norm_first=True,
        ),
        num_layers=n,
    ).eval()
    streaming = _StreamingEncoder(d, h, ffn, 0.0, n).eval()
    streaming.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(4, 256, d)
    with torch.no_grad():
        ref_out = ref(x)
        stream_out, _ = streaming(x)
    assert (ref_out - stream_out).abs().max() < 1e-4


def test_causal_config_produces_different_output() -> None:
    """Same weights, causal mask → different drone_logits."""
    torch.manual_seed(0)
    cfg_bi = TemporalTransformerV3Config(seq_len=8, e_max=64, n_layers=2)
    cfg_c = TemporalTransformerV3Config(
        seq_len=8, e_max=64, n_layers=2, causal=True,
    )
    m_bi = TemporalTrackTransformerV3(cfg_bi).eval()
    m_c = TemporalTrackTransformerV3(cfg_c).eval()
    m_c.load_state_dict(m_bi.state_dict(), strict=True)
    patches, events, mask = _rand_inputs(batch=2, seq=8, e_max=64)
    with torch.no_grad():
        o_bi = m_bi(patches, events, mask)
        o_c = m_c(patches, events, mask)
    diff = (o_bi["drone_logits"] - o_c["drone_logits"]).abs().max()
    assert diff > 1e-3, f"causal mask must change output: {diff}"


def test_causal_state_dict_compatible_with_bidirectional() -> None:
    """A causal model loads bidirectional weights with strict=True."""
    cfg_bi = TemporalTransformerV3Config(seq_len=8, e_max=64, n_layers=2)
    cfg_c = TemporalTransformerV3Config(
        seq_len=8, e_max=64, n_layers=2, causal=True,
    )
    m_bi = TemporalTrackTransformerV3(cfg_bi)
    m_c = TemporalTrackTransformerV3(cfg_c)
    m_c.load_state_dict(m_bi.state_dict(), strict=True)
    m_bi.load_state_dict(m_c.state_dict(), strict=True)


def test_static_causal_mask_shape() -> None:
    """Mask is (1+T+E, 1+T+E) bool, B·H smaller than per-batch."""
    cfg = TemporalTransformerV3Config(seq_len=8, e_max=64, causal=True)
    m = TemporalTrackTransformerV3(cfg).eval()
    mask = m._build_causal_mask(8, 64, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == (1 + 8 + 64, 1 + 8 + 64)
    # CLS row/col always open.
    assert not mask[0, :].any()
    assert not mask[:, 0].any()
    # Frame 0 should not see frame 7.
    assert mask[1, 8].item() is True
    # Frame 7 should see frame 0.
    assert mask[8, 1].item() is False


def test_kv_cache_streaming_matches_full_forward() -> None:
    """K/V cache at non-CLS positions matches the equivalent full forward.

    CLS at position 0 has documented streaming staleness — when an
    incremental step doesn't re-query CLS, its output stays from the
    cold-start call. Frame/event positions match.
    """
    torch.manual_seed(0)
    d, h, ffn, n = 128, 8, 512, 1
    streaming = _StreamingEncoder(d, h, ffn, 0.0, n).eval()
    L = 100
    x = torch.randn(2, L, d)
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool), diagonal=1,
    )
    causal_mask[0, :] = False
    causal_mask[:, 0] = False
    with torch.no_grad():
        full_out, _ = streaming(x, attn_mask=causal_mask)
        out_a, cache = streaming(
            x[:, :50, :], attn_mask=causal_mask[:50, :50],
        )
        out_b, _ = streaming(
            x[:, 50:, :], attn_mask=causal_mask[50:, :],
            kv_cache=cache,
        )
        incr = torch.cat([out_a, out_b], dim=1)
    # Non-CLS positions: must match the full forward.
    diff = (full_out[:, 1:, :] - incr[:, 1:, :]).abs().max()
    assert diff < 1e-3, f"non-CLS K/V cache mismatch: {diff}"
