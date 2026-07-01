"""V3 temporal transformer — spatial BGR + raw-event dual branches.

The voxel V2 attempt failed because per-event signal got drowned in
GroupNorm-noisy channels of the 16-channel encoder (3 overnight runs,
all six arms collapsed). V3 sidesteps the encoder mismatch entirely:

* **Spatial branch** — the proven 3-channel BGR aggregate path
  (untouched ``_PatchEncoder`` from v1/v2). Provides the dense
  spatial overview the drone head needs.
* **Event branch** — raw ``(t, x, y, p)`` tokens projected via a
  small MLP into the 128-D transformer working space. Provides
  sub-window temporal resolution the BGR aggregate throws away
  (propeller chop, blade-flash duty cycle).

The two token streams are concatenated (with a token-type embedding
to disambiguate frame vs event tokens) and fed into a single
TransformerEncoder. The CLS token + per-frame outputs drive the
existing four heads (drone, embedding, validity, bbox_mask).

Event tokenisation lives in :mod:`evio.detection.event_tokens`. The
encoder consumes ``(events, event_mask)`` produced there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_F

from evio.detection.temporal_transformer import _PatchEncoder


@dataclass(frozen=True, slots=True)
class TemporalTransformerV3Config:
    """Hyperparameters for `TemporalTrackTransformerV3`."""

    seq_len: int = 16
    patch_size: int = 32
    embed_dim: int = 128
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 512
    embedding_dim: int = 128
    dropout: float = 0.1
    e_max: int = 1024
    event_in_dim: int = 4
    enable_bbox_mask_head: bool = True
    causal: bool = False  # v4: causal attention mask + per-track K/V cacheable
    # R5: explicit per-track recurrent state (h_track). When enabled the model
    # takes an extra ``h_in`` and returns ``h_track`` (h_out), a fixed-size
    # state carried across windows by the caller — the deployable alternative
    # to the per-window dense event attention (E0 Sub-gate B validated the
    # (B, state_dim) f32 in/out round-trips on the Arc). Default off = the v3/v4
    # forward is byte-identical and existing checkpoints load unchanged.
    state_enabled: bool = False
    state_dim: int = 256


class _StreamingMHA(nn.Module):
    """Multi-head attention with optional per-call K/V cache.

    Mathematically equivalent to ``nn.MultiheadAttention(batch_first=
    True, bias=True)`` when ``kv_cache`` is None and no causal masking
    is intended. We use a single ``in_proj_weight`` of shape
    ``(3*D, D)`` named identically to PyTorch's MHA so v3 checkpoints
    load with strict=True.

    When ``kv_cache`` is provided, ``query`` is interpreted as only
    the NEW tokens (queries for the latest step), ``key/value`` are
    the new tokens' K/V (computed locally), then concatenated with
    the cached K/V history. The output has the same length as the
    new ``query``. The returned cache is the appended (K_new, V_new)
    tuple ready to be passed back next call.
    """

    def __init__(
        self, embed_dim: int, n_heads: int, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            msg = f"embed_dim={embed_dim} not divisible by n_heads={n_heads}"
            raise ValueError(msg)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout_p = dropout
        # Match nn.MultiheadAttention's naming so strict=True loads
        # work with v3 checkpoints.
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * embed_dim, embed_dim),
        )
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _project(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        qkv = torch_F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, head_dim)
        batch, length, _ = x.shape
        x = x.view(batch, length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self._project(query)
        q = self._split_heads(q)  # (B, H, L_new, hd)
        k_new = self._split_heads(k)
        v_new = self._split_heads(v)
        if kv_cache is not None:
            k_full = torch.cat([kv_cache[0], k_new], dim=2)
            v_full = torch.cat([kv_cache[1], v_new], dim=2)
        else:
            k_full = k_new
            v_full = v_new
        new_cache = (k_full, v_full)

        # attn_mask: (L_q, L_k) where True = block
        # key_padding_mask: (B, L_k) where True = pad
        if key_padding_mask is not None:
            kp = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L_k)
        else:
            kp = None
        # Use SDPA which handles the masks + softmax + dropout in a
        # fused kernel; works on MPS and CUDA.
        if attn_mask is not None and kp is not None:
            combined = attn_mask.unsqueeze(0).unsqueeze(0) | kp
        elif attn_mask is not None:
            combined = attn_mask.unsqueeze(0).unsqueeze(0).expand(
                q.shape[0], self.n_heads, -1, -1,
            )
        elif kp is not None:
            combined = kp.expand(-1, self.n_heads, q.shape[2], -1)
        else:
            combined = None
        if combined is not None:
            # SDPA expects float mask added to logits OR bool with
            # True=allow. We've used True=block above; invert for SDPA.
            attn_bias = torch.zeros_like(
                combined, dtype=q.dtype,
            ).masked_fill(combined, float("-inf"))
        else:
            attn_bias = None
        out = torch_F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        # (B, H, L_new, hd) -> (B, L_new, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(
            q.shape[0], q.shape[2], self.embed_dim,
        )
        return self.out_proj(out), new_cache


class _StreamingEncoderLayer(nn.Module):
    """Transformer encoder layer w/ K/V cache support.

    Mathematically equivalent to ``nn.TransformerEncoderLayer(d_model,
    nhead, dim_feedforward, dropout, batch_first=True, norm_first=
    True)``. State-dict key names match PyTorch's so checkpoints load
    with ``strict=True``.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = _StreamingMHA(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # norm_first
        x_norm = self.norm1(x)
        attn_out, new_cache = self.self_attn(
            x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            kv_cache=kv_cache,
        )
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        ff = self.linear2(self.dropout(torch_F.relu(self.linear1(x_norm))))
        x = x + self.dropout2(ff)
        return x, new_cache


class _StreamingEncoder(nn.Module):
    """Stack of streaming encoder layers; per-layer K/V cache."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int,
        dropout: float, n_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_StreamingEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
            ) for _ in range(n_layers)],
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[
        torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, c = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                kv_cache=layer_cache,
            )
            new_cache.append(c)
        return x, new_cache


class _EventEncoder(nn.Module):
    """Per-event linear-GELU-linear into the transformer's working space."""

    def __init__(self, embed_dim: int, in_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, max(32, embed_dim // 4))
        self.fc2 = nn.Linear(max(32, embed_dim // 4), embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch_F.gelu(self.fc1(x)))


class _BboxMaskHead(nn.Module):
    """Per-frame mask decoder: token -> (H, W) logit map."""

    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.fc = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = tokens.shape
        logits = self.fc(tokens)
        return logits.view(batch, seq, self.patch_size, self.patch_size)


class TemporalTrackTransformerV3(nn.Module):
    """Dual-branch transformer over BGR frames + raw event tokens.

    Forward signature:
        ``forward(patches, events, event_mask)``:
            ``patches``: (B, T, H, W, 3) uint8 or float32 in [0, 1].
            ``events``: (B, E_max, 4) float32 — t/x/y normalised + p signed.
            ``event_mask``: (B, E_max) bool — True for valid event tokens.

    Returns the same dict v2 emits: ``drone_logits`` (B, T),
    ``embeddings`` (B, T, embedding_dim) L2-normed, ``validity``
    (B,), plus ``bbox_mask`` (B, T, H, W) when enabled.
    """

    def __init__(self, config: TemporalTransformerV3Config) -> None:
        super().__init__()
        self.config = config
        self.spatial_encoder = _PatchEncoder(
            embed_dim=config.embed_dim, in_channels=3,
        )
        self.event_encoder = _EventEncoder(
            config.embed_dim, in_dim=config.event_in_dim,
        )
        # type 0 = frame token, type 1 = event token
        self.type_embed = nn.Embedding(2, config.embed_dim)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, config.embed_dim),
        )
        # Positional embed only on CLS + frame tokens; events carry
        # their position in the input features.
        self.pos_embed_frames = nn.Parameter(
            torch.zeros(1, config.seq_len + 1, config.embed_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_frames, std=0.02)
        if config.causal:
            # Hand-rolled stack exposing per-layer K/V hooks for the
            # streaming inference path. Math-equivalent to
            # nn.TransformerEncoder(norm_first=True) in the non-cached
            # forward; state dict keys match for cross-loading.
            self.transformer: nn.Module = _StreamingEncoder(
                d_model=config.embed_dim, nhead=config.n_heads,
                dim_feedforward=config.ffn_dim, dropout=config.dropout,
                n_layers=config.n_layers,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.n_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=config.n_layers,
            )
        self.drone_head = nn.Linear(config.embed_dim, 1)
        self.embedding_head = nn.Linear(
            config.embed_dim, config.embedding_dim,
        )
        self.validity_head = nn.Linear(config.embed_dim, 1)
        # R5 explicit-state cell — created only when enabled so v3/v4
        # checkpoints (no state params) still load strict=True.
        if config.state_enabled:
            self.state_cell: nn.GRUCell | None = nn.GRUCell(
                config.embed_dim, config.state_dim,
            )
            self.state_proj: nn.Linear | None = nn.Linear(
                config.state_dim, config.embed_dim,
            )
        else:
            self.state_cell = None
            self.state_proj = None
        if config.enable_bbox_mask_head:
            self.bbox_mask_head: _BboxMaskHead | None = _BboxMaskHead(
                config.embed_dim, config.patch_size,
            )
        else:
            self.bbox_mask_head = None
        # Cached static causal mask (set on first forward call when
        # config.causal is True). Tuple is (seq, e_max, device).
        self._cached_causal_mask: torch.Tensor | None = None
        self._cached_causal_mask_key: tuple[int, int, torch.device] | None = (
            None
        )

    def _prepare_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != 3:
            msg = f"expected (B, T, H, W, 3), got {tuple(x.shape)}"
            raise ValueError(msg)
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        elif x.dtype != torch.float32:
            x = x.to(torch.float32)
        batch, seq, height, width, _ = x.shape
        x = x.reshape(batch * seq, height, width, 3)
        return x.permute(0, 3, 1, 2).contiguous()

    def _build_causal_mask(
        self, seq: int, e_max: int, device: torch.device,
    ) -> torch.Tensor:
        """Static (L, L) causal mask shared across batch.

        Token order: [CLS, frame_0..frame_{T-1}, event_0..event_{E-1}].
        Causality rules:
            * CLS attends to everything (it's the pooled summary).
            * Frame token at position t attends to CLS + frames [0..t]
              + events with normalised time <= (t+1)/T.
            * Event token at position i attends to CLS + frames with
              position <= floor((i+1) * T / E) - 1 + events [0..i].

        Position-based for the event axis: `tokenize_events_in_track`
        subsamples events uniformly in time when count > e_max and
        pads with zeros when count < e_max (event_mask kills the
        padding). So event position i has t_norm ≈ i / e_max, and a
        position-based mask is the correct approximation for the
        uniform-stride case and irrelevant (masked out) for the
        sparse case. Memory: single (L, L) bool tensor, B·H times
        smaller than the per-batch version.

        Returns a boolean mask aligned with PyTorch's `attn_mask`
        convention: True at (q, k) means k is masked for q's
        attention. Cached on `self._cached_causal_mask` keyed by
        (seq, e_max, device).
        """
        if (
            self._cached_causal_mask is not None
            and self._cached_causal_mask_key == (seq, e_max, device)
        ):
            return self._cached_causal_mask

        # Per-token continuous time, CLS=0, frame_t = (t+1)/T,
        # event_i = i / e_max (after the uniform stride subsample).
        cls_time = torch.zeros(1, device=device, dtype=torch.float32)
        frame_time = (
            torch.arange(1, seq + 1, device=device, dtype=torch.float32)
            / seq
        )
        event_time = (
            torch.arange(e_max, device=device, dtype=torch.float32)
            / max(e_max, 1)
        )
        all_time = torch.cat([cls_time, frame_time, event_time])

        # mask[q, k] = True if k is FUTURE relative to q.
        mask = all_time.unsqueeze(0) > all_time.unsqueeze(1)
        # CLS row + CLS column always-open.
        mask[0, :] = False
        mask[:, 0] = False

        self._cached_causal_mask = mask
        self._cached_causal_mask_key = (seq, e_max, device)
        return mask

    def forward(
        self,
        patches: torch.Tensor,
        events: torch.Tensor,
        event_mask: torch.Tensor,
        h_in: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq = patches.shape[0], patches.shape[1]
        device = patches.device

        spatial_tokens = self.spatial_encoder(self._prepare_patches(patches))
        spatial_tokens = spatial_tokens.view(batch, seq, -1)

        if events.dtype != torch.float32:
            events = events.to(torch.float32)
        event_tokens = self.event_encoder(events)  # (B, E, D)

        type_frame = self.type_embed(
            torch.zeros(1, dtype=torch.long, device=device),
        ).unsqueeze(0)
        type_event = self.type_embed(
            torch.ones(1, dtype=torch.long, device=device),
        ).unsqueeze(0)
        spatial_tokens = spatial_tokens + type_frame
        event_tokens = event_tokens + type_event

        cls = (
            self.cls_token.expand(batch, -1, -1)
            + self.pos_embed_frames[:, :1]
        )
        spatial_tokens = (
            spatial_tokens + self.pos_embed_frames[:, 1: seq + 1]
        )

        z = torch.cat([cls, spatial_tokens, event_tokens], dim=1)

        # Padding mask: TransformerEncoder treats True as PAD (skip).
        valid_frames = torch.ones(
            batch, 1 + seq, dtype=torch.bool, device=device,
        )
        valid = torch.cat([valid_frames, event_mask.bool()], dim=1)
        key_padding = ~valid

        if self.config.causal:
            # Shared (L, L) mask across batch — both nn.TransformerEncoder
            # and _StreamingEncoder accept a 2D attn_mask that broadcasts
            # across batch and head axes.
            attn_mask = self._build_causal_mask(
                seq, events.shape[1], device,
            )
            if isinstance(self.transformer, _StreamingEncoder):
                z, _ = self.transformer(
                    z, attn_mask=attn_mask, key_padding_mask=key_padding,
                )
            else:
                z = self.transformer(
                    z, mask=attn_mask, src_key_padding_mask=key_padding,
                )
        else:
            z = self.transformer(z, src_key_padding_mask=key_padding)
        cls_out = z[:, 0]
        frame_out = z[:, 1: 1 + seq]

        # R5: update the per-track state from this window's CLS summary and
        # condition the heads on the accumulated state (additive, so head dims
        # are unchanged). h_track rides out to the caller for the next window.
        h_out: torch.Tensor | None = None
        if self.state_cell is not None and self.state_proj is not None:
            if h_in is None:
                h_in = torch.zeros(
                    batch, self.config.state_dim, device=device,
                    dtype=cls_out.dtype,
                )
            h_out = self.state_cell(cls_out, h_in)
            cond = self.state_proj(h_out)
            frame_out = frame_out + cond.unsqueeze(1)
            cls_out = cls_out + cond

        out: dict[str, torch.Tensor] = {
            "drone_logits": self.drone_head(frame_out).squeeze(-1),
            "embeddings": torch_F.normalize(
                self.embedding_head(frame_out), dim=-1,
            ),
            "validity": self.validity_head(cls_out).squeeze(-1),
        }
        if h_out is not None:
            out["h_track"] = h_out
        if self.bbox_mask_head is not None:
            out["bbox_mask"] = self.bbox_mask_head(frame_out)
        return out


def count_parameters(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


@dataclass(slots=True)
class TemporalTransformerV3Bundle:
    model: TemporalTrackTransformerV3
    config: TemporalTransformerV3Config
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def to(self, device: torch.device) -> TemporalTransformerV3Bundle:
        self.device = device
        self.model.to(device).eval()
        return self

    @torch.no_grad()
    def predict(
        self,
        patches: np.ndarray,
        events: np.ndarray,
        event_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.ascontiguousarray(patches)).to(self.device)
        e = torch.from_numpy(np.ascontiguousarray(events)).to(self.device)
        m = torch.from_numpy(np.ascontiguousarray(event_mask)).to(self.device)
        out = self.model(x, e, m)
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in out.items()
        }


def load_temporal_transformer_v3(
    checkpoint_path: Path,
    *,
    config: TemporalTransformerV3Config | None = None,
    device: torch.device | None = None,
) -> TemporalTransformerV3Bundle:
    checkpoint_path = Path(checkpoint_path)
    if config is None:
        meta_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_meta.json",
        )
        meta = json.loads(meta_path.read_text())
        config = TemporalTransformerV3Config(**meta["config"])
    if device is None:
        if torch.xpu.is_available():
            device = torch.device("xpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model = TemporalTrackTransformerV3(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return TemporalTransformerV3Bundle(
        model=model, config=config,
    ).to(device)


__all__ = [
    "TemporalTrackTransformerV3",
    "TemporalTransformerV3Bundle",
    "TemporalTransformerV3Config",
    "count_parameters",
    "load_temporal_transformer_v3",
]
