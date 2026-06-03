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
        if config.enable_bbox_mask_head:
            self.bbox_mask_head: _BboxMaskHead | None = _BboxMaskHead(
                config.embed_dim, config.patch_size,
            )
        else:
            self.bbox_mask_head = None

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

    def forward(
        self,
        patches: torch.Tensor,
        events: torch.Tensor,
        event_mask: torch.Tensor,
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

        z = self.transformer(z, src_key_padding_mask=key_padding)
        cls_out = z[:, 0]
        frame_out = z[:, 1: 1 + seq]

        out: dict[str, torch.Tensor] = {
            "drone_logits": self.drone_head(frame_out).squeeze(-1),
            "embeddings": torch_F.normalize(
                self.embedding_head(frame_out), dim=-1,
            ),
            "validity": self.validity_head(cls_out).squeeze(-1),
        }
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
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
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
