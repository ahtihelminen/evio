"""V2 temporal track transformer with auxiliary-feature input and an
optional self-supervised bbox-mask head.

Adds three things on top of `temporal_transformer.TemporalTrackTransformer`:

1. **Aux-feature input** ``(B, T, N_AUX)``. Each frame's per-detection
   pipeline scalars (``z_score``, ``m_coh``, ``best_psr``, ``vx_px_s``,
   ``vy_px_s``, ``bg_class``, ``bbox_w``, ``bbox_h``) plus motion
   placeholders (``omega_x``, ``omega_y``, ``omega_z``, ``accel_mag``)
   are projected via a small linear and summed into the patch token.
   Defaults to zeros if the corpus does not carry these fields, so the
   model trains today on the existing patch-only corpus and gains real
   information once the corpus is rebuilt.

2. **Bbox-mask head** ``(B, T, H, W)``. A linear projection from each
   non-CLS token back to a per-pixel logit map at the input patch
   resolution. Trained self-supervised against the patch's own event
   intensity (a soft "where in the crop are the events?" target).
   Forces spatial attention to learn target localisation, not just
   yes/no drone.

3. **MC-dropout-friendly inference**. The encoder dropout is exposed so
   inference can run K stochastic forward passes for uncertainty.

The patch encoder is unchanged from v1 so MAE-pretrained weights from
``pretrain_temporal_mae.py`` load with ``strict=True`` into the v2
model's ``encoder`` submodule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evio.detection.temporal_transformer import _PatchEncoder

# Layout of the auxiliary feature vector. The corpus may not populate all
# of these yet; missing fields default to zero (no information).
AUX_FEATURE_NAMES: tuple[str, ...] = (
    "z_score",
    "m_coh",
    "best_psr",
    "vx_px_s",
    "vy_px_s",
    "bg_class",
    "bbox_w",
    "bbox_h",
    "omega_x",
    "omega_y",
    "omega_z",
    "accel_mag",
)
N_AUX = len(AUX_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class TemporalTransformerV2Config:
    """Hyperparameters for `TemporalTrackTransformerV2`.

    ``aux_dim`` defaults to len(AUX_FEATURE_NAMES); pass a different
    value if rebuilding the corpus with a different aux layout.
    """

    seq_len: int = 16
    patch_size: int = 32
    in_channels: int = 3
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    embedding_dim: int = 128
    dropout: float = 0.1
    aux_dim: int = N_AUX
    enable_bbox_mask_head: bool = True


def _group_count(channels: int) -> int:
    for g in (16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _AuxProjector(nn.Module):
    """Small MLP that projects (B, T, N_AUX) to (B, T, embed_dim).

    Two linear layers with a GELU between them. Layer-norm on the input
    so unscaled raw features (z_score in [0, 30], m_coh in [0, 1])
    don't dominate. When ``aux_dim`` is 0 (corpus has nothing to add),
    the projector becomes a constant-zero pass-through.
    """

    def __init__(self, aux_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.aux_dim = aux_dim
        if aux_dim == 0:
            self.proj = None
            return
        self.norm = nn.LayerNorm(aux_dim)
        self.fc1 = nn.Linear(aux_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.proj = nn.Sequential(self.norm, self.fc1, self.act, self.fc2)

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return torch.zeros(
                aux.shape[0], aux.shape[1], 0,
                device=aux.device, dtype=aux.dtype,
            )
        return self.proj(aux)


class _BboxMaskHead(nn.Module):
    """Per-frame mask decoder: token -> (H, W) logit map.

    Lightweight: just a Linear(embed_dim, H*W). The supervision is a
    soft event-density target derived at training time from the input
    patch itself (BCE-with-logits against ``patch.mean(axis=-1)`` thresh
    -> binary), so the head learns where in the crop the events live.
    """

    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.fc = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = tokens.shape
        logits = self.fc(tokens)
        return logits.view(batch, seq, self.patch_size, self.patch_size)


class TemporalTrackTransformerV2(nn.Module):
    """Drop-in successor to `TemporalTrackTransformer` with aux input.

    Forward signature
    -----------------
    ``forward(patches, aux=None)``:
        ``patches``: (B, T, H, W, 3) uint8 or float32 in [0, 1].
        ``aux``: (B, T, N_AUX) float32 or None (defaults to zeros).

    Returns a dict with the v1 keys (``drone_logits``, ``embeddings``,
    ``validity``) plus ``bbox_mask`` when the head is enabled. All
    tensors are on the model's device.
    """

    def __init__(self, config: TemporalTransformerV2Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = _PatchEncoder(
            embed_dim=config.embed_dim, in_channels=config.in_channels,
        )
        self.aux = _AuxProjector(config.aux_dim, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.seq_len + 1, config.embed_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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
        self.embedding_head = nn.Linear(config.embed_dim, config.embedding_dim)
        self.validity_head = nn.Linear(config.embed_dim, 1)
        if config.enable_bbox_mask_head:
            self.bbox_mask_head: _BboxMaskHead | None = _BboxMaskHead(
                config.embed_dim, config.patch_size,
            )
        else:
            self.bbox_mask_head = None

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = self.config.in_channels
        if x.ndim != 5 or x.shape[-1] != in_channels:
            msg = (
                f"expected (B, T, H, W, {in_channels}), got {tuple(x.shape)}"
            )
            raise ValueError(msg)
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        elif x.dtype != torch.float32:
            x = x.to(torch.float32)
        batch, seq, height, width, _ = x.shape
        x = x.reshape(batch * seq, height, width, in_channels)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        patches: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq = patches.shape[0], patches.shape[1]
        patch_tokens = self.encoder(self._prepare_input(patches))
        patch_tokens = patch_tokens.view(batch, seq, -1)

        if aux is None:
            aux = torch.zeros(
                batch, seq, self.config.aux_dim,
                device=patch_tokens.device, dtype=patch_tokens.dtype,
            )
        aux_tokens = self.aux(aux)
        tokens = patch_tokens + aux_tokens

        cls = self.cls_token.expand(batch, -1, -1)
        z = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : seq + 1]
        z = self.transformer(z)
        cls_out, frame_out = z[:, 0], z[:, 1:]

        out: dict[str, torch.Tensor] = {
            "drone_logits": self.drone_head(frame_out).squeeze(-1),
            "embeddings": nn.functional.normalize(
                self.embedding_head(frame_out), dim=-1,
            ),
            "validity": self.validity_head(cls_out).squeeze(-1),
        }
        if self.bbox_mask_head is not None:
            out["bbox_mask"] = self.bbox_mask_head(frame_out)
        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        patches: torch.Tensor,
        aux: torch.Tensor | None = None,
        *,
        n_samples: int = 8,
    ) -> dict[str, torch.Tensor]:
        """MC-dropout: K forward passes with dropout left active.

        Returns mean and std for `drone_logits` and `validity`.
        """
        self.train()  # enable dropout
        drone_acc: list[torch.Tensor] = []
        valid_acc: list[torch.Tensor] = []
        for _ in range(n_samples):
            out = self.forward(patches, aux)
            drone_acc.append(torch.sigmoid(out["drone_logits"]))
            valid_acc.append(torch.sigmoid(out["validity"]))
        self.eval()
        drone_stack = torch.stack(drone_acc, dim=0)
        valid_stack = torch.stack(valid_acc, dim=0)
        return {
            "drone_prob_mean": drone_stack.mean(dim=0),
            "drone_prob_std": drone_stack.std(dim=0),
            "validity_prob_mean": valid_stack.mean(dim=0),
            "validity_prob_std": valid_stack.std(dim=0),
        }


def count_parameters(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


@dataclass(slots=True)
class TemporalTransformerV2Bundle:
    model: TemporalTrackTransformerV2
    config: TemporalTransformerV2Config
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def to(self, device: torch.device) -> TemporalTransformerV2Bundle:
        self.device = device
        self.model.to(device).eval()
        return self

    @torch.no_grad()
    def predict(
        self,
        patches: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.ascontiguousarray(patches)).to(self.device)
        a = (
            torch.from_numpy(np.ascontiguousarray(aux)).to(self.device)
            if aux is not None else None
        )
        out = self.model(x, a)
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in out.items()
        }


def load_temporal_transformer_v2(
    checkpoint_path: Path,
    *,
    config: TemporalTransformerV2Config | None = None,
    device: torch.device | None = None,
) -> TemporalTransformerV2Bundle:
    checkpoint_path = Path(checkpoint_path)
    if config is None:
        meta_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_meta.json",
        )
        meta = json.loads(meta_path.read_text())
        config = TemporalTransformerV2Config(**meta["config"])
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    model = TemporalTrackTransformerV2(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return TemporalTransformerV2Bundle(
        model=model, config=config,
    ).to(device)


__all__ = [
    "AUX_FEATURE_NAMES",
    "N_AUX",
    "TemporalTrackTransformerV2",
    "TemporalTransformerV2Bundle",
    "TemporalTransformerV2Config",
    "count_parameters",
    "load_temporal_transformer_v2",
]
