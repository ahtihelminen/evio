"""Per-track temporal transformer with drone, Re-ID, and validity heads.

Replaces the single-patch CNN classifier (`classifier.py`) with a model
that consumes the last T patches of a tracked candidate and outputs
three coupled signals in one forward pass:

1. Per-frame drone logits (B, T) — sigmoid for probability.
2. Per-frame L2-normed Re-ID embedding (B, T, embedding_dim) — cosine
   similarity across tracks supersedes the broken
   `SignatureBuffer.signature()` mean.
3. Trajectory-validity logit (B,) — sigmoid for the "[CLS]" probability
   that this is a coherent track at all.

The patch encoder reuses the GroupNorm conv-stem pattern from
`FourClassClassifier` (BatchNorm is avoided because training uses
class-balanced sampling while eval sees the natural distribution).

Input shape is `(B, T, H, W, C)` where `C = config.in_channels` — 3 for
the legacy BGR aggregate corpus, ``2 * n_bins`` for voxel-grid corpora
(see :mod:`evio.detection.voxel_patch`). Dtype `uint8` or `float32` in
`[0, 1]`. The module handles the channel permutation and uint8→float
scaling internally so callers can pass either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TemporalTransformerConfig:
    """Hyperparameters for `TemporalTrackTransformer`.

    The spec calls for ~500k params; defaults below land near ~900k with
    `embed_dim=128`, `n_layers=4`, `ffn_dim=512`. The hard cap enforced
    by the param-count test is 1M.
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


def _group_count(channels: int) -> int:
    """Match `FourClassClassifier`'s GroupNorm grouping rule."""
    for g in (16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _PatchEncoder(nn.Module):
    """3 conv blocks (stride-2 each) → AdaptiveAvgPool2d(1) → Linear.

    Borrowed from `FourClassClassifier`. Output: one `embed_dim`-D token
    per input patch.
    """

    def __init__(self, embed_dim: int, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        c1, c2, c3 = 32, 64, 128
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c1), c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c2), c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c3), c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(c3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.project(z)


class TemporalTrackTransformer(nn.Module):
    """Temporal transformer over a T-frame patch sequence.

    Architecture
    ------------
    - Per-frame patch encoder (3 stride-2 conv blocks + linear) → T tokens.
    - Learned `[CLS]` token prepended → T+1 tokens.
    - Learned positional embedding (T+1, embed_dim).
    - `nn.TransformerEncoder` with `norm_first=True` (works better for
      small models).
    - Three heads:
      * `drone_head`: Linear(embed_dim, 1) applied to each of the T
        non-CLS tokens → (B, T) logits.
      * `embedding_head`: Linear(embed_dim, embedding_dim) applied to
        each of the T non-CLS tokens, then L2-norm along the last dim.
      * `validity_head`: Linear(embed_dim, 1) applied to the [CLS]
        token → (B,) logit.
    """

    def __init__(self, config: TemporalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _PatchEncoder(
            embed_dim=config.embed_dim, in_channels=config.in_channels,
        )
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

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, C) uint8 or float → (B*T, C, H, W) float32 in [0, 1]."""
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, seq = x.shape[0], x.shape[1]
        tokens = self.encoder(self._prepare_input(x))
        tokens = tokens.view(batch, seq, -1)
        cls = self.cls_token.expand(batch, -1, -1)
        z = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : seq + 1]
        z = self.transformer(z)
        cls_out, frame_out = z[:, 0], z[:, 1:]
        return {
            "drone_logits": self.drone_head(frame_out).squeeze(-1),
            "embeddings": nn.functional.normalize(
                self.embedding_head(frame_out), dim=-1,
            ),
            "validity": self.validity_head(cls_out).squeeze(-1),
        }


def count_parameters(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


@dataclass(slots=True)
class TemporalTransformerBundle:
    """Trained `TemporalTrackTransformer` packaged for inference."""

    model: TemporalTrackTransformer
    config: TemporalTransformerConfig
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def to(self, device: torch.device) -> TemporalTransformerBundle:
        self.device = device
        self.model.to(device).eval()
        return self

    @torch.no_grad()
    def predict(self, patches: np.ndarray) -> dict[str, np.ndarray]:
        """patches: (B, T, H, W, 3) uint8.

        Returns dict with `drone_logits`, `embeddings`, `validity` as
        CPU numpy arrays (float32).
        """
        x = torch.from_numpy(np.ascontiguousarray(patches)).to(self.device)
        out = self.model(x)
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in out.items()
        }


def load_temporal_transformer(
    checkpoint_path: Path,
    *,
    config: TemporalTransformerConfig | None = None,
    device: torch.device | None = None,
) -> TemporalTransformerBundle:
    """Load a state_dict checkpoint and wrap it in a bundle.

    If `config` is None, read it from a sibling `<stem>_meta.json` next
    to the .pt (same convention as `train_classifier.py`).
    """
    checkpoint_path = Path(checkpoint_path)
    if config is None:
        meta_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_meta.json",
        )
        meta = json.loads(meta_path.read_text())
        config = TemporalTransformerConfig(**meta["config"])
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    model = TemporalTrackTransformer(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return TemporalTransformerBundle(
        model=model, config=config,
    ).to(device)


__all__ = [
    "TemporalTrackTransformer",
    "TemporalTransformerBundle",
    "TemporalTransformerConfig",
    "count_parameters",
    "load_temporal_transformer",
]
