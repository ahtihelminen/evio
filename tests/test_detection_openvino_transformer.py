"""Parity test: the OpenVINO v2 backend matches the torch backend.

Self-contained — builds a random-init v2 model, exports it to an FP16 IR,
and compares ``drone_logits`` from the torch-CPU bundle against the
OpenVINO-CPU bundle on identical uint8 patches. Skips when openvino is not
installed, so it is a no-op in environments without the iGPU extra.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evio.detection.temporal_transformer_v2 import (
    TemporalTrackTransformerV2,
    TemporalTransformerV2Config,
    load_temporal_transformer_v2,
)
from evio.detection.temporal_transformer_v3 import (
    TemporalTrackTransformerV3,
    TemporalTransformerV3Config,
    load_temporal_transformer_v3,
)

ovt = pytest.importorskip("evio.detection.openvino_transformer")


def _write_random_checkpoint(tmp_path: Path) -> Path:
    config = TemporalTransformerV2Config()
    model = TemporalTrackTransformerV2(config).eval()
    ckpt = tmp_path / "v2.pt"
    torch.save(model.state_dict(), ckpt)
    meta = {"config": dataclasses.asdict(config)}
    ckpt.with_name("v2_meta.json").write_text(json.dumps(meta))
    return ckpt


def test_openvino_matches_torch(tmp_path: Path) -> None:
    ckpt = _write_random_checkpoint(tmp_path)
    ir_path = ovt.export_v2_to_openvino(ckpt, tmp_path)

    cpu = torch.device("cpu")
    torch_bundle = load_temporal_transformer_v2(ckpt, device=cpu)
    ov_bundle = ovt.load_openvino_transformer_v2(ir_path, device="CPU")

    rng = np.random.default_rng(0)
    patches = rng.integers(0, 256, (5, 16, 32, 32, 3), dtype=np.uint8)

    torch_logits = torch_bundle.predict(patches)["drone_logits"]
    ov_logits = ov_bundle.predict(patches)["drone_logits"]

    assert ov_logits.shape == torch_logits.shape == (5, 16)
    np.testing.assert_allclose(ov_logits, torch_logits, atol=5e-2, rtol=2e-2)


def test_openvino_handles_single_track(tmp_path: Path) -> None:
    """The dynamic batch axis serves a one-track call (common live case)."""
    ckpt = _write_random_checkpoint(tmp_path)
    ir_path = ovt.export_v2_to_openvino(ckpt, tmp_path)
    ov_bundle = ovt.load_openvino_transformer_v2(ir_path, device="CPU")
    patches = np.zeros((1, 16, 32, 32, 3), dtype=np.uint8)
    assert ov_bundle.predict(patches)["drone_logits"].shape == (1, 16)


def test_openvino_rejects_aux(tmp_path: Path) -> None:
    ckpt = _write_random_checkpoint(tmp_path)
    ir_path = ovt.export_v2_to_openvino(ckpt, tmp_path)
    ov_bundle = ovt.load_openvino_transformer_v2(ir_path, device="CPU")
    patches = np.zeros((1, 16, 32, 32, 3), dtype=np.uint8)
    aux = np.zeros((1, 16, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="aux"):
        ov_bundle.predict(patches, aux)


def _write_random_v3_checkpoint(tmp_path: Path) -> Path:
    config = TemporalTransformerV3Config()
    model = TemporalTrackTransformerV3(config).eval()
    ckpt = tmp_path / "v3.pt"
    torch.save(model.state_dict(), ckpt)
    meta = {"config": dataclasses.asdict(config)}  # e_max present -> v3 route
    ckpt.with_name("v3_meta.json").write_text(json.dumps(meta))
    return ckpt


def test_openvino_v3_matches_torch_with_embeddings(tmp_path: Path) -> None:
    """v3/v4 export: both drone_logits AND the ReID embeddings match torch."""
    ckpt = _write_random_v3_checkpoint(tmp_path)
    ir_path = ovt.export_v3_to_openvino(ckpt, tmp_path)

    cpu = torch.device("cpu")
    torch_bundle = load_temporal_transformer_v3(ckpt, device=cpu)
    ov_bundle = ovt.load_openvino_transformer_v3(ir_path, device="CPU")
    cfg = torch_bundle.config

    rng = np.random.default_rng(0)
    ps = cfg.patch_size
    patches = rng.integers(0, 256, (3, cfg.seq_len, ps, ps, 3), dtype=np.uint8)
    events = rng.standard_normal(
        (3, cfg.e_max, cfg.event_in_dim),
    ).astype(np.float32)
    mask = np.ones((3, cfg.e_max), dtype=np.bool_)

    torch_out = torch_bundle.predict(patches, events, mask)
    ov_out = ov_bundle.predict(patches, events, mask)

    assert ov_out["drone_logits"].shape == (3, cfg.seq_len)
    assert ov_out["embeddings"].shape == torch_out["embeddings"].shape
    np.testing.assert_allclose(
        ov_out["drone_logits"], torch_out["drone_logits"],
        atol=5e-2, rtol=2e-2,
    )
    np.testing.assert_allclose(
        ov_out["embeddings"], torch_out["embeddings"], atol=5e-2, rtol=2e-2,
    )


def _write_random_r5_checkpoint(tmp_path: Path) -> Path:
    config = TemporalTransformerV3Config(
        seq_len=8, e_max=64, n_layers=2, state_enabled=True, state_dim=32,
    )
    model = TemporalTrackTransformerV3(config).eval()
    ckpt = tmp_path / "r5.pt"
    torch.save(model.state_dict(), ckpt)
    ckpt.with_name("r5_meta.json").write_text(
        json.dumps({"config": dataclasses.asdict(config)}),
    )
    return ckpt


def test_openvino_r5_stateful_matches_torch(tmp_path: Path) -> None:
    """R5 export: drone_logits, embeddings AND h_track match torch, and the
    IR is stateful (4 inputs / 3 outputs — h_in threads to h_out)."""
    ckpt = _write_random_r5_checkpoint(tmp_path)
    ir_path = ovt.export_v3_to_openvino(ckpt, tmp_path)

    cpu = torch.device("cpu")
    torch_bundle = load_temporal_transformer_v3(ckpt, device=cpu)
    ov_bundle = ovt.load_openvino_transformer_v3(ir_path, device="CPU")
    assert ov_bundle.stateful
    cfg = torch_bundle.config

    rng = np.random.default_rng(0)
    ps = cfg.patch_size
    b = 3
    patches = rng.integers(0, 256, (b, cfg.seq_len, ps, ps, 3), dtype=np.uint8)
    events = rng.standard_normal(
        (b, cfg.e_max, cfg.event_in_dim),
    ).astype(np.float32)
    mask = np.ones((b, cfg.e_max), dtype=np.bool_)
    h_in = rng.standard_normal((b, cfg.state_dim)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_bundle.model(
            torch.from_numpy(patches), torch.from_numpy(events),
            torch.from_numpy(mask), torch.from_numpy(h_in),
        )
    ov_out = ov_bundle.predict(patches, events, mask, h_in)

    assert ov_out["h_track"].shape == (b, cfg.state_dim)
    for key in ("drone_logits", "embeddings", "h_track"):
        np.testing.assert_allclose(
            ov_out[key], torch_out[key].numpy(), atol=5e-2, rtol=2e-2,
        )
