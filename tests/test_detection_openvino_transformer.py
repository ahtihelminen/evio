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
