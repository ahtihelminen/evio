"""``scripts/convert_checkpoint_to_ir.py`` produces a usable IR pair.

This is the repeatable, committed conversion step the migration issue
requires: given a ``.pt`` (+ optional meta sidecar), it writes the IR the
live path auto-picks up, without requiring the caller to know v2 vs v3.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("evio.detection.openvino_transformer")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_checkpoint_to_ir as convert_script

from evio.detection.temporal_transformer_v2 import (
    TemporalTrackTransformerV2,
    TemporalTransformerV2Config,
)


def test_convert_v2_checkpoint_writes_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TemporalTransformerV2Config()
    model = TemporalTrackTransformerV2(config).eval()
    ckpt = tmp_path / "live_detect_v3.pt"
    torch.save(model.state_dict(), ckpt)
    ckpt.with_name("live_detect_v3_meta.json").write_text(
        json.dumps({"config": dataclasses.asdict(config)}),
    )

    monkeypatch.setattr(sys, "argv", ["convert_checkpoint_to_ir.py", str(ckpt)])
    assert convert_script.main() == 0

    ir_xml = ckpt.with_name(f"{ckpt.stem}_ov.xml")
    ir_bin = ckpt.with_name(f"{ckpt.stem}_ov.bin")
    assert ir_xml.is_file()
    assert ir_bin.is_file()
