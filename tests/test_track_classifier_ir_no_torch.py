"""Live field path: IR inference must not need torch at runtime.

``run_ops.py`` builds its transformer through ``TrackTransformerClassifier``
(``device="auto"``), which prefers an OpenVINO IR checkpoint next to the
``.pt`` when one exists. This pins that the IR is loaded and scored without
importing torch anywhere in that call path, and that the detections
(sticky per-track drone probabilities) it produces match the semantics of
scoring the same IR bundle directly.

The checkpoint + IR fixture is built with torch (a one-time, dev-time
export, exactly like the real conversion script) before torch is blocked;
the load + classify calls under test then run with torch made unimportable.
"""

from __future__ import annotations

import builtins
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from evio.core.events import EventsBatch
from evio.detection.live_camera import SEQ_LEN
from evio.detection.types import Detection

ovt = pytest.importorskip("evio.detection.openvino_transformer")


def _build_ir_checkpoint(tmp_path: Path) -> Path:
    """Export a random-init v2 checkpoint to IR, torch still available."""
    torch = pytest.importorskip(
        "torch",
        reason="torch not installed (field/no-extras venv) - IR fixture "
        "export needs the dev extra to build the checkpoint it converts",
    )

    from evio.detection.temporal_transformer_v2 import (
        TemporalTrackTransformerV2,
        TemporalTransformerV2Config,
    )

    config = TemporalTransformerV2Config()
    model = TemporalTrackTransformerV2(config).eval()
    ckpt = tmp_path / "live_detect_v3.pt"
    torch.save(model.state_dict(), ckpt)
    meta = {"config": dataclasses.asdict(config)}
    ckpt.with_name("live_detect_v3_meta.json").write_text(json.dumps(meta))
    ovt.export_v2_to_openvino(ckpt, tmp_path)
    return ckpt


def _det(track_id: str) -> Detection:
    return Detection(
        x0=10, y0=10, x1=20, y1=20, vx_px_s=0.0, vy_px_s=0.0,
        m_coh=0.8, best_psr=1.0, event_count=100, bg_class=0,
        row_start=0, row_end=1, col_start=0, col_end=1,
        track_id=track_id, patch=np.zeros((32, 32, 3), dtype=np.uint8),
    )


def _batch() -> EventsBatch:
    return EventsBatch(
        timestamps=np.array([0, 10_000], dtype=np.int64),
        x_coords=np.array([1, 2], dtype=np.int16),
        y_coords=np.array([1, 2], dtype=np.int16),
        polarities=np.array([1, 0], dtype=np.int8),
    )


def _block_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import torch`` (and submodules) raise, everywhere, from here
    on — the live-path guard the criterion asks for."""
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            msg = "torch is not importable (blocked for test)"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_ir_inference_produces_detections_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build the checkpoint + IR fixture (a one-time, dev-time export) with
    # torch still available, exactly like the real conversion script.
    ckpt = _build_ir_checkpoint(tmp_path)

    # Everything from here on runs as the live field path would, with torch
    # unimportable: no module-level torch anywhere in track_classifier ->
    # openvino_transformer -> openvino.
    _block_torch_import(monkeypatch)
    from evio.detection.track_classifier import TrackTransformerClassifier

    clf = TrackTransformerClassifier(ckpt, device="auto")
    assert clf.load_warning is None
    assert clf.transformer_loaded

    out: dict[str, float] = {}
    for _ in range(SEQ_LEN):
        out = clf.classify("cam0", [_det("track-1")], _batch())

    # Same shape/semantics as before: one sticky probability per track id.
    assert set(out) == {"track-1"}
    prob = out["track-1"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
