"""Regenerate the committed OpenVINO IR fixture used by
``tests/test_track_classifier_ir_no_torch.py``.

Run from the repo root with the dev extra — torch is needed only here, to
build the checkpoint and convert it, never to load the result:

    uv run --frozen --extra dev python tests/fixtures/track_classifier_ir/regenerate.py

Writes ``live_detect_v3_meta.json``, ``live_detect_v3_ov.xml`` and
``live_detect_v3_ov.bin`` into this directory; those three files are what is
committed. The ``.pt`` the export reads is written to a temporary directory
and discarded, because ``TrackTransformerClassifier`` never opens it once an
IR sits beside it — that IR-only load is exactly the field path the test pins.

The weights are a random-init ``TemporalTrackTransformerV2`` on its default
config, seeded so a rerun reproduces the same IR. The test asserts tensor
shape and probability semantics, not learned behaviour, so trained weights
would add megabytes without adding coverage.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import torch

from evio.detection.openvino_transformer import export_v2_to_openvino
from evio.detection.temporal_transformer_v2 import (
    TemporalTrackTransformerV2,
    TemporalTransformerV2Config,
)

SEED = 0
STEM = "live_detect_v3"
FIXTURE_DIR = Path(__file__).parent


def main() -> None:
    torch.manual_seed(SEED)
    config = TemporalTransformerV2Config()
    model = TemporalTrackTransformerV2(config).eval()
    meta = json.dumps({"config": dataclasses.asdict(config)})

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / f"{STEM}.pt"
        torch.save(model.state_dict(), ckpt)
        ckpt.with_name(f"{STEM}_meta.json").write_text(meta)
        export_v2_to_openvino(ckpt, FIXTURE_DIR)

    (FIXTURE_DIR / f"{STEM}_meta.json").write_text(meta)


if __name__ == "__main__":
    main()
