"""Convert a temporal-transformer ``.pt`` checkpoint to OpenVINO IR.

Dev-time, one-off step (needs torch — not part of the field runtime): run
this once per checkpoint to produce the ``<stem>_ov.xml``/``.bin`` pair the
live path (``track_classifier.load_track_transformer``) auto-picks up next
to the ``.pt`` file, so the mast never loads the ``.pt`` or imports torch.

Auto-detects v2 vs v3/v4 from the ``<stem>_meta.json`` sidecar, same
convention as the live loader (``e_max`` present -> v3/v4).

    uv run --extra dev python scripts/convert_checkpoint_to_ir.py \
        /path/to/live_detect_v3.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evio.detection.openvino_transformer import (
    export_v2_to_openvino,
    export_v3_to_openvino,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Source .pt checkpoint.")
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for the .xml/.bin pair (default: next to the .pt).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = args.checkpoint
    if not ckpt.is_file():
        print(f"{ckpt} is not a file", file=sys.stderr)
        return 2

    meta_path = ckpt.with_name(f"{ckpt.stem}_meta.json")
    is_v3 = False
    if meta_path.exists():
        cfg = json.loads(meta_path.read_text()).get("config", {})
        is_v3 = "e_max" in cfg

    export = export_v3_to_openvino if is_v3 else export_v2_to_openvino
    xml_path = export(ckpt, args.output_dir)
    print(f"wrote {xml_path} ({'v3/v4' if is_v3 else 'v2'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
