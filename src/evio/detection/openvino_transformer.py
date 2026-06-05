"""OpenVINO FP16 backend for the v2 temporal transformer (Arc 140T iGPU).

The v2 model is tiny (~1.1M params) but the live path runs it on a CPU
thread shared with detection. Moving it to the integrated Arc 140T frees
that thread and removes the worst-case classification spike.

The live path feeds uint8 patches ``(B, 16, 32, 32, 3)`` and reads only
``drone_logits``; aux is always ``None`` (the v2 corpus carries no aux).
So the exported graph takes float32 patches in ``[0, 1]`` with zero aux
baked in and returns ``drone_logits`` ``(B, 16)``. Normalisation lives in
:meth:`OpenVINOTransformerV2Bundle.predict`, mirroring the uint8 branch of
the torch model exactly so the two backends agree numerically.

Export needs torch (dev-time, any platform) and converts the nn.Module
straight to an IR via OpenVINO's PyTorch frontend — no ONNX intermediate.
Inference needs only numpy + openvino: ``GPU`` on the NUC, ``CPU``
elsewhere (incl. macOS arm64, which is how the parity test runs off-rig).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov


def _normalize_patches(patches: np.ndarray) -> np.ndarray:
    """Match the torch model's dtype handling: uint8 -> /255, else as-is."""
    if patches.dtype == np.uint8:
        return patches.astype(np.float32) / 255.0
    return np.ascontiguousarray(patches, dtype=np.float32)


class OpenVINOTransformerV2Bundle:
    """Drop-in for ``TemporalTransformerV2Bundle`` over an OpenVINO IR.

    Exposes the same ``predict(patches, aux=None) -> {"drone_logits": ...}``
    contract the live path calls, so swapping it in needs no call-site
    changes. ``aux`` must stay ``None`` — the exported graph has no aux
    input (the live v2 path never supplies one).
    """

    def __init__(self, compiled: ov.CompiledModel) -> None:
        self._compiled = compiled
        self._output = compiled.output(0)

    def predict(
        self,
        patches: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if aux is not None:
            msg = "OpenVINO v2 backend has no aux input (exported zero-aux)"
            raise ValueError(msg)
        logits = self._compiled(_normalize_patches(patches))[self._output]
        return {"drone_logits": np.asarray(logits, dtype=np.float32)}


def load_openvino_transformer_v2(
    ir_path: Path,
    *,
    device: str = "GPU",
) -> OpenVINOTransformerV2Bundle:
    """Compile a saved IR for ``device`` and wrap it in a bundle.

    ``device`` is an OpenVINO device name: ``GPU`` (Arc 140T), ``CPU``,
    or ``AUTO``. Raises if the device is unavailable so the caller can
    fall back to the torch backend.
    """
    core = ov.Core()
    model = core.read_model(Path(ir_path))
    compiled = core.compile_model(model, device)
    return OpenVINOTransformerV2Bundle(compiled)


def export_v2_to_openvino(
    checkpoint: Path,
    output_dir: Path | None = None,
) -> Path:
    """Export the v2 checkpoint to an FP16 OpenVINO IR.

    Writes ``<stem>_ov.xml`` / ``.bin`` next to the checkpoint and returns
    the ``.xml`` path. The batch axis is left dynamic so a varying number
    of ready tracks can be scored in one call. Runs on CPU; needs torch.
    """
    import torch
    from torch import nn

    from evio.detection.temporal_transformer_v2 import (
        load_temporal_transformer_v2,
    )

    checkpoint = Path(checkpoint)
    out_dir = Path(output_dir) if output_dir is not None else checkpoint.parent
    bundle = load_temporal_transformer_v2(
        checkpoint, device=torch.device("cpu"),
    )

    class _DroneLogits(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, patches: torch.Tensor) -> torch.Tensor:
            return self.inner(patches)["drone_logits"]

    wrapper = _DroneLogits(bundle.model).eval()
    cfg = bundle.config
    example = torch.zeros(
        1, cfg.seq_len, cfg.patch_size, cfg.patch_size, cfg.in_channels,
    )
    # Trace ourselves (check_trace=False) and hand OpenVINO the TorchScript
    # graph: its own auto-trace runs a verification re-trace that spuriously
    # diverges on the squeeze in the drone head.
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    ov_model = ov.convert_model(traced, example_input=example)
    # Leave the batch axis dynamic so a varying ready-track count scores in
    # one call.
    ov_model.reshape(
        ov.PartialShape(
            [-1, cfg.seq_len, cfg.patch_size, cfg.patch_size, cfg.in_channels],
        ),
    )
    xml_path = out_dir / f"{checkpoint.stem}_ov.xml"
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    return xml_path


__all__ = [
    "OpenVINOTransformerV2Bundle",
    "export_v2_to_openvino",
    "load_openvino_transformer_v2",
]
