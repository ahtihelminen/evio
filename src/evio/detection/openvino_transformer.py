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
    config: dict[str, str] | None = None,
) -> OpenVINOTransformerV2Bundle:
    """Compile a saved IR for ``device`` and wrap it in a bundle.

    ``device`` is an OpenVINO device name: ``GPU`` (Arc 140T), ``CPU``,
    or ``AUTO``. Raises if the device is unavailable so the caller can
    fall back to the torch backend.
    ``config`` passes plugin properties straight to ``compile_model`` —
    e.g. the parity test pins ``INFERENCE_PRECISION_HINT: "f32"`` on CPU,
    since that plugin's default there is fp16 and its reduced-precision
    kernel selection is sensitive to system load, occasionally drifting
    past the parity test's tolerance under a busy machine (SN-443).
    """
    core = ov.Core()
    model = core.read_model(Path(ir_path))
    compiled = core.compile_model(model, device, config or {})
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


class OpenVINOTransformerV3Bundle:
    """OpenVINO backend for the v3/v4 dual-branch transformer.

    Same call surface as :class:`TemporalTransformerV3Bundle.predict` so
    it drops into :func:`predict_drone_probs` with no branch. The graph
    emits TWO outputs — ``drone_logits`` ``(B, T)`` and the L2-normalised
    ReID ``embeddings`` ``(B, T, embed_dim)`` — so the learned-association
    arm (Phase 2) can read the same embedding the torch backend produces.
    """

    def __init__(self, compiled: ov.CompiledModel) -> None:
        self._compiled = compiled
        # Output order matches the export wrapper's tuple: (logits, embeddings)
        # + h_track when the graph is stateful (R5). A stateful IR has a 4th
        # input (h_in) and a 3rd output (h_out).
        self._logits = compiled.output(0)
        self._embeddings = compiled.output(1)
        self._stateful = len(compiled.inputs) == 4
        self._state_out = compiled.output(2) if self._stateful else None
        self._state_dim = (
            compiled.inputs[3].get_partial_shape()[1].get_length()
            if self._stateful else 0
        )

    @property
    def stateful(self) -> bool:
        return self._stateful

    def predict(
        self,
        patches: np.ndarray,
        events: np.ndarray,
        event_mask: np.ndarray,
        h_in: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        norm_patches = _normalize_patches(patches)
        events_f = np.ascontiguousarray(events, dtype=np.float32)
        mask_bool = np.ascontiguousarray(event_mask, dtype=np.bool_)
        inputs: list[np.ndarray] = [norm_patches, events_f, mask_bool]
        if self._stateful:
            if h_in is None:
                h_in = np.zeros(
                    (norm_patches.shape[0], self._state_dim), dtype=np.float32,
                )
            inputs.append(np.ascontiguousarray(h_in, dtype=np.float32))
        result = self._compiled(inputs)
        out = {
            "drone_logits": np.asarray(result[self._logits], dtype=np.float32),
            "embeddings": np.asarray(
                result[self._embeddings], dtype=np.float32,
            ),
        }
        if self._state_out is not None:
            out["h_track"] = np.asarray(
                result[self._state_out], dtype=np.float32,
            )
        return out


def load_openvino_transformer_v3(
    ir_path: Path,
    *,
    device: str = "GPU",
    config: dict[str, str] | None = None,
) -> OpenVINOTransformerV3Bundle:
    """Compile a saved v3/v4 IR for ``device``.

    ``device`` = ``GPU`` (Arc 140T), ``CPU``, or ``AUTO``. Raises if the
    target device is unavailable so the caller can fall through to torch.
    ``config`` passes plugin properties straight to ``compile_model`` —
    e.g. the parity test pins ``INFERENCE_PRECISION_HINT: "f32"`` on CPU,
    since that plugin's default there is fp16 and its reduced-precision
    kernel selection is sensitive to system load, occasionally drifting
    past the parity test's tolerance under a busy machine (SN-443).
    """
    core = ov.Core()
    model = core.read_model(Path(ir_path))
    compiled = core.compile_model(model, device, config or {})
    return OpenVINOTransformerV3Bundle(compiled)


def export_v3_to_openvino(
    checkpoint: Path,
    output_dir: Path | None = None,
    *,
    e_max: int | None = None,
) -> Path:
    """Export a v3 or v4 checkpoint to an FP16 OpenVINO IR.

    ``e_max`` overrides the exported event-token count (default = the
    checkpoint's ``cfg.e_max``). The event encoder/attention have no
    fixed-E parameters, so a model trained at 1024 can export a cheaper
    256-token graph — the R5 latency win (fewer tokens → cheaper attention).

    Works for both bidirectional v3 and causal/streaming v4 — same
    model class, the meta sidecar selects the attention path. Exported
    graph: patches ``(B, T, H, W, 3)`` (uint8 or float32), events
    ``(B, E_max, 4)`` float32, event_mask ``(B, E_max)`` bool; returns
    ``drone_logits`` ``(B, T)`` AND the L2-normalised ReID ``embeddings``
    ``(B, T, embed_dim)`` (already computed by the model — the live drone
    path ignores it; the Phase-2 learned-association arm consumes it).
    Batch axis is dynamic.
    """

    import torch
    from torch import nn

    from evio.detection.temporal_transformer_v3 import (
        load_temporal_transformer_v3,
    )

    checkpoint = Path(checkpoint)
    out_dir = Path(output_dir) if output_dir is not None else checkpoint.parent
    bundle = load_temporal_transformer_v3(
        checkpoint, device=torch.device("cpu"),
    )

    cfg = bundle.config
    if getattr(cfg, "conditioning_dim", 0) > 0:
        msg = (
            "conditioning_dim > 0 is not yet threaded through the OpenVINO "
            "export — the IR would silently freeze conditioning at zero, "
            "diverging live inference from offline eval. Export refused."
        )
        raise NotImplementedError(msg)

    class _LogitsAndEmbeddings(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(
            self,
            patches: torch.Tensor,
            events: torch.Tensor,
            event_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.inner(patches, events, event_mask)
            return out["drone_logits"], out["embeddings"]

    class _StatefulWrapper(nn.Module):
        """R5: threads h_track as a 4th input and 3rd output."""

        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(
            self,
            patches: torch.Tensor,
            events: torch.Tensor,
            event_mask: torch.Tensor,
            h_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = self.inner(patches, events, event_mask, h_in)
            return out["drone_logits"], out["embeddings"], out["h_track"]

    emax = e_max or cfg.e_max
    example_patches = torch.zeros(
        1, cfg.seq_len, cfg.patch_size, cfg.patch_size, 3,
    )
    example_events = torch.zeros(1, emax, cfg.event_in_dim)
    example_mask = torch.ones(1, emax, dtype=torch.bool)
    example_inputs: tuple[torch.Tensor, ...] = (
        example_patches, example_events, example_mask,
    )
    reshape: dict[int, ov.PartialShape] = {
        0: ov.PartialShape(
            [-1, cfg.seq_len, cfg.patch_size, cfg.patch_size, 3],
        ),
        1: ov.PartialShape([-1, emax, cfg.event_in_dim]),
        2: ov.PartialShape([-1, emax]),
    }
    if cfg.state_enabled:
        wrapper: nn.Module = _StatefulWrapper(bundle.model).eval()
        example_inputs = (*example_inputs, torch.zeros(1, cfg.state_dim))
        reshape[3] = ov.PartialShape([-1, cfg.state_dim])
    else:
        wrapper = _LogitsAndEmbeddings(bundle.model).eval()
    traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
    ov_model = ov.convert_model(traced, example_input=example_inputs)
    ov_model.reshape(reshape)
    xml_path = out_dir / f"{checkpoint.stem}_ov.xml"
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    return xml_path


__all__ = [
    "OpenVINOTransformerV2Bundle",
    "OpenVINOTransformerV3Bundle",
    "export_v2_to_openvino",
    "export_v3_to_openvino",
    "load_openvino_transformer_v2",
    "load_openvino_transformer_v3",
]
