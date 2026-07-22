"""Shared temporal-transformer drone classifier.

Two consumers score the same v2/v3 checkpoints from different detection
front-ends, so the load + inference logic lives here once:

* :class:`~evio.streaming.live_detection.LiveDetectionEngine` (panorama
  overlay) feeds ``CameraDetectionResult`` patches/events through
  :func:`predict_drone_probs`.
* :class:`TrackTransformerClassifier` (the SAPIENT runner) keeps its own
  per-track patch/event rings — the runner runs a hot-swappable pipeline
  and a configurable tracker that ``CameraDetector`` doesn't model — and
  returns a sticky ``{track_id: drone_prob}`` map so the verdict can ride
  onto each ``Detection.classification`` into SAPIENT.

The transformer is optional at runtime: if torch or the checkpoint is
missing, :func:`load_track_transformer` returns a ``None`` bundle and a
human-readable ``load_warning`` so the caller keeps detecting + tracking
with neutral (unclassified) output.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evio.detection.event_tokens import tokenize_events_in_track
from evio.detection.live_camera import EVENT_MAX, SEQ_LEN
from evio.detection.reid import extract_event_patch

if TYPE_CHECKING:
    from evio.core.events import EventsBatch
    from evio.detection.types import Detection

_log = logging.getLogger(__name__)

_Bbox = tuple[int, int, int, int]


@dataclass(slots=True)
class LoadedTransformer:
    """Outcome of a checkpoint load attempt.

    ``bundle`` is None when torch or the checkpoint is unavailable; the
    caller then runs detector+tracker only and surfaces ``load_warning``.
    """

    bundle: object | None
    is_v3: bool
    is_v4: bool  # v3 with causal mask + streaming-correct per-window verdict
    load_warning: str | None


def load_track_transformer(
    ckpt: Path, device: str = "auto",
) -> LoadedTransformer:
    """Load a v2 or v3 transformer, auto-detecting the version from meta.

    A ``<stem>_meta.json`` sidecar carrying ``e_max`` in its ``config``
    marks a v3 (dual-branch) checkpoint; otherwise it's treated as v2 and
    an OpenVINO IR is preferred on an Intel iGPU when present.
    """
    try:
        import torch

        from evio.detection.temporal_transformer_v2 import (
            load_temporal_transformer_v2,
        )
        from evio.detection.temporal_transformer_v3 import (
            load_temporal_transformer_v3,
        )
    except ImportError as exc:
        warning = (
            f"transformer disabled (torch import failed: {exc}) — "
            "detector+tracker boxes only"
        )
        _log.warning(warning)
        return LoadedTransformer(None, False, False, warning)

    ckpt = Path(ckpt)
    meta_path = ckpt.with_name(f"{ckpt.stem}_meta.json")
    is_v3 = False
    is_v4 = False
    if meta_path.exists():
        try:
            cfg = json.loads(meta_path.read_text()).get("config", {})
            is_v3 = "e_max" in cfg
            is_v4 = bool(cfg.get("causal"))
        except (OSError, ValueError):
            is_v3 = False
            is_v4 = False

    if device in ("auto", "openvino", "gpu"):
        ov_bundle = _try_load_openvino(ckpt, is_v3=is_v3)
        if ov_bundle is not None:
            return LoadedTransformer(ov_bundle, is_v3, is_v4, None)

    try:
        dev = (
            None if device in ("auto", "openvino", "gpu")
            else torch.device(device)
        )
        if is_v3:
            bundle = load_temporal_transformer_v3(ckpt, device=dev)
            variant = "v4 (causal/streaming)" if is_v4 else "v3"
            _log.info("loaded %s transformer from %s", variant, ckpt)
        else:
            bundle = load_temporal_transformer_v2(ckpt, device=dev)
            _log.info("loaded v2 transformer from %s", ckpt)
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        warning = (
            f"transformer disabled (checkpoint load failed: {exc}) — "
            "detector+tracker boxes only"
        )
        _log.warning(warning)
        return LoadedTransformer(None, False, False, warning)
    return LoadedTransformer(bundle, is_v3, is_v4, None)


def _try_load_openvino(ckpt: Path, *, is_v3: bool) -> object | None:
    """Load an Arc iGPU IR for v2/v3/v4, or None to fall back to torch.

    Opt-in: export an IR with ``scripts/export_transformer_openvino.py
    --checkpoint <ckpt>``. Returns None when the IR is absent, openvino
    isn't installed, or no Intel GPU is present — dev machines
    transparently stay on the torch path.
    """
    ir_path = ckpt.with_name(f"{ckpt.stem}_ov.xml")
    if not ir_path.exists():
        return None
    try:
        from evio.detection.openvino_transformer import (
            load_openvino_transformer_v2,
            load_openvino_transformer_v3,
        )
    except ImportError:
        return None
    loader = (
        load_openvino_transformer_v3 if is_v3
        else load_openvino_transformer_v2
    )
    try:
        bundle = loader(ir_path, device="GPU")
    except (RuntimeError, OSError) as exc:
        _log.info("OpenVINO GPU unavailable (%s); using torch", exc)
        return None
    variant = "v3/v4" if is_v3 else "v2"
    _log.info(
        "loaded %s transformer on Arc iGPU (OpenVINO): %s", variant, ir_path,
    )
    return bundle


def predict_drone_probs(
    bundle: object,
    patches: np.ndarray,
    events: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    *,
    is_v4: bool = False,
) -> np.ndarray:
    """Per-track drone probability from one stacked classify batch.

    ``patches`` is ``(B, T, H, W, 3)``; ``events``/``mask`` are the v3
    raw-event branch (omitted on v2). Returns ``(B,)`` sigmoid drone
    probabilities — mean over T for v2/v3, latest-frame logit for v4
    (its causal mask makes `drone_logits[:, -1]` the posterior given
    observations 0..T-1, which is the streaming-correct verdict).
    """
    if events is not None:
        logits = bundle.predict(patches, events, mask)["drone_logits"]
    else:
        logits = bundle.predict(patches)["drone_logits"]
    if is_v4:
        return _sigmoid(logits[:, -1])
    return _sigmoid(logits).mean(axis=1)


class TrackTransformerClassifier:
    """Owns the transformer bundle + per-track patch/event rings.

    Stateful across windows: each :meth:`classify` call appends this
    window's patch (and v3 event context) to the per-track ring and, for
    tracks whose ring just filled, runs the transformer and remembers the
    verdict. Returns the sticky verdict for every currently-tracked id so
    a track keeps its last label between the 16-window classify cadences.
    Rings are keyed by ``(serial, track_id)`` so several cameras share one
    bundle without id collisions.
    """

    def __init__(
        self,
        ckpt: Path,
        *,
        device: str = "auto",
        window_us: int = 10_000,
        stale_ticks: int = 300,
        classify_cadence: int = 1,
    ) -> None:
        loaded = load_track_transformer(ckpt, device)
        self._bundle = loaded.bundle
        self._is_v3 = loaded.is_v3
        self._is_v4 = loaded.is_v4
        self.load_warning = loaded.load_warning
        self._window_us = window_us
        self._stale_ticks = stale_ticks
        # Re-score each mature track at most once per `classify_cadence`
        # windows (verdict held sticky between). 1 = every window (default,
        # unchanged); higher divides the transformer's per-window load so a
        # heavy model (v4's 1024-token event attention) stays real-time at
        # 6-cam on the Arc — E0 Sub-gate A: v4 needs cadence >= ~14.
        self._classify_cadence = max(1, classify_cadence)
        # R5: if the model carries an explicit h_track state, keep a per-track
        # state vector and thread it across this track's scored windows (fed in,
        # updated out). Stateless models (v2/v3/v4) leave this inert.
        self._stateful = getattr(self._bundle, "stateful", None)
        if self._stateful is None:
            cfg = getattr(self._bundle, "config", None)
            self._stateful = bool(getattr(cfg, "state_enabled", False))
        self._state_dim = _state_dim_of(self._bundle) if self._stateful else 0
        self._state: dict[tuple[str, str], np.ndarray] = {}
        self._patch_rings: dict[tuple[str, str], deque[np.ndarray]] = {}
        self._bbox_rings: dict[tuple[str, str], deque[tuple[_Bbox, int]]] = {}
        self._batch_rings: dict[str, deque[EventsBatch]] = {}
        self._verdicts: dict[tuple[str, str], float] = {}
        self._seen_tick: dict[tuple[str, str], int] = {}
        self._scored_tick: dict[tuple[str, str], int] = {}
        self._tick = 0

    @property
    def transformer_loaded(self) -> bool:
        return self._bundle is not None

    @property
    def is_v3(self) -> bool:
        return self._is_v3

    def classify(
        self, serial: str, tracked: list[Detection], batch: EventsBatch,
    ) -> dict[str, float]:
        """Update rings, score ready tracks, return sticky verdicts."""
        if self._bundle is None or not tracked:
            return {}
        self._tick += 1
        if self._is_v3:
            self._batch_rings.setdefault(
                serial, deque(maxlen=SEQ_LEN),
            ).append(batch)
        ts_mid_us = (
            int(batch.timestamps[0]) + int(batch.timestamps[-1])
        ) // 2
        for det in tracked:
            if not det.track_id:
                continue
            key = (serial, det.track_id)
            self._seen_tick[key] = self._tick
            bbox = (det.x0, det.y0, det.x1, det.y1)
            patch = (
                det.patch if det.patch is not None
                else extract_event_patch(batch, bbox)
            )
            self._patch_rings.setdefault(
                key, deque(maxlen=SEQ_LEN),
            ).append(patch)
            if self._is_v3:
                self._bbox_rings.setdefault(
                    key, deque(maxlen=SEQ_LEN),
                ).append((bbox, ts_mid_us))

        self._score_ready(serial, tracked)
        self._prune()
        return {
            det.track_id: self._verdicts[(serial, det.track_id)]
            for det in tracked
            if det.track_id and (serial, det.track_id) in self._verdicts
        }

    def _score_ready(
        self, serial: str, tracked: list[Detection],
    ) -> None:
        ready = [
            (serial, tid)
            for tid in dict.fromkeys(d.track_id for d in tracked if d.track_id)
            if len(self._patch_rings.get((serial, tid), ())) == SEQ_LEN
            and self._is_due((serial, tid))
        ]
        if not ready:
            return
        patches = np.stack(
            [np.stack(tuple(self._patch_rings[k]), axis=0) for k in ready],
            axis=0,
        ).astype(np.uint8)
        events = mask = None
        if self._is_v3:
            events, mask = self._build_event_batch(serial, ready)
            if events is None:
                return  # event history still warming up; skip this tick
        if self._stateful:
            probs = self._score_stateful(ready, patches, events, mask)
        else:
            probs = predict_drone_probs(
                self._bundle, patches, events, mask, is_v4=self._is_v4,
            )
        for key, prob in zip(ready, probs, strict=True):
            self._verdicts[key] = float(prob)
            self._scored_tick[key] = self._tick

    def _score_stateful(
        self,
        ready: list[tuple[str, str]],
        patches: np.ndarray,
        events: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        """Thread each ready track's h_track through one batched R5 forward."""
        h_in = np.stack(
            [self._state.get(k, np.zeros(self._state_dim, np.float32))
             for k in ready],
            axis=0,
        ).astype(np.float32)
        out = self._bundle.predict(patches, events, mask, h_in=h_in)
        h_out = out["h_track"]
        for i, key in enumerate(ready):
            self._state[key] = h_out[i]
        logits = out["drone_logits"]
        if self._is_v4:
            return _sigmoid(logits[:, -1])
        return _sigmoid(logits).mean(axis=1)

    def _is_due(self, key: tuple[str, str]) -> bool:
        """True if `key` has never been scored or its cadence has elapsed."""
        last = self._scored_tick.get(key)
        return last is None or self._tick - last >= self._classify_cadence

    def _build_event_batch(
        self, serial: str, ready: list[tuple[str, str]],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        ring = self._batch_rings.get(serial)
        if not ring:
            return None, None
        merged = _concat_batches(tuple(ring))
        events = np.zeros((len(ready), EVENT_MAX, 4), dtype=np.float32)
        mask = np.zeros((len(ready), EVENT_MAX), dtype=np.bool_)
        for i, key in enumerate(ready):
            bbox_ring = self._bbox_rings.get(key)
            if bbox_ring is None or len(bbox_ring) != SEQ_LEN:
                continue
            ev, m = tokenize_events_in_track(
                merged,
                bboxes=[b for b, _ in bbox_ring],
                timestamps_us=[t for _, t in bbox_ring],
                window_us=self._window_us,
                e_max=EVENT_MAX,
            )
            events[i] = ev
            mask[i] = m
        return events, mask

    def _prune(self) -> None:
        stale = [
            key for key, tick in self._seen_tick.items()
            if self._tick - tick > self._stale_ticks
        ]
        for key in stale:
            self._seen_tick.pop(key, None)
            self._patch_rings.pop(key, None)
            self._bbox_rings.pop(key, None)
            self._verdicts.pop(key, None)
            self._scored_tick.pop(key, None)
            self._state.pop(key, None)


def _concat_batches(batches: tuple[EventsBatch, ...]) -> EventsBatch:
    if len(batches) == 1:
        return batches[0]
    from evio.core.events import EventsBatch

    return EventsBatch(
        timestamps=np.concatenate([b.timestamps for b in batches]),
        x_coords=np.concatenate([b.x_coords for b in batches]),
        y_coords=np.concatenate([b.y_coords for b in batches]),
        polarities=np.concatenate([b.polarities for b in batches]),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def _state_dim_of(bundle: object) -> int:
    """h_track width from an OV bundle (`_state_dim`) or torch bundle config."""
    dim = getattr(bundle, "_state_dim", None)
    if dim:
        return int(dim)
    cfg = getattr(bundle, "config", None)
    return int(getattr(cfg, "state_dim", 0))


__all__ = [
    "LoadedTransformer",
    "TrackTransformerClassifier",
    "load_track_transformer",
    "predict_drone_probs",
]
