"""Stand-in for a stateful (R5) transformer bundle.

`TrackTransformerClassifier._score_stateful` (`evio/detection/
track_classifier.py`) threads a per-track state vector through
`bundle.predict(patches, events, mask, h_in=h_in)`, reading back
`out["h_track"]` and `out["drone_logits"]`. Used by
`tests/test_detection_track_classifier.py` to pin that threading without
a compiled OpenVINO IR or a torch checkpoint.

`tests/test_fakes_fidelity_detection.py` checks the shape contract the
real bundles document (`evio/detection/openvino_transformer.py`,
`evio/detection/temporal_transformer_v3.py`): `h_track` rides out at the
same shape `h_in` rode in at, one row per input track.

Differences from the real device:
- only the shape contract is shared: the logits and the state it returns
  are not a trained model's, so no score from it means anything.
"""

from __future__ import annotations

import numpy as np

from evio.detection.track_classifier import SEQ_LEN


class FakeStatefulBundle:
    """Records the h_in fed each predict; returns h_out = h_in + 1."""

    stateful = True
    _state_dim = 8

    def __init__(self) -> None:
        self.seen_h: list[np.ndarray] = []

    def predict(
        self, patches: np.ndarray, events: object, mask: object,
        h_in: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.seen_h.append(np.asarray(h_in).copy())
        n = patches.shape[0]
        return {
            "drone_logits": np.full((n, SEQ_LEN), 2.0, dtype=np.float32),
            "h_track": (h_in + 1.0).astype(np.float32),
        }
