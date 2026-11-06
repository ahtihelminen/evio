"""Stand-in for `evio.detection.types.Detection`, scoped to the five fields
`assemble_features` reads.

Used by `tests/test_detection_kinematic_llr.py` so the fusion-feature
assembly tests don't need a full 14-field `Detection`.
`tests/test_fakes_fidelity_detection.py` runs both through
`assemble_features` with matching values and checks they agree, including
the documented `periodicity_score` edge case ("None = not evaluated",
`evio/detection/types.py`).

Differences from the real device:
- it carries five of the fourteen `Detection` fields, so code that reads
  any other field fails on it where it would work on a real `Detection`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeDet:
    m_coh: float = 0.4
    best_psr: float = 5.0
    event_count: int = 20
    z_score: float = 2.5
    periodicity_score: float | None = None
