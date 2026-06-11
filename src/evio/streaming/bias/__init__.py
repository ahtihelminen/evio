"""Bias autotuner v2 (``plans/bias_autotuner_v2.md``).

A bias write is legal for exactly three reasons — CALIBRATE, REGULATE,
VERIFY — never search. Module map (plan §16):

* ``kernel``     — safety write layer: rails, bias_diff pin, rate limits
* ``estimators`` — support-filter noise split, hot map, censoring, blanking
* ``servo``      — fast suppress-only loop + slow per-polarity budget servo
* ``storm``      — effectiveness gate, no-op detector, STORM-vector jump
* ``regimes``    — profile table, classifier, warm trims, bumpless ramps
* ``trial``      — the one trial engine (descent keeps, probe reverts)
* ``verify``     — dither lock-in map + blink-probe scheduler
* ``controller`` — orchestrator; pure tick with strict actuation arbitration
* ``telemetry``  — status lines, action log, state snapshots
* ``calibrate``  — Mode A balanced-pair servo, kappa estimation, presets
"""

from .controller import BiasControllerV2
from .types import (
    BiasCommand,
    CameraBiasState,
    CameraObservations,
    Polarity,
    ProbeEvidence,
    ProbeRequest,
    ProbeVerdict,
    Reason,
    Regime,
    TickResult,
)

__all__ = [
    "BiasCommand",
    "BiasControllerV2",
    "CameraBiasState",
    "CameraObservations",
    "Polarity",
    "ProbeEvidence",
    "ProbeRequest",
    "ProbeVerdict",
    "Reason",
    "Regime",
    "TickResult",
]
