"""Decision stage: the existing verdict gates over a stream's verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evio.detection.verdict_gate import build_verdict_gate
from evio.dtc.errors import ConfigError
from evio.dtc.manifest import Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import Alarm, Alarms, Evidence, Step, Tracks, Verdicts


@dataclass(frozen=True)
class VerdictGateConfig:
    """Mirrors `build_verdict_gate`; `threshold` is the per-verdict operating
    point. The fusion gate needs `kinematic_llr_path`."""

    kind: str = "mofk"
    threshold: float = 0.5
    k: int = 4
    m: int = 2
    alpha: float = 0.05
    beta: float = 0.1
    temperature: float = 1.0
    leak: float = 0.0
    kinematic_llr_path: str | None = None
    fusion_weight: float = 1.0
    rotor_llr_path: str | None = None
    rotor_weight: float = 1.0
    irstd_llr_path: str | None = None
    irstd_weight: float = 1.0
    fusion_clamp: float | None = 2.0


def _paths(config: VerdictGateConfig) -> dict[str, Path | None]:
    return {
        name: Path(value) if value else None
        for name, value in vars(config).items()
        if name.endswith("_path")
    }


@method(
    "verdict-gate",
    stages=(Stage.DECIDE,),
    config=VerdictGateConfig,
    evidence=("kinematics",),
)
class VerdictGate:
    """Alarm on tracks whose verdicts pass the persistence or SPRT gate.

    Only fresh verdicts (age 0) are fed to the gate, so stale evidence is
    never counted twice; a track keeps alarming while its gate stays fired.
    """

    def __init__(self, config: VerdictGateConfig, context: Context) -> None:
        self.config = config
        self.stream = str(context.stream)
        self.reset()

    def reset(self) -> None:
        config = self.config
        options = {
            name: value
            for name, value in vars(config).items()
            if name not in {"kind", "threshold"}
        }
        try:
            self._gate = build_verdict_gate(
                config.kind, **options | _paths(config)
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
        self._firing: set[str] = set()

    def step(
        self,
        step: Step,
        tracks: Tracks,
        verdicts: Verdicts,
        evidence: Evidence,
    ) -> Alarms:
        features = evidence.get("kinematics", {})
        for verdict in verdicts.items:
            if verdict.age_steps > 0:
                continue
            fired = self._gate.update(
                self.stream,
                verdict.track_id,
                verdict.probability,
                self.config.threshold,
                features=features.get(verdict.track_id),
            )
            (self._firing.add if fired else self._firing.discard)(
                verdict.track_id
            )
        self._gate.prune(self.stream, set(tracks.live_ids))
        self._firing &= tracks.live_ids
        score = getattr(self._gate, "score", None)
        return Alarms(
            tuple(
                Alarm(
                    self.stream,
                    track_id,
                    score(self.stream, track_id) if score else 0.0,
                )
                for track_id in sorted(self._firing)
            )
        )
