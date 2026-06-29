"""Autotuner v2 shared types — the contract between bias modules.

Design rules these types encode (``plans/bias_autotuner_v2.md`` §15-16):

* A bias write is legal for exactly three reasons — CALIBRATE, REGULATE,
  VERIFY — and every :class:`BiasCommand` carries its reason.
* The controller core is a pure function ``(state, observations) ->
  (state', commands)``; every field of :class:`CameraBiasState` is plain
  data so a tick can be serialized and replayed offline.
* Time is always a caller-supplied monotonic ``now`` (seconds). No module
  in this package reads the clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, StrEnum, auto


class Polarity(StrEnum):
    ON = "on"
    OFF = "off"


class Reason(StrEnum):
    """Why a bias write happened. The action log is the philosophy, audited."""

    CALIBRATE = "calibrate"
    REGULATE_UP = "regulate_up"  # fast suppress / storm-ladder step
    REGULATE_DOWN = "regulate_down"  # slow-servo descent trial
    RAMP = "ramp"  # bumpless regime transfer
    STORM = "storm"  # jump to the regime's vetted STORM vector
    PROBE = "probe"  # blink-probe raise (always reverted)
    REVERT = "revert"  # trial/probe revert to pre-state
    DITHER = "dither"  # PRBS dither around the setpoint
    STUCK_MASK = "stuck_mask"  # DEM hardware pixel mask promote/probe/demote


class Regime(StrEnum):
    CLEAR_DAY = "clear_day"
    OVERCAST_DAY = "overcast_day"
    RAIN_CLUTTER = "rain_clutter"
    TWILIGHT = "twilight"
    LOW_LIGHT = "low_light"


class CensorCause(IntFlag):
    """Why measurements are distrusted (S_c). EMAs HOLD while any bit is set."""

    NONE = 0
    DELIVERED_HIGH = auto()
    ERC_PINNED = auto()
    BUS_GOVERNOR = auto()
    FLAT_LIE = auto()


class TrialIntent(StrEnum):
    DESCENT = "descent"  # REGULATE_DOWN: keep if better, else revert
    PROBE = "probe"  # VERIFY: always revert, emit ProbeEvidence


class ProbeVerdict(StrEnum):
    CONTRAST = "contrast"  # survived the raise: contrast-driven (real mover)
    NOISE = "noise"  # died as the exponential noise model predicted
    CLUMP = "clump"  # ON-only signature: leak / hot-pixel clump
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class BiasCommand:
    """One vetted register write. ``value`` is the absolute offset to set."""

    serial: str
    knob: str
    value: int
    reason: Reason
    trial_id: str | None = None


@dataclass(frozen=True)
class PolaritySignals:
    """Worker-side estimator output for one event window (one camera).

    Counts are post-subsample, scaled back to full-stream estimates.
    ``pair_fraction`` = share of same-pixel opposite-polarity short-ISI
    pairs (rho_p) — the live signature of the shot-noise pair cycle.
    """

    window_s: float
    passed_on: float
    passed_off: float
    rejected_on: float
    rejected_off: float
    hot_on: float
    hot_off: float
    pair_fraction: float
    active_pixels: int
    subsample_factor: int = 1
    # Path-A DEM-mask nominations this window: (x, y, dominant_polarity)
    # with dominant_polarity 1=ON, 0=OFF. Already gated worker-side on
    # rate + single-polarity + spatial isolation + not-in-a-track-box.
    stuck_candidates: tuple[tuple[int, int, int], ...] = ()

    def rejected_eps(self, polarity: Polarity) -> float:
        n = self.rejected_on if polarity is Polarity.ON else self.rejected_off
        return n / self.window_s if self.window_s > 0 else 0.0

    def passed_eps(self, polarity: Polarity) -> float:
        n = self.passed_on if polarity is Polarity.ON else self.passed_off
        return n / self.window_s if self.window_s > 0 else 0.0


@dataclass(frozen=True)
class EnvSignals:
    """From IMX636Camera.read_env(); any field may be None (sensor timeout)."""

    lux: float | None = None
    die_temp_c: float | None = None
    pixel_dead_time_us: float | None = None


@dataclass(frozen=True)
class DetectionSignals:
    """What the detection stack tells the controller each tick."""

    n_candidates: int = 0
    n_drone_tracks: int = 0  # confirmed tracks on this camera
    max_prob: float = 0.0  # best live transformer verdict
    coherent_metric: float = 0.0  # m_coh
    n_periodic: int = 0  # SN-31 comb-confirmed tracks


@dataclass(frozen=True)
class ProbeRequest:
    """A borderline candidate asking to be verified (VERIFY trigger)."""

    serial: str
    request_id: str
    cell_x: int
    cell_y: int
    local_rate_eps: float  # candidate-local event rate before the probe
    suspected_polarity: Polarity | None = None  # ON-dominant => clump suspect


@dataclass(frozen=True)
class ProbeEvidence:
    request_id: str
    verdict: ProbeVerdict
    survival_ratio: float  # observed post/pre candidate-local rate
    predicted_noise_ratio: float  # what the exponential model predicted
    probed_polarity: Polarity


@dataclass(frozen=True)
class CameraObservations:
    """Everything controller.tick() consumes for one camera, one tick."""

    serial: str
    signals: PolaritySignals | None = None
    delivered_eps: float = 0.0  # host-counted, post-ERC
    sdk_rate_eps: float = 0.0  # readout claim (rail-lies above ~9 Mev/s)
    erc_drop_fraction: float = 0.0
    bus_governor_active: bool = False
    env: EnvSignals = field(default_factory=EnvSignals)
    detection: DetectionSignals = field(default_factory=DetectionSignals)
    probe_requests: tuple[ProbeRequest, ...] = ()


@dataclass
class PresetBox:
    """One hand-vetted operating box: camera x regime (plan §8, §16.6)."""

    regime: Regime
    diff_on: int  # preset center (balance pair from Mode A / hand seed)
    diff_off: int
    fo: int = 0
    hpf: int = 0
    refr: int = 0
    afk_enabled: bool = False  # AFK OFF outdoors by default (SN-31)
    erc_cap_eps: float = 3_000_000.0
    budget_on_eps: float = 75_000.0  # per-polarity = BGT_cam / 2
    budget_off_eps: float = 75_000.0
    storm_diff_on: int = 40  # vetted heavy-suppression jump point
    storm_diff_off: int = 60
    asym_allowed: bool = False

    def center(self, knob: str) -> int:
        return int(getattr(self, knob))

    def lo(self, knob: str) -> int:
        from .constants import BOX_LO

        return self.center(knob) + BOX_LO

    def hi(self, knob: str) -> int:
        from .constants import BOX_HI

        return self.center(knob) + BOX_HI


@dataclass
class Trial:
    """One in-flight write->blank->judge->resolve cycle (descent OR probe)."""

    trial_id: str
    serial: str
    intent: TrialIntent
    knob: str
    pre_value: int
    new_value: int
    started_at: float
    blank_until: float
    judge_until: float
    baseline_noise_eps: dict[str, float] = field(default_factory=dict)
    baseline_coherent: float = 0.0
    probe: ProbeRequest | None = None
    baseline_local_rate_eps: float = 0.0


@dataclass
class FastState:
    violation_streak: dict[str, int] = field(
        default_factory=lambda: {"on": 0, "off": 0, "total": 0}
    )
    last_eval_at: float = 0.0
    idle_evals: int = 0  # consecutive evaluations with no violation/write
    last_write_at: float = 0.0


@dataclass
class SlowState:
    cooldown_until: float = 0.0
    consecutive_fails: int = 0
    last_trial_at: float = 0.0


@dataclass
class StormState:
    active: bool = False
    effectiveness_strikes: int = 0
    blind_steps: int = 0
    frozen: bool = False  # thresholds declared ineffective; containment mode
    jumped: bool = False  # STORM vector applied
    degraded: bool = False  # camera flagged DEGRADED for fusion/SAPIENT
    below_ceiling_since: float | None = None
    predicted_drop_ln: float = 0.0  # expectation for the step under judgment
    pre_step_noise_eps: float = 0.0


@dataclass
class DitherState:
    enabled: bool = False
    sequence_pos: int = 0
    current_offset: int = 0
    next_flip_at: float = 0.0


@dataclass
class RampState:
    active: bool = False
    target: dict[str, int] = field(default_factory=dict)
    descent_hold_until: float = 0.0


@dataclass
class EstimatorState:
    """Main-side EMA bank + censoring for one camera."""

    r_n_eps: dict[str, float] = field(
        default_factory=lambda: {"on": 0.0, "off": 0.0}
    )
    r_s_eps: dict[str, float] = field(
        default_factory=lambda: {"on": 0.0, "off": 0.0}
    )
    r_hot_eps: dict[str, float] = field(
        default_factory=lambda: {"on": 0.0, "off": 0.0}
    )
    pair_fraction: float = 0.0
    censor: CensorCause = CensorCause.NONE
    censor_clear_at: float = 0.0  # when S_c last became false
    blank_until: float = 0.0
    sdk_rate_history: list[tuple[float, float]] = field(default_factory=list)
    last_update_at: float = 0.0

    @property
    def censored(self) -> bool:
        return self.censor != CensorCause.NONE


@dataclass
class CameraBiasState:
    """Complete per-camera controller state — serialized every tick."""

    serial: str
    registers: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("fo", "hpf", "diff_on", "diff", "diff_off", "refr"), 0
        )
    )
    regime: Regime = Regime.CLEAR_DAY
    candidate_regime: Regime | None = None
    regime_streak: int = 0
    regime_entered_at: float = 0.0
    estimator: EstimatorState = field(default_factory=EstimatorState)
    fast: FastState = field(default_factory=FastState)
    slow: SlowState = field(default_factory=SlowState)
    storm: StormState = field(default_factory=StormState)
    trial: Trial | None = None
    dither: DitherState = field(default_factory=DitherState)
    ramp: RampState = field(default_factory=RampState)
    kappa_reg_per_decade: float = 12.0  # per-camera loop gain (Mode A refines)
    last_probe_at: float = 0.0
    write_log: list[tuple[float, str, int]] = field(default_factory=list)
    manual_until: float = 0.0  # MANUAL override TTL; controller holds off


@dataclass
class TickResult:
    """What one controller tick produced."""

    commands: list[BiasCommand] = field(default_factory=list)
    evidence: list[ProbeEvidence] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)  # action-log rows
    status_lines: list[str] = field(default_factory=list)
    # (serial, ("pixel_mask", slot, x, y, enabled)) DEM control messages.
    # NOT BiasCommands — they never flow through the kernel or the
    # ("bias", ...) worker path.
    pixel_mask_msgs: list[tuple[str, tuple]] = field(default_factory=list)
