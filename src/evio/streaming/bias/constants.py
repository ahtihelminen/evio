"""Autotuner v2 constants — values trace to ``plans/bias_autotuner_v2.md`` §14.

Each constant names the experiment that tunes it (see the plan's constants
table). Values here are the design seeds; Mode A/B/C calibration refines them.
"""

from __future__ import annotations

# --- Knob model -------------------------------------------------------------
# Register offset ranges relative to factory trim (verified vs the in-house
# IMX636 driver BiasConfig). ``diff`` is listed for completeness but PINNED:
# the safety kernel rejects every write to it (hard user rule 2026-06-10).
BIAS_RANGES: dict[str, tuple[int, int]] = {
    "fo": (-35, 55),
    "hpf": (0, 120),
    "diff_on": (-85, 140),
    "diff": (-25, 23),
    "diff_off": (-35, 190),
    "refr": (-20, 235),
}
BIAS_NAMES: tuple[str, ...] = tuple(BIAS_RANGES)
PINNED_KNOBS: tuple[str, ...] = ("diff",)

# Hard floors with margin above the -85/-35 rails that caused the original
# field storm. The kernel enforces these regardless of any loop or preset.
HARD_FLOOR_DIFF_ON = -60
HARD_FLOOR_DIFF_OFF = -20

# Preset box: runtime-reachable offsets relative to the regime preset value.
BOX_LO = -12
BOX_HI = 25

# --- Budgets (the "noise level we want", ev/s per camera; plan §16.6) -------
DEFAULT_BUDGETS_EPS: dict[str, float] = {
    "clear_day": 150_000.0,
    "overcast_day": 150_000.0,
    "twilight": 250_000.0,  # IMX636 noise peaks near 3-5 lx on-chip (dusk)
    "rain_clutter": 100_000.0,
    "low_light": 120_000.0,
}
CEIL_FACTOR = 1.5  # fast-loop trigger = CEIL_FACTOR x per-polarity budget
DEADBAND_LN = 0.3  # slow servo acts only when ln(BGT/R_N) exceeds this

# --- Transport / saturation (censoring) -------------------------------------
ERC_CAP_EPS = 3_000_000.0  # static per-camera ERC budget; 6x sits under USB3
R_CEIL_FRACTION = 0.75  # fast-loop delivered-rate trigger vs ERC cap
SAT_DELIVERED_EPS = 6_300_000.0  # 0.7 x the ~9 Mev/s SDK readout rail
ERC_PINNED_FRACTION = 0.95  # delivered within 5% of ERC cap => "ge, unknown"
FLAT_LIE_MEAN_EPS = 8_000_000.0  # SDK readout mean above this ...
FLAT_LIE_MAX_COV = 0.01  # ... with CoV below this over FLAT_LIE_WIN_S = lie
FLAT_LIE_WIN_S = 3.0
CENSOR_CLEAR_S = 60.0  # S_c must be false this long before slow loop acts

# --- Estimators --------------------------------------------------------------
K_SUP = 2  # distinct-pixel supporters required within the window
SUP_RADIUS_PX = 2  # Chebyshev radius (5x5 neighborhood)
SUP_WIN_S = 0.05  # trailing support window
SAME_PIX_RUN = 2  # same-pixel same-polarity run that whitelists slow crossers
SAME_PIX_WIN_S = 0.15
HOT_PX_RATE_EPS = 100.0  # per-pixel rate marking a hot pixel
HOT_EMA_TAU_S = 60.0
HOT_SUSTAIN_S = 10.0
RATE_EMA_TAU_S = 10.0  # main-side R_N/R_S/R_hot EMAs
SUBSAMPLE_ABOVE_EPS = 1_000_000.0  # estimator may 1/8-subsample above this
SUBSAMPLE_FACTOR = 8

# --- Blanking (controller never reads its own actuation transient) ----------
T_BLANK_S: dict[str, float] = {
    "diff_on": 1.5,
    "diff_off": 1.5,
    "fo": 2.0,
    "hpf": 2.0,
    "refr": 0.3,
}

# --- Fast loop ----------------------------------------------------------------
FAST_EVAL_S = 2.0
FAST_WINDOW_S = 3.0
FAST_TRIGGER_STREAK = 2
KAPPA_SEED_REG_PER_DECADE = 12.0  # noise-vs-threshold loop gain seed
STEP_FAST_MAX = 6
STEP_CENSORED_MAX = 2  # fixed-max-step while measurements are censored

# --- Slow loop / trials -------------------------------------------------------
SLOW_MIN_SPACING_S = 60.0
N_IDLE = 5  # fast loop idle evaluations before slow loop may act
K_SLOW_REG_PER_LN = 2.5  # ~half the kappa model gain, conservative
STEP_SLOW_MAX = 2
# Speed ramp: while the largest per-polarity error ln(BGT/R_N) exceeds
# RAMP_ERROR_LN the descent is far below budget, so it takes bigger steps at a
# tighter cadence to reach useful sensitivity in seconds; it tapers back to the
# gentle STEP_SLOW_MAX / SLOW_MIN_SPACING_S as the error closes on the deadband.
# Each ramp step is still a judged, revertible trial, so an overshoot is caught
# by the fast loop and reverted.
RAMP_ERROR_LN = 1.5
STEP_SLOW_MAX_RAMP = 8
SLOW_RAMP_SPACING_S = 12.0
JUDGE_WIN_S = 6.0
SCENE_STATIONARY_WIN_S = 30.0
SCENE_STATIONARY_MAX_COV = 0.5
REVERT_COOLDOWN_S = 120.0
COOLDOWN_MAX_S = 1800.0
MCOH_DEGRADE_RATIO = 0.8  # post/pre coherent metric below this fails a trial

# --- Storm ladder -------------------------------------------------------------
EFFECTIVENESS_MIN = 0.30  # observed/predicted log-rate drop below this ...
EFFECTIVENESS_STRIKES = 2  # ... this many times consecutively => ineffective
NOOP_BLIND_STEPS = 3  # blind steps while censored before the STORM jump
STORM_DEESCALATE_FRACTION = 0.5
STORM_DEESCALATE_S = 120.0

# --- Regimes ------------------------------------------------------------------
NIGHT_LUX = 10.0
DARK_LUX = 50.0
TWILIGHT_SUN_ELEV_DEG = 6.0  # |elevation| below this (and above -6) = twilight
NIGHT_SUN_ELEV_DEG = -6.0
REGIME_STREAK = 2  # classifier hysteresis (proven whipsaw fix)
REGIME_DWELL_S = 300.0
RAMP_MAX_STEP = 2  # bumpless transfer: registers per write interval
POST_RAMP_DESCENT_HOLD_S = 600.0
WARM_TRIM_HALF_LIFE_S = 7 * 24 * 3600.0  # decay toward preset center

# --- Feedforward (expectation trims, never blind walks; <= +-4 registers) ---
THERMAL_REF_C = 35.0
LEAK_DOUBLING_C = 7.0  # warm/day: expected ON noise x2 per this many deg C
DARK_SHOT_LN_SLOPE_PER_C = -0.28  # cold/night OFF noise, sign/order only
FEEDFORWARD_MAX_REG = 4

# --- Asymmetry (NIGHT_ASYM; flag stays off until Mode B validates) -----------
ASYM_SENSITIVE_BUDGET_FRACTION = 0.75
ASYM_INSENSITIVE_BUDGET_FRACTION = 0.1
ASYM_SENSITIVE_MAX_BELOW_BALANCE = 15
ASYM_INSENSITIVE_MAX_ABOVE_BALANCE = 25

# --- Verification -------------------------------------------------------------
PROBE_STEP = 6  # suppress-side raise on one polarity
PROBE_HOLD_S = 2.0
PROBE_RATION_S = 60.0  # at most one probe per camera per this interval
PROBE_SURVIVAL_CONTRAST = 0.6  # candidate keeps >= this of its rate => CONTRAST
PROBE_SURVIVAL_NOISE = 0.25  # at/below the predicted noise drop => NOISE
DITHER_AMPLITUDE_REG = 2
DITHER_HOLD_S = 30.0
LOCKIN_MIN_CYCLES = 8  # PRBS cycles before the sensitivity map is conclusive
LOCKIN_FLAT_NAT_PER_REG = 0.05  # |dlnR/dtheta| below this = contrast-driven
LOCKIN_STEEP_NAT_PER_REG = 0.12  # above this = noise-dominated

# --- Write hygiene ------------------------------------------------------------
WRITE_MIN_SPACING_S = 2.0  # per knob per camera (the 2 s cap; never exempt)
WRITE_BUDGET_REG = 12  # rate-of-change cap ...
WRITE_BUDGET_WIN_S = 600.0  # ... per knob per this window (ladder/ramps exempt)
SETTLE_S = 0.1  # register write settle (I2C-class transaction)

# --- Misc ---------------------------------------------------------------
ZENITH_SERIAL = "4108810810"  # mono zenith cam: excluded from bias control
PROBE_REASON_VERSION = 1  # bump when probe semantics change (provenance)
