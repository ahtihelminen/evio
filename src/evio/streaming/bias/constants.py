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

# Hard floors = the hardware rails (Elia 2026-07-03: "go all the way to the
# hardware min/max — don't leave performance on the table"). The original -60
# / -20 static margins dated from the pre-trial-engine field storm (e26967a
# era, suppress-only controller, no masking, lossy transport); today every
# descent step is a judged revertible trial behind the R_N floors, the budget
# servo, fast suppress and the storm ladder — the DYNAMIC guards are the
# protection, and the static margin was measurably the binding constraint
# (fleet railed at the box floor with rn 10-100x under budget, rig 2026-07-03).
HARD_FLOOR_DIFF_ON = -85
HARD_FLOOR_DIFF_OFF = -35
# Scripted bias sweeps (edge bias_control) run with the storm ladder and every
# runtime guard SUSPENDED, so the dynamic-guards argument above does not apply
# there: sweeps keep the old static margins. A below-margin sweep needs an
# explicit human decision, not an inherited constant.
SWEEP_FLOOR_DIFF_ON = -60
SWEEP_FLOOR_DIFF_OFF = -20
# fo sweep floor (2026-07-03 incident): a scripted/ramped fo staircase to -35
# on a lit scene collapses the photoreceptor bandwidth 3-4 decades — the
# camera is effectively blind. Response data: collapse became total ~fo<=-20
# on a clear-day scene. Sweeps below this need an explicit per-knob spec
# override (SweepBounds.allow_below_floor), never a default.
SWEEP_FLOOR_FO = -20
# Min-rate (blindness) abort for sweep runners: whole-frame delivered below
# this for N consecutive judged steps aborts + restores pre-sweep registers.
# Distinct from censored-abort (too HIGH): silence must never judge happily.
SWEEP_MIN_RATE_EPS = 2_000.0
SWEEP_MIN_RATE_STEPS = 2

# --- Rate-collapse (blindness) watchdog — controller-side G2 -----------------
# After ANY applied write, if delivered collapses below RATE_COLLAPSE_FRACTION
# of the pre-write-burst baseline (a >=30 s-quiet median, frozen while writes
# are in flight so a staircase can't walk the baseline down with itself) and
# stays there RATE_COLLAPSE_HOLD_S, the controller alarms + reverts to the
# last SIGHTED register state. Keyed RELATIVE to the pre-write rate so the
# legit quiet-sky descent (R_N low, R_hot ~0) is never fought; inert when the
# baseline itself is below RATE_COLLAPSE_BASELINE_MIN_EPS (a genuinely dark
# night scene has nothing for the watchdog to protect — and low-side lies
# are indistinguishable from night there).
RATE_COLLAPSE_FRACTION = 0.01
RATE_COLLAPSE_HOLD_S = 10.0
RATE_COLLAPSE_BASELINE_MIN_EPS = 1_000.0
RATE_COLLAPSE_BASELINE_QUIET_S = 30.0
RATE_COLLAPSE_COOLDOWN_S = 60.0

# Preset box: runtime-reachable offsets relative to the regime preset value —
# applies to the calibration knobs (fo / refr / hpf). The servoed threshold
# pair is deliberately NOT box-bound: it may use the full hardware range.
BOX_LO = -12
BOX_HI = 25
FULL_RANGE_KNOBS: tuple[str, ...] = ("diff_on", "diff_off")

# --- Budgets (the "noise level we want", ev/s per camera; plan §16.6) -------
# Re-based 2026-07-03 (Elia): every earlier value was chosen while the
# transport silently dropped ~98% of events (rates read ~50x low). Day = 300k
# total/cam on lossless capture; other regimes keep their pre-fix ratios.
DEFAULT_BUDGETS_EPS: dict[str, float] = {
    "clear_day": 300_000.0,    # total per camera; per-polarity = /2 = 150k
    "overcast_day": 300_000.0,
    "twilight": 500_000.0,  # IMX636 noise peaks near 3-5 lx on-chip (dusk)
    "rain_clutter": 200_000.0,
    "low_light": 250_000.0,
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

# --- Stuck pixels: cold-start fast-flag + DEM-mask detection ----------------
# A stuck oscillator is a far hotter, far more pathological thing than the
# 100 ev/s noise-pool hot pixel (HOT_PX_RATE_EPS): the field cameras show
# 1-10 fixed pixels at 54-400 kHz that are 99.8-99.9% of every event. Two
# thresholds, kept separate from the noise-pool hot floor:
#  * FAST_STUCK_RATE_EPS — an *instantaneous* per-pixel rate this high (no
#    real sub-pixel drone sustains 20 kHz on ONE pixel) is excluded from the
#    noise pool after only FAST_STUCK_SUSTAIN_S, shrinking the cold-start
#    suppress transient (the t<HOT_SUSTAIN_S FastSuppressor clamp).
#  * STUCK_MASK_EPS — the per-pixel hot-EMA floor (sustained STUCK_SUSTAIN_S)
#    that nominates a pixel as a hardware DEM-mask candidate. The discriminator
#    that makes auto-masking safe is polarity purity: a real stuck pixel locks
#    to ONE polarity (operator profiling: 99.8-99.9% single-polarity), while
#    any real signal — a hovering glint, a faint mover, a rotor — mixes
#    polarity and stays at/below 99%. So nomination also requires spatial
#    isolation AND single-polarity > STUCK_POLARITY_FRACTION, and that gate is
#    sufficient to mask directly — no periodicity / live-track veto is needed.
FAST_STUCK_RATE_EPS = 20_000.0
FAST_STUCK_SUSTAIN_S = 1.0
STUCK_MASK_EPS = 2_000.0  # scan-tool / provenance floor (static-scene P0 scan)
STUCK_SUSTAIN_S = 10.0
# Live-masking rate floor is RELATIVE to the ambient per-pixel noise, so it
# tracks the autotuner's operating point: a stuck pixel the autotuner has
# throttled (high refractory) still stands out as a >>ambient outlier, and a
# busy scene raises the floor so ordinary noise is never masked. Nominate iff
# the pixel's hot-EMA exceeds max(STUCK_MASK_EPS_MIN, STUCK_AMBIENT_FACTOR x
# ambient_per_pixel_eps). PROVENANCE: the floor was briefly 300 because
# opsnuc-r 2026-07-01 showed throttled stuck pixels at "~1 kev/s" — a PRE-FIX
# reading taken while the transport silently dropped ~98% of events (true
# rates ~50x higher, comfortably above 2 kev/s). Under lossless capture a
# 300 ev/s floor is reachable by genuine scene pixels, so it is back at the
# validated 2 kev/s; the relative 30x-ambient term self-corrects on top.
STUCK_MASK_EPS_MIN = 2_000.0
STUCK_AMBIENT_FACTOR = 30.0
# A pixel is stuck iff strictly MORE than this share of its events are the
# dominant polarity (operator rule: "99% or lower is not stuck, above is").
STUCK_POLARITY_FRACTION = 0.99
STUCK_MAX_CANDIDATES = 64  # cap the per-window candidate report (= DEM slots)

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
# Descent backstop: halt the servo from opening biases when the genuine
# (hot-excluded) noise R_N has collapsed below the regime floor *because the
# rate is concentrated in hot-excluded oscillators* (R_hot above the floor) —
# opening further only blooms stuck pixels, not signal. This is NOT a "quiet
# sky" brake: a genuinely clean low-noise sky (R_N low, R_hot ~0) must STILL
# descend to maximum sensitivity for far / single-pixel drones (the speed
# ramp). Mode-C re-tunes these per camera from the clean per-pixel noise
# floor (clean cam 4110038334 is the cross-check).
# Scaled in lockstep with the 2026-07-03 budget re-base (floor/budget
# ratio preserved per regime).
R_N_DESCENT_FLOOR_EPS: dict[str, float] = {
    "clear_day": 8_000.0,
    "overcast_day": 8_000.0,
    "twilight": 8_000.0,
    "rain_clutter": 10_000.0,
    "low_light": 4_000.0,
}
JUDGE_WIN_S = 6.0
SCENE_STATIONARY_WIN_S = 30.0
SCENE_STATIONARY_MAX_COV = 0.5
# Below this mean delivered rate the CoV test is waived (trivially
# stationary). A relative-variance brake is systematically harsher on quiet
# scenes: at a few hundred ev/s one bird/gust doubles the rate and blows
# CoV>0.5, freezing descent on exactly the cameras with the most headroom
# (post-masking rig, 2026-07-02: 4 of 6 cams frozen at stationary:0 on
# 100-700 ev/s genuine scene). At this scale there is no activity for the
# brake to protect a descent trial from -- real judgment-corrupting scene
# change shows up in kev/s, and a crossing target is covered by the
# no-tracks-nearby gate.
SCENE_STATIONARY_SPARSE_EPS = 5_000.0
REVERT_COOLDOWN_S = 120.0
COOLDOWN_MAX_S = 1800.0
MCOH_DEGRADE_RATIO = 0.8  # post/pre coherent metric below this fails a trial

# --- Storm ladder -------------------------------------------------------------
EFFECTIVENESS_MIN = 0.30  # observed/predicted log-rate drop below this ...
EFFECTIVENESS_STRIKES = 2  # ... this many times consecutively => ineffective
NOOP_BLIND_STEPS = 3  # blind steps while censored before the STORM jump
STORM_DEESCALATE_FRACTION = 0.5
STORM_DEESCALATE_S = 120.0

# --- Stuck-pixel DEM registry (main-side policy) ----------------------------
# The registry owns promote / active-demote-probe / slot allocation. DEM has
# 64 per-pixel slots (imx636.PIXEL_MASK_SLOTS); the allocator dedupes (x, y)
# and asserts the cap (the driver does not detect double-booking).
DEM_MAX_SLOTS = 64
STUCK_PROMOTE_TICKS = 2  # consecutive candidate ticks before a slot is taken
STUCK_PROBE_PERIOD_S = 120.0  # cadence of the active unmask-and-observe probe
# A masked pixel emits nothing (DEM-zeroed), so healing is learned by briefly
# unmasking and watching. The probe window must outlast the candidate
# re-detection latency (the worker needs ~STUCK_SUSTAIN_S to re-nominate a
# still-stuck pixel), then the slot is re-masked regardless — it is never left
# open indefinitely. A pixel that has not re-nominated for STUCK_UNSTICK_S of
# masked life (confirmed by a clean probe) is released.
STUCK_PROBE_WINDOW_S = 15.0
STUCK_UNSTICK_S = 300.0
# Probe backoff (rig 2026-07-03): with 40-64 long-lived oscillators per
# camera, the fixed 120 s cadence re-admitted every flooder at 12.5% duty —
# ~100-200 kev/s of standing fake noise that consumed the OFF budget (rn_off
# read 88-236k on a 25-56 kev/s scene), so the servo balanced diff_off against
# the probe cycle instead of the sky. Each consecutive FAILED probe (the pixel
# re-fired) doubles that pixel's probe period up to PROBE_BACKOFF_MAX_S; after
# PROBE_FAILS_PERMANENT consecutive failures the pixel is PERMANENT and never
# runtime-probed again — the boot-time rescan (release-all + re-detect at
# every camera open) is the healing path for permanents. A clean probe resets
# the failure count. Probes also open at most ONE at a time per camera:
# from_config seeded every slot's timer to the same instant, so previously all
# 64 probes opened simultaneously (the whole flood back for 15 s every 120 s).
PROBE_BACKOFF_MAX_S = 4 * 3600.0
PROBE_FAILS_PERMANENT = 5
# Probe verdicts are judged on the DIRECTLY OBSERVED event rate at the probed
# coordinate (the worker counts events at probation pixels and ships them in
# PolaritySignals.probation_counts) — NEVER on re-nomination: a masked pixel's
# 60 s hot-EMA decays toward zero, so re-nomination inside the 15 s blink is
# physically impossible below ~25 kev/s (floor re-cross + 10 s sustain > 15 s)
# and a still-stuck throttled pixel would read "healed" on every backed-off
# probe (2026-07-04 adversarial review, 2 confirmed findings). Observed rate
# above this refires the probe (counts as a failure); a blink with NO
# probation report at all is INCONCLUSIVE — re-mask, keep the failure count,
# never release. Half the absolute stuck floor: err toward keeping a mask
# (cheap) over releasing a flooder (expensive).
PROBE_REFIRE_EPS = 1_000.0
# Stuck pixels beyond the 64 hardware DEM slots get a SOFTWARE mask: the
# worker drops their events after recording (footage stays a lossless record
# of what the sensor delivered) and before detection / render / the noise
# estimator. The cap is a sanity bound — the nomination triad (sustained fixed
# coordinate + >99% single-polarity + not-under-a-track-box) keeps diffuse
# storms out by construction. A software mask cannot save ERC/USB transport
# budget (the events still cross the wire), so hardware slots are preferred:
# when a DEM slot frees, the longest-lived software mask is promoted into it.
SOFT_MASK_MAX = 1024

# --- Regimes ------------------------------------------------------------------
NIGHT_LUX = 10.0
DARK_LUX = 50.0
TWILIGHT_SUN_ELEV_DEG = 6.0  # |elevation| below this (and above -6) = twilight
NIGHT_SUN_ELEV_DEG = -6.0
# Sun at/above this = day, PERIOD: the lux-keyed twilight/low_light branches
# are unreachable (the on-chip lux readout can read <50 lx through the optics
# on an overcast afternoon and whipsawed the fleet into twilight mid-day).
# Additionally the lux-keyed LOW_LIGHT vote requires sun corroboration (sun
# unknown or inside the twilight cone) at ANY elevation — the (6, 10] band and
# the ~60N winter (max elevation < 10 all day) otherwise leave the flaky-lux
# night vote armed in broad daylight (2026-07-04 review).
DAY_SUN_ELEV_DEG = 10.0
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
# Sized so the descent speed ramp is not defeated by its own budget: the ramp
# takes up to STEP_SLOW_MAX_RAMP=8 registers per SLOW_RAMP_SPACING_S=12 s, so
# the old 12-reg budget throttled it after ~1.5 steps per 10 min (rig
# 2026-07-03: 129-186 write_budget rejections on the ramping cameras). 48
# allows ~6 ramp steps per window while still capping a pathological
# regulate-loop; every step remains a judged, revertible trial.
WRITE_BUDGET_REG = 48  # rate-of-change cap ...
WRITE_BUDGET_WIN_S = 600.0  # ... per knob per this window (ladder/ramps exempt)
SETTLE_S = 0.1  # register write settle (I2C-class transaction)

# --- Misc ---------------------------------------------------------------
ZENITH_SERIAL = "4108810810"  # mono zenith cam: excluded from bias control
PROBE_REASON_VERSION = 1  # bump when probe semantics change (provenance)
