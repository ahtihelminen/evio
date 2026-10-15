"""The deployed simplified detection pipeline, as a per-stream detector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evio.detection.grid import GridConfig
from evio.detection.pipeline.dual_background import (
    ClassSpecificConfig,
    DualBackgroundConfig,
)
from evio.detection.pipeline.simplified_pipeline import (
    SimplifiedDetectionPipeline,
    SimplifiedPipelineConfig,
    SimplifiedPipelineResult,
)
from evio.detection.types import Detection
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Cost, Stage
from evio.dtc.protocols import Context
from evio.dtc.registry import method
from evio.dtc.types import (
    OK,
    Candidate,
    Candidates,
    Region,
    State,
    Status,
    Step,
)


@dataclass(frozen=True)
class SimplifiedConfig:
    """Flat keys and deployed defaults of `_build_simplified`.

    `width` and `height` default to the EXTENT fact.
    """

    width: int | None = None
    height: int | None = None
    rows: int = 72
    cols: int = 128
    time_bins: int = 1
    z_threshold: int = 10
    update_threshold: int = 2
    calibration_windows: int = 10
    count_weight: int = 10
    multi_weight: int = 10
    coactivity_weight: int = 100
    count_std_floor: int = 1
    multi_std_floor: int = 1
    alpha: int = 5
    ema_alpha: float | None = None
    model_update_weight: float = 0.1
    calibration_time_us: int = 1_000_000
    random_seed: int = 7


def _pipeline(
    config: SimplifiedConfig, context: Context
) -> SimplifiedDetectionPipeline:
    extent_width, extent_height = context.facts[EXTENT]
    grid = GridConfig(
        width=config.width or extent_width,
        height=config.height or extent_height,
        rows=config.rows,
        cols=config.cols,
        time_bins=config.time_bins,
        window_us=context.step_us,
    )
    class_config = ClassSpecificConfig(
        z_threshold=config.z_threshold,
        update_threshold=config.update_threshold,
        calibration_windows=config.calibration_windows,
        count_weight=config.count_weight,
        multi_weight=config.multi_weight,
        coactivity_weight=config.coactivity_weight,
        count_std_floor=config.count_std_floor,
        multi_std_floor=config.multi_std_floor,
    )
    background = DualBackgroundConfig(
        sky=class_config,
        not_sky=class_config,
        alpha=config.alpha,
        ema_alpha=config.ema_alpha,
        model_update_weight=config.model_update_weight,
    )
    return SimplifiedDetectionPipeline(
        grid,
        SimplifiedPipelineConfig(
            calibration_time_us=config.calibration_time_us,
            random_seed=config.random_seed,
            background=background,
        ),
    )


def _status(result: SimplifiedPipelineResult) -> Status:
    if not result.calibrating:
        return OK
    target = result.class_calibration_target_us
    elapsed = result.class_calibration_elapsed_us
    progress = min(1.0, elapsed / target) if target else 1.0
    return Status(State.CALIBRATING, progress)


@method(
    "simplified",
    stages=(Stage.DETECT,),
    config=SimplifiedConfig,
    requires=(EXTENT,),
    cost=Cost.HEAVY,
)
class Simplified:
    """Per-class background model plus motion coherence, on a coarse grid."""

    def __init__(self, config: SimplifiedConfig, context: Context) -> None:
        self.config = config
        self.context = context
        self.stream = str(context.stream)
        self.reset()

    def reset(self) -> None:
        self.pipeline = _pipeline(self.config, self.context)

    def step(self, step: Step, priors: Sequence[Region]) -> Candidates:
        result = self.pipeline.update(step.windows[self.stream].events)
        items = tuple(
            self._candidate(detection) for detection in result.detections
        )
        return Candidates(items, _status(result))

    def _candidate(self, detection: Detection) -> Candidate:
        return Candidate(
            self.stream,
            Region(
                box=(
                    float(detection.x0),
                    float(detection.y0),
                    float(detection.x1),
                    float(detection.y1),
                ),
                velocity_px_s=(detection.vx_px_s, detection.vy_px_s),
            ),
            detection.m_coh,
            {
                "z_score": detection.z_score,
                "best_psr": detection.best_psr,
                "event_count": float(detection.event_count),
                "bg_class": float(detection.bg_class),
            },
        )
