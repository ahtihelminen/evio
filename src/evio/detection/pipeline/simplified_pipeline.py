"""Simplified detection pipeline with classification and motion coherence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from evio.core.events import EventsBatch
from evio.detection.background.classes import (
    BackgroundClassesClassifier,
    BgClassesResult,
)
from evio.detection.grid import GridConfig
from evio.detection.pipeline.dual_background import (
    DualBackgroundConfig,
    DualBackgroundResult,
    DualBackgroundScorer,
)
from evio.detection.pipeline.simplified_tbd import (
    SimplifiedTbdConfig,
    SimplifiedTbdResult,
    SimplifiedTbdStage,
)
from evio.detection.types import Detection

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class SimplifiedPipelineConfig:
    """Configuration for the simplified detection pipeline."""

    calibration_time_us: int = 100_000
    random_seed: int = 7
    background: DualBackgroundConfig = field(
        default_factory=DualBackgroundConfig
    )
    tbd: SimplifiedTbdConfig = field(default_factory=SimplifiedTbdConfig)

    def __post_init__(self) -> None:
        if self.calibration_time_us < 0:
            msg = "calibration_time_us must be >= 0"
            raise ValueError(msg)


@dataclass(slots=True)
class SimplifiedPipelineResult:
    """Result from the simplified detection pipeline."""

    window_index: int
    calibrating: bool
    class_map: IntArray | None
    z_map: FloatArray
    detections: list[Detection]
    class_calibration_elapsed_us: int
    class_calibration_target_us: int
    bg_result: DualBackgroundResult | None = None


class SimplifiedDetectionPipeline:
    """Three-stage detection pipeline with per-class background modeling.

    Stage 1: Binary background classification (sky vs not-sky)
    Stage 2: Per-class background abnormality scoring
    Stage 3: Motion coherence validation for candidates
    """

    def __init__(
        self,
        grid: GridConfig,
        config: SimplifiedPipelineConfig,
    ) -> None:
        self._grid = grid
        self._config = config

        self._classifier = BackgroundClassesClassifier(
            grid=grid,
            calibration_time_us=config.calibration_time_us,
            random_seed=config.random_seed,
        )
        self._background = DualBackgroundScorer(grid, config.background)
        self._tbd = SimplifiedTbdStage(grid, config.tbd)

        self._window_index = 0

    @property
    def window_index(self) -> int:
        return self._window_index

    @property
    def grid(self) -> GridConfig:
        return self._grid

    def update(self, batch: EventsBatch) -> SimplifiedPipelineResult:
        """Process an event batch through all pipeline stages."""
        class_result = self._classifier.update(batch)

        bg_result: DualBackgroundResult | None = None
        if class_result.class_map is None:
            z_map = np.zeros(
                (self._grid.rows, self._grid.cols), dtype=np.float32
            )
            detections: list[Detection] = []
            calibrating = True
        else:
            bg_result = self._background.update(batch, class_result.class_map)
            tbd_result = self._update_tbd(batch, bg_result, class_result)
            z_map = bg_result.z_map
            detections = tbd_result.detections
            calibrating = class_result.calibrating or bg_result.calibrating

        result = SimplifiedPipelineResult(
            window_index=self._window_index,
            calibrating=calibrating,
            class_map=class_result.class_map,
            z_map=z_map,
            detections=detections,
            class_calibration_elapsed_us=class_result.calibration_elapsed_us,
            class_calibration_target_us=class_result.calibration_target_us,
            bg_result=bg_result,
        )
        self._window_index += 1
        return result

    def _update_tbd(
        self,
        batch: EventsBatch,
        bg_result: DualBackgroundResult,
        class_result: BgClassesResult,
    ) -> SimplifiedTbdResult:
        """Update TBD stage with per-class thresholds."""
        if class_result.class_map is None:
            return SimplifiedTbdResult(
                window_index=self._window_index,
                integration_span_us=0,
                buffered_event_count=0,
                baseline_mean=0.0,
                baseline_var=0.0,
                baseline_count=0,
                detections=[],
            )

        return self._tbd.update(
            batch=batch,
            z_map=bg_result.z_map,
            class_map=class_result.class_map,
            sky_threshold=self._config.background.sky.z_threshold,
            not_sky_threshold=self._config.background.not_sky.z_threshold,
        )

    def reset(self) -> None:
        """Reset pipeline state for re-calibration."""
        self._window_index = 0
        self._classifier = BackgroundClassesClassifier(
            grid=self._grid,
            calibration_time_us=self._config.calibration_time_us,
            random_seed=self._config.random_seed,
        )
        self._background = DualBackgroundScorer(
            self._grid, self._config.background
        )
        self._tbd = SimplifiedTbdStage(self._grid, self._config.tbd)
