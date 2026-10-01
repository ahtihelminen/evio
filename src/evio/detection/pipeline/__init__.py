from typing import Any

from evio.detection.grid import GridConfig
from evio.detection.pipeline.dual_background import (
    ClassSpecificConfig,
    DualBackgroundConfig,
    DualBackgroundResult,
    DualBackgroundScorer,
)
from evio.detection.pipeline.registry import (
    PIPELINES,
    DetectionPipeline,
    PipelineFactory,
    build,
    list_pipelines,
    register,
)
from evio.detection.pipeline.simplified_pipeline import (
    SimplifiedDetectionPipeline,
    SimplifiedPipelineConfig,
    SimplifiedPipelineResult,
)
from evio.detection.pipeline.simplified_tbd import (
    SimplifiedTbdConfig,
    SimplifiedTbdResult,
    SimplifiedTbdStage,
)
from evio.detection.types import Classification, Detection


@register("simplified")
def _build_simplified(cfg: dict[str, Any]) -> SimplifiedDetectionPipeline:
    grid = GridConfig(
        width=int(cfg.get("width", 1280)),
        height=int(cfg.get("height", 720)),
        rows=int(cfg.get("rows", 72)),
        cols=int(cfg.get("cols", 128)),
        time_bins=int(cfg.get("time_bins", 1)),
        window_us=int(cfg.get("window_us", 10_000)),
    )
    class_cfg = ClassSpecificConfig(
        z_threshold=int(cfg.get("z_threshold", 10)),
        update_threshold=int(cfg.get("update_threshold", 2)),
        calibration_windows=int(cfg.get("calibration_windows", 10)),
        count_weight=int(cfg.get("count_weight", 10)),
        multi_weight=int(cfg.get("multi_weight", 10)),
        coactivity_weight=int(cfg.get("coactivity_weight", 100)),
        count_std_floor=int(cfg.get("count_std_floor", 1)),
        multi_std_floor=int(cfg.get("multi_std_floor", 1)),
    )
    background = DualBackgroundConfig(
        sky=class_cfg, not_sky=class_cfg,
        alpha=int(cfg.get("alpha", 5)),
        ema_alpha=cfg.get("ema_alpha"),
        model_update_weight=float(cfg.get("model_update_weight", 0.1)),
    )
    config = SimplifiedPipelineConfig(
        calibration_time_us=int(cfg.get("calibration_time_us", int(1e6))),
        background=background,
    )
    return SimplifiedDetectionPipeline(grid, config)



__all__ = [
    "PIPELINES",
    "ClassSpecificConfig",
    "Classification",
    "Detection",
    "DetectionPipeline",
    "DualBackgroundConfig",
    "DualBackgroundResult",
    "DualBackgroundScorer",
    "PipelineFactory",
    "SimplifiedDetectionPipeline",
    "SimplifiedPipelineConfig",
    "SimplifiedPipelineResult",
    "SimplifiedTbdConfig",
    "SimplifiedTbdResult",
    "SimplifiedTbdStage",
    "build",
    "list_pipelines",
    "register",
]
