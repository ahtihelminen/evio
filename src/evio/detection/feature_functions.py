from collections.abc import Callable

from evio.detection.features.basic_features import (
    cell_8_coactivity,
    cell_active_pixels,
    cell_event_count,
    cell_polarity_skewness,
)
from evio.detection.features.entropy import cell_shannon_entropy

FEATURE_FUNCTIONS: dict[str, Callable] = {
    "event_count": cell_event_count,
    "active_pixels": cell_active_pixels,
    "8_coactivity": cell_8_coactivity,
    "polarity_skewness": cell_polarity_skewness,
    "entropy": cell_shannon_entropy,
}

DEFAULT_FEATURE_NAME = "event_count"


def get_feature_fn(name: str) -> Callable:
    return FEATURE_FUNCTIONS.get(name, FEATURE_FUNCTIONS[DEFAULT_FEATURE_NAME])
