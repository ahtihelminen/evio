from collections.abc import Callable

from evio.detection.features.features import (
    cell_8_coactivity,
    cell_active_pixels,
    cell_event_count,
    cell_polarity_skewness,
)

FEATURE_FUNCTIONS: dict[str, Callable] = {
    "cell_event_count": cell_event_count,
    "cell_active_pixels": cell_active_pixels,
    "cell_8_coactivity": cell_8_coactivity,
    "cell_polarity_skewness": cell_polarity_skewness,
}

DEFAULT_FEATURE_NAME = "cell_8_coactivity"


def get_feature_fn(name: str) -> Callable:
    return FEATURE_FUNCTIONS.get(name, FEATURE_FUNCTIONS[DEFAULT_FEATURE_NAME])
