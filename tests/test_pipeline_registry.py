from __future__ import annotations

import pytest

from evio.detection.pipeline import (
    SimplifiedDetectionPipeline,
    build,
    list_pipelines,
)

pytestmark = pytest.mark.t1


def test_registry_includes_known_pipelines() -> None:
    """`simplified` is the only entry; the registry is the seam, not a menu."""
    assert list_pipelines() == ["simplified"]


def test_build_simplified_returns_correct_class() -> None:
    pipe = build("simplified", {"width": 1280, "height": 720})
    assert isinstance(pipe, SimplifiedDetectionPipeline)



def test_build_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown pipeline"):
        build("does-not-exist")


def test_build_simplified_calibration_time_matches_documented_warmup() -> None:
    """Drift guard for registry.py's docstring: fails if the factory default
    or the documented ~1 s / 100-window warmup figure moves without the
    other.
    """
    pipe = build("simplified", {})
    assert pipe._classifier._calibration_time_us == 1_000_000
