from __future__ import annotations

import pytest

from evio.detection.pipeline import (
    GmmOnlineBgDetectionPipeline,
    SimplifiedDetectionPipeline,
    build,
    list_pipelines,
)


def test_registry_includes_known_pipelines() -> None:
    names = list_pipelines()
    assert "simplified" in names
    assert "gmm_online" in names


def test_build_simplified_returns_correct_class() -> None:
    pipe = build("simplified", {"width": 1280, "height": 720})
    assert isinstance(pipe, SimplifiedDetectionPipeline)


def test_build_gmm_online_returns_correct_class() -> None:
    pipe = build("gmm_online", {"width": 1280, "height": 720})
    assert isinstance(pipe, GmmOnlineBgDetectionPipeline)


def test_build_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown pipeline"):
        build("does-not-exist")
