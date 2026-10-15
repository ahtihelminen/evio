"""The stubbed GPU voxel CNN: manifest, dependency and facts machinery."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import fakes.dtc_voxel_cnn as voxel_module
import numpy as np
import pytest
from fakes.dtc_voxel_cnn import VoxelCnn, VoxelCnnConfig

from evio.dtc.errors import ResourceError
from evio.dtc.facts import EXTENT, missing_facts
from evio.dtc.manifest import Compute, Cost, Stage, check_dependencies
from evio.dtc.protocols import Context
from evio.dtc.types import Step
from tests.support.dtc import blob_events, empty_events, step_of, window

pytestmark = pytest.mark.t1

STEP_US = 10_000


def build(**config: int) -> VoxelCnn:
    context = Context("cam", ("cam",), {EXTENT: (1280, 720)}, STEP_US)
    return VoxelCnn(VoxelCnnConfig(**config), context)


def blob_step(cx: int = 640, cy: int = 360, seed: int = 0) -> Step:
    batch = blob_events(0, STEP_US, cx, cy, seed=seed)
    return step_of(0, STEP_US, window("cam", batch, 0, STEP_US))


def test_manifest_declares_a_heavy_gpu_detector() -> None:
    manifest = VoxelCnn.manifest
    assert manifest.stages == (Stage.DETECT,)
    assert manifest.compute is Compute.GPU
    assert manifest.cost is Cost.HEAVY
    assert manifest.dependencies == ("torch",)
    assert manifest.requires == (EXTENT,)
    assert manifest.cadence == 1


def test_dependencies_pass_when_torch_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    check_dependencies(VoxelCnn.manifest)


def test_dependencies_raise_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ResourceError, match="torch"):
        check_dependencies(VoxelCnn.manifest)


def test_missing_extent_is_reported() -> None:
    required = VoxelCnn.manifest.requires
    assert missing_facts(required, {}) == (EXTENT,)
    assert missing_facts(required, {EXTENT: (1280, 720)}) == ()


def test_module_has_no_top_level_torch_import() -> None:
    tree = ast.parse(Path(voxel_module.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"torch", "openvino"}


def test_voxel_grid_counts_every_event_in_its_time_bin() -> None:
    detector = build(time_bins=4)
    only = blob_step().windows["cam"]
    grid = detector.voxel_grid(only)
    assert grid.shape == (4, 45, 80)
    assert grid.sum() == len(only.events.timestamps)


def test_detects_the_blob_in_the_densest_voxel() -> None:
    candidates = build(top_k=1).step(blob_step(), ()).items
    assert len(candidates) == 1
    x, y = candidates[0].region.center
    assert abs(x - 640) <= 16
    assert abs(y - 360) <= 16


def test_each_cell_is_proposed_once() -> None:
    boxes = [c.region.box for c in build(top_k=5).step(blob_step(), ()).items]
    assert len(boxes) == len(set(boxes))


def test_empty_window_gives_no_candidates() -> None:
    quiet = step_of(0, STEP_US, window("cam", empty_events(), 0, STEP_US))
    assert build().step(quiet, ()).items == ()


def test_step_is_deterministic_and_reset_restores_it() -> None:
    detector = build()
    first = detector.step(blob_step(), ())
    detector.reset()
    assert detector.step(blob_step(), ()) == first


def test_empty_window_with_no_minimum_gives_no_candidates() -> None:
    quiet = step_of(0, STEP_US, window("cam", empty_events(), 0, STEP_US))
    assert build(min_events=0).step(quiet, ()).items == ()


def test_step_does_not_mutate_the_window() -> None:
    step = blob_step()
    before = step.windows["cam"].events.timestamps.copy()
    build().step(step, ())
    assert np.array_equal(step.windows["cam"].events.timestamps, before)
