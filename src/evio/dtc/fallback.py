"""The zero-config chain evio ships: blobs, tracks, then drone / no-drone.

For a source with an extent and nothing else installed: the rate-excess blob
detector, the blob tracker, the torch-free kinematic classifier and the
persistence gate, so a fresh checkout can detect, track and raise alarms
without a checkpoint. A real deployment replaces the stages with its own
methods.
"""

from __future__ import annotations

from evio.dtc.spec import PipelineSpec, StageSpec


def fallback_spec() -> PipelineSpec:
    return PipelineSpec(
        (
            StageSpec("rate-excess"),
            StageSpec("blob-tracker"),
            StageSpec("kinematic"),
            StageSpec("verdict-gate"),
        )
    )
