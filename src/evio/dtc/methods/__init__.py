"""Shipped methods; importing this registers them all."""

from evio.dtc.methods import (
    blob_tracker,
    kinematic,
    oracle,
    rate_excess,
    simplified,
    trackmanager,
    transformer,
    verdict_gate,
)

__all__ = [
    "blob_tracker",
    "kinematic",
    "oracle",
    "rate_excess",
    "simplified",
    "trackmanager",
    "transformer",
    "verdict_gate",
]
