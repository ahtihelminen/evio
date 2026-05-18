"""Mount pose — where the appliance is and which way it faces in the world.

`world_yaw_deg` is the configurable slot that decouples the math from
whatever method we eventually use to determine the appliance's heading
relative to true north (operator sighting at install / onboard magnetometer
/ dual-antenna GNSS / ...). Whatever supplies the number, the projection
math consumes it uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MountPose:
    """Position + heading of the appliance in the world."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    world_yaw_deg: float = 0.0
