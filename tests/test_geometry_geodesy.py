"""Known-answer checks for `geodetic_to_azel` (SN-607)."""

from __future__ import annotations

import pytest

from evio.geometry.geodesy import geodetic_to_azel

pytestmark = pytest.mark.t1

_LAT, _LON, _ALT = 65.17738298, 26.04051037, 50.0


def test_geodetic_to_azel_overhead_and_due_east() -> None:
    _, overhead_el, overhead_range = geodetic_to_azel(
        _LAT, _LON, _ALT, _LAT, _LON, _ALT + 500.0,
    )
    east_az, east_el, _ = geodetic_to_azel(
        _LAT, _LON, _ALT, _LAT, _LON + 0.001, _ALT,
    )

    assert overhead_el == pytest.approx(90.0)
    assert overhead_range == pytest.approx(500.0, abs=1e-6)
    assert east_az == pytest.approx(90.0, abs=1e-3)
    assert east_el == pytest.approx(0.0, abs=1e-2)
