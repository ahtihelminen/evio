"""Importing OpenVINO under the suite must not phone home.

`import openvino` starts Intel's GA4 telemetry, which sends from a worker
process: a network call the laptop suite promises not to make, and a worker
the session leak gate catches when the run is short enough to outlive it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.t1


def test_openvino_falls_back_to_its_telemetry_stub() -> None:
    pytest.importorskip("openvino")
    from openvino.tools.ovc import telemetry_utils

    assert telemetry_utils.tm.__name__ == "openvino.tools.ovc.telemetry_stub"
