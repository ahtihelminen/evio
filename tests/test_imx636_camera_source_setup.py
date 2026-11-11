"""`Imx636CameraSource.__enter__` start-up steps, run against the IMX636 double.

Opening a camera switches on the anti-flicker filter and the event rate
controller when asked, then re-programs the saved DEM masks of its serial
(`_reload_dem_masks`). The double in `tests/fakes/imx636_camera.py` covers
what `stuck_pixel_scan` calls; the module-level pieces `__enter__` also needs
(`IMX636.open`, `ERCConfig`, `set_erc`, `set_cd_callback`) are added here.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fakes.imx636_camera import (
    FakeImx636Camera,
    FakePixelMask,
    fake_imx636_module,
)

from evio.streaming.stuck_pixel_scan import SOFT_MASK_MAX

pytestmark = pytest.mark.t1

SERIAL = "S123"


def erc_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(enabled=False, event_rate=0)


@dataclass
class SetupCamera(FakeImx636Camera):
    """The double plus the calls only `__enter__` makes."""

    erc: list[types.SimpleNamespace] = field(default_factory=list)
    antiflicker_calls: int = 0
    runtime_rejected: set[tuple[int, int]] = field(default_factory=set)
    opened_serial: str | None = None

    def set_cd_callback(self, _callback: object) -> None:
        pass

    def set_erc(self, cfg: types.SimpleNamespace) -> None:
        self.erc.append(cfg)

    def set_antiflicker(self, cfg: object) -> None:
        self.antiflicker_calls += 1
        super().set_antiflicker(cfg)  # type: ignore[arg-type]

    def set_pixel_mask(self, slot: int, mask: FakePixelMask) -> None:
        if (mask.x, mask.y) in self.runtime_rejected:
            message = "mask rejected"
            raise RuntimeError(message)
        super().set_pixel_mask(slot, mask)


@pytest.fixture
def camera(monkeypatch: pytest.MonkeyPatch) -> SetupCamera:
    cam = SetupCamera.idle(1280, 720)
    module = fake_imx636_module()
    module.ERCConfig = erc_config

    def open_camera(serial: str) -> SetupCamera:
        cam.opened_serial = serial
        return cam

    module.IMX636 = types.SimpleNamespace(open=open_camera)
    monkeypatch.setitem(sys.modules, "imx636", module)
    from evio.io import imx636_camera
    monkeypatch.setattr(imx636_camera, "imx636", module)
    return cam


def _open(camera: SetupCamera, tmp_path: Path, **kwargs: object) -> object:
    from evio.io.imx636_camera import Imx636CameraSource
    source = Imx636CameraSource(
        serial=kwargs.pop("serial", SERIAL),  # type: ignore[arg-type]
        dem_mask_dir=tmp_path, **kwargs,  # type: ignore[arg-type]
    )
    return source.__enter__()


def _manifest(tmp_path: Path, data: object, serial: str = SERIAL) -> None:
    (tmp_path / f"{serial}.json").write_text(json.dumps(data), encoding="utf-8")


def _slots(camera: SetupCamera) -> dict[int, tuple[int, int]]:
    return {s: (m.x, m.y) for s, m in camera.pixel_masks.items()}


def test_anti_flicker_programs_the_band_stop_filter(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _open(camera, tmp_path, anti_flicker=True)

    assert camera.antiflicker_calls == 1
    cfg = camera.antiflicker
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.band_stop is True
    assert cfg.low_freq == 50
    assert cfg.high_freq == 500
    assert cfg.duty_cycle == 50  # type: ignore[attr-defined]


def test_no_anti_flicker_by_default(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _open(camera, tmp_path, anti_flicker=False)

    assert camera.antiflicker_calls == 0
    assert camera.antiflicker is None


def test_erc_is_programmed_with_the_requested_rate(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _open(camera, tmp_path, erc_event_rate=1_000_000)

    assert len(camera.erc) == 1
    assert camera.erc[0].enabled is True
    assert camera.erc[0].event_rate == 1_000_000


def test_no_erc_when_the_rate_is_zero(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _open(camera, tmp_path, erc_event_rate=0)

    assert camera.erc == []


def test_masks_with_slot_fields_go_to_those_slots(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _manifest(tmp_path, {"masks": [
        {"slot": 5, "x": 10, "y": 20},
        {"slot": 2, "x": 30, "y": 40},
    ]})

    _open(camera, tmp_path)

    assert _slots(camera) == {5: (10, 20), 2: (30, 40)}
    assert all(m.enabled for m in camera.pixel_masks.values())


def test_v1_manifest_without_slots_uses_the_list_index(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _manifest(tmp_path, {"masks": [
        {"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6},
    ]})

    _open(camera, tmp_path)

    assert _slots(camera) == {0: (1, 2), 1: (3, 4), 2: (5, 6)}


def test_only_the_first_64_masks_are_programmed(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _manifest(tmp_path, {"masks": [{"x": i, "y": 0} for i in range(70)]})

    _open(camera, tmp_path)

    assert _slots(camera) == {i: (i, 0) for i in range(64)}


def test_rejected_masks_are_skipped_and_not_counted(
    camera: SetupCamera, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    camera.runtime_rejected = {(7, 7)}
    _manifest(tmp_path, {"masks": [
        {"slot": 0, "x": 1, "y": 1},
        {"slot": 1, "x": 7, "y": 7},  # RuntimeError
        {"slot": 90, "x": 2, "y": 2},  # ValueError: slot out of range
        {"slot": 3, "x": 3, "y": 3},
    ]})

    _open(camera, tmp_path)

    assert _slots(camera) == {0: (1, 1), 3: (3, 3)}
    assert "reloaded 2 DEM mask(s)" in capsys.readouterr().out


def test_invalid_json_warns_and_leaves_the_camera_usable(
    camera: SetupCamera, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / f"{SERIAL}.json").write_text("{not json", encoding="utf-8")

    source = _open(camera, tmp_path)

    assert "[warn] dem-mask config unreadable" in capsys.readouterr().out
    assert source.cam is camera  # type: ignore[attr-defined]
    assert source.soft_mask_coords == []  # type: ignore[attr-defined]
    assert camera.pixel_masks == {}


def test_soft_masks_out_of_bounds_are_dropped(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _manifest(tmp_path, {"soft_masks": [
        {"x": 0, "y": 0},
        {"x": 1279, "y": 719},
        {"x": 1280, "y": 0},
        {"x": 0, "y": 720},
        {"x": -1, "y": 5},
        {"x": 5, "y": -1},
        {"x": 9},
    ]})

    source = _open(camera, tmp_path)

    assert source.soft_mask_coords == [(0, 0), (1279, 719)]  # type: ignore[attr-defined]


def test_soft_masks_beyond_the_cap_are_dropped(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _manifest(tmp_path, {"soft_masks": [
        {"x": i % 1280, "y": i // 1280} for i in range(SOFT_MASK_MAX + 10)
    ]})

    source = _open(camera, tmp_path)

    coords = source.soft_mask_coords  # type: ignore[attr-defined]
    assert len(coords) == SOFT_MASK_MAX
    last = SOFT_MASK_MAX - 1
    assert coords[-1] == (last % 1280, last // 1280)


def test_event_serial_programs_no_masks(
    camera: SetupCamera, tmp_path: Path,
) -> None:
    _manifest(tmp_path, {
        "masks": [{"x": 1, "y": 2}], "soft_masks": [{"x": 3, "y": 4}],
    }, serial="event-0")

    source = _open(camera, tmp_path, serial="event-0")

    assert camera.opened_serial == ""
    assert camera.pixel_masks == {}
    assert source.soft_mask_coords == []  # type: ignore[attr-defined]
