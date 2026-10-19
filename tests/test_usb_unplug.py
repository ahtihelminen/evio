from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fakes.sysfs_usb import add_usb_device
from support.usb_unplug import find_usb_devices, unplugged

pytestmark = pytest.mark.t1


@pytest.fixture
def sysfs(tmp_path: Path) -> Path:
    add_usb_device(tmp_path, "usb1", "1d6b", "0002")
    add_usb_device(tmp_path, "1-2", "04b4", "00f5", serial="P50905")
    add_usb_device(tmp_path, "1-3", "04b4", "00f5", serial="P51000")
    return tmp_path


def test_find_resolves_a_device_by_vendor_and_product(sysfs: Path) -> None:
    assert find_usb_devices(sysfs, vendor="1d6b", product="0002") == [
        sysfs / "usb1",
    ]


def test_find_resolves_a_device_by_serial(sysfs: Path) -> None:
    assert find_usb_devices(sysfs, serial="P51000") == [sysfs / "1-3"]


def test_find_narrows_by_id_and_serial_together(sysfs: Path) -> None:
    found = find_usb_devices(
        sysfs, vendor="04b4", product="00f5", serial="P50905",
    )
    assert found == [sysfs / "1-2"]


def test_find_lists_every_identical_device_in_port_order(sysfs: Path) -> None:
    found = find_usb_devices(sysfs, vendor="04b4", product="00f5")
    assert found == [sysfs / "1-2", sysfs / "1-3"]


def test_find_narrows_by_port_path(sysfs: Path) -> None:
    found = find_usb_devices(sysfs, vendor="04b4", product="00f5", port="1-3")
    assert found == [sysfs / "1-3"]


def test_find_returns_nothing_for_an_absent_device(sysfs: Path) -> None:
    assert find_usb_devices(sysfs, serial="nope") == []


def test_find_ignores_a_port_whose_ids_do_not_match(sysfs: Path) -> None:
    assert find_usb_devices(sysfs, vendor="04b4", port="usb1") == []


def test_unplugged_detaches_then_reattaches(sysfs: Path) -> None:
    authorized = sysfs / "1-2" / "authorized"
    with unplugged(serial="P50905", sysfs=sysfs) as device:
        assert device == sysfs / "1-2"
        assert authorized.read_text() == "0"
    assert authorized.read_text() == "1"


def test_unplugged_reattaches_when_the_body_fails(sysfs: Path) -> None:
    with pytest.raises(RuntimeError), unplugged(serial="P50905", sysfs=sysfs):
        raise RuntimeError
    assert (sysfs / "1-2" / "authorized").read_text() == "1"


def test_a_missing_device_skips_naming_what_was_looked_for(sysfs: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="serial='nope'"), \
            unplugged(serial="nope", sysfs=sysfs):
        pass


def test_no_write_access_skips_naming_the_file_and_the_setup(
    sysfs: Path,
) -> None:
    authorized = sysfs / "1-2" / "authorized"
    authorized.chmod(0o444)
    with pytest.raises(pytest.skip.Exception) as skipped, \
            unplugged(serial="P50905", sysfs=sysfs):
        pass
    assert str(authorized) in str(skipped.value)
    assert "docs/hardware-tests.md" in str(skipped.value)
    assert authorized.read_text() == "1\n"


def test_a_platform_other_than_linux_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(pytest.skip.Exception, match="Linux"), \
            unplugged(serial="P50905"):
        pass


def test_unplugged_takes_the_one_device_named_by_port(sysfs: Path) -> None:
    with unplugged(vendor="04b4", product="00f5", port="1-3", sysfs=sysfs) as d:
        assert d == sysfs / "1-3"
        assert (sysfs / "1-3" / "authorized").read_text() == "0"
        assert (sysfs / "1-2" / "authorized").read_text() == "1\n"
    assert (sysfs / "1-3" / "authorized").read_text() == "1"


def test_several_identical_devices_skip_naming_their_ports(
    sysfs: Path,
) -> None:
    with pytest.raises(pytest.skip.Exception) as skipped, \
            unplugged(vendor="04b4", product="00f5", sysfs=sysfs):
        pass
    message = str(skipped.value)
    assert "1-2" in message
    assert "1-3" in message
    assert "port" in message
    assert (sysfs / "1-2" / "authorized").read_text() == "1\n"


def test_a_port_with_no_matching_device_skips_naming_the_port(
    sysfs: Path,
) -> None:
    with pytest.raises(pytest.skip.Exception, match="port='9-9'"), \
            unplugged(vendor="04b4", port="9-9", sysfs=sysfs):
        pass
