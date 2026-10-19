# Hardware tests: software crash and unplug

`make test-hardware` runs the `hardware` tier. Its recovery tests cause the
failures themselves, so nobody has to touch the rig:

- `tests/support/crash.py`: `crash_recorder(root)` runs a recorder in a
  subprocess, SIGKILLs its process group mid-write and returns the recording
  directory.
- `tests/support/usb_unplug.py`: `with unplugged(vendor=..., product=...)`
  (the four-hex-digit sysfs ids, or `serial=`) writes `0` to the device's
  `/sys/bus/usb/devices/<port>/authorized` and writes `1` on exit, also when
  the test fails.

## Choosing which device to unplug

`unplugged` takes the one device that matches. A unit carries several
identical cameras (six `1409:8e00` IMX636), and their USB serial is empty, so
a vendor and product match them all: the helper then skips, naming the ports,
rather than unplug an arbitrary one. Name the port, the sysfs directory in
`/sys/bus/usb/devices`, with `port=`; the IMX636 hardware test reads it from
`EVIO_UNPLUG_PORT`:

```
EVIO_UNPLUG_PORT=2-1.1 make test-hardware
```

A rig with a single camera needs nothing. Find a port with `lsusb -t`.

## What the unplug is

`authorized` = 0 deauthorizes the device: its interfaces and drivers go, but
the device itself stays enumerated. The IMX636 driver needs an interface to
read a serial, so that camera disappears from its enumeration; the U3V
camera's serial is read from the USB descriptor, so it stays listed. The
helper is therefore not a cable pull, and a test for the U3V camera cannot
use it to assert disappearance.

## Granting write access to `authorized`

The attribute is root-owned `0644`. Without write access, or off Linux, a
test using `unplugged` skips and names the missing piece. Grant it to a group
the test user is in, for the camera vendors only:

```
# /etc/udev/rules.d/99-evio-usb-unplug.rules
# Look up each camera's idVendor with `lsusb` (the first half of ID xxxx:yyyy).
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="1409", \
  RUN+="/bin/chgrp plugdev /sys%p/authorized", RUN+="/bin/chmod g+w /sys%p/authorized"
```

Then:

```
sudo groupadd -f plugdev && sudo usermod -aG plugdev "$USER"   # log in again
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=usb && sudo udevadm settle
ls -l /sys/bus/usb/devices/*/authorized    # the camera's: -rw-rw-r-- root plugdev
```

The rule runs again on every replug and reboot. Running the tier as root
(`sudo -E make test-hardware`) works too, but leaves root-owned files in the
tree, so prefer the rule.
