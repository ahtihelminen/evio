# Hardware tests: software crash and unplug

`make test-hardware` runs the `hardware` tier. Its recovery tests cause the
failures themselves, so nobody has to touch the rig:

- `tests/support/crash.py`: `crash_recorder(root)` runs a recorder in a
  subprocess, SIGKILLs its process group mid-write and returns the recording
  directory.
- `tests/support/usb_unplug.py`: `with unplugged(serial=...)` (or
  `vendor=`/`product=`, as the four-hex-digit sysfs ids) writes `0` to the
  device's `/sys/bus/usb/devices/<port>/authorized`, which unbinds it as a
  pulled cable would, and writes `1` on exit, also when the test fails.

## Granting write access to `authorized`

The attribute is root-owned `0644`. Without write access, or off Linux, a
test using `unplugged` skips and names the missing piece. Grant it to a group
the test user is in, for the camera vendors only:

```
# /etc/udev/rules.d/99-evio-usb-unplug.rules
# Look up each camera's idVendor with `lsusb` (the first half of ID xxxx:yyyy).
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="04b4", \
  RUN+="/bin/chgrp plugdev /sys%p/authorized", RUN+="/bin/chmod g+w /sys%p/authorized"
```

Then:

```
sudo groupadd -f plugdev && sudo usermod -aG plugdev "$USER"   # log in again
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=usb
ls -l /sys/bus/usb/devices/*/authorized    # the camera's: -rw-rw-r-- root plugdev
```

The rule runs again on every replug and reboot. Running the tier as root
(`sudo -E make test-hardware`) works too, but leaves root-owned files in the
tree, so prefer the rule.
