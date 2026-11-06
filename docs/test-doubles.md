# Test doubles

Every module under `tests/fakes/` and `tests/support/`, with the test that
pins it to the real thing, and where it differs from the real device. The
differences are in the `Differences from the real device:` section of each
module's docstring, next to the code. `tests/test_double_inventory.py` fails
when a module is missing here, names a test that does not exist, or lacks
that section.

A conformance test is one that runs the double and the real thing side by
side, or runs the double's output through the real consumer's checks. A
test that merely uses the double does not count. Where there is none the
cell says `GAP`: the double is trusted on the strength of its docstring.
Writing those tests is tracked apart from this list.

## `tests/fakes/`

| Module | Conformance test |
| --- | --- |
| `tests/fakes/clock.py` | GAP: no test compares it with `time` |
| `tests/fakes/dtc_broken.py` | `tests/test_dtc_conformance.py` |
| `tests/fakes/dtc_counting_pipeline.py` | GAP: no test compares it with the real pipeline |
| `tests/fakes/dtc_frame.py` | GAP: only its own method tests use it |
| `tests/fakes/dtc_joint.py` | GAP: only its own method tests use it |
| `tests/fakes/dtc_point.py` | GAP: only its own method tests use it |
| `tests/fakes/dtc_rate_excess_reference.py` | `tests/test_dtc_method_rate_excess.py` |
| `tests/fakes/dtc_timer.py` | GAP: no test compares it with the real `Timer` |
| `tests/fakes/dtc_toys.py` | GAP: no test compares the toys with real methods |
| `tests/fakes/dtc_transformer.py` | GAP: no test compares it with the real loader and predictor |
| `tests/fakes/dtc_voxel_cnn.py` | GAP: no test compares it with the real CNN method |
| `tests/fakes/imx636_camera.py` | GAP: no test compares it with the `imx636` driver |
| `tests/fakes/imx636_stream.py` | `tests/test_fakes_fidelity_imx636_stream.py` |
| `tests/fakes/kinematic_llr.py` | `tests/test_fakes_fidelity_detection.py` |
| `tests/fakes/pacer.py` | `tests/test_fakes_fidelity_pacer.py` |
| `tests/fakes/sysfs_usb.py` | GAP: no test compares it with a real sysfs tree |
| `tests/fakes/track_classifier.py` | `tests/test_fakes_fidelity_detection.py` |
| `tests/fakes/u3v_camera.py` | `tests/test_fakes_fidelity_u3v_camera.py` |

## `tests/support/`

Most of these are helpers, not stand-ins; they are listed so the list is
complete, and the same rule applies to them.

| Module | Conformance test |
| --- | --- |
| `tests/support/coverage_floor.py` | `tests/test_coverage_manifest.py` |
| `tests/support/crash.py` | `tests/test_crash_recorder.py` |
| `tests/support/dat.py` | `tests/test_dt8_codec_roundtrip.py` |
| `tests/support/dtc.py` | GAP: builders, checked only through the tests that use them |
| `tests/support/dtc_chain.py` | GAP: builders, checked only through the tests that use them |
| `tests/support/fakescan.py` | `tests/test_no_adhoc_fakes.py` |
| `tests/support/hygiene.py` | GAP: no test of the leak report itself |
| `tests/support/network_guard.py` | `tests/test_socket_guard.py` |
| `tests/support/recorder_child.py` | `tests/test_crash_recorder.py` |
| `tests/support/usb_unplug.py` | `tests/test_usb_unplug.py` |
