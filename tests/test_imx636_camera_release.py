"""`Imx636CameraSource.__exit__` must release the camera, not just stop it.

`__exit__` called `cam.stop()` and kept `self.cam`, so the C++ `IMX636`
destructor never ran and its libusb interface claim was held for the life
of the process. Boot diagnostics opens every camera in the parent through
this context manager, so the parent kept all six claimed and each forked
`camera_worker` then failed to open ANY of them:

    [debug] open failed serial=...: IMX636: no compatible camera found.

Verified on ops001 (2026-09-08): a child process could open a camera
before the parent ran a diagnostics cycle, failed with that exact error
afterwards, and succeeded again the moment the parent dropped its
reference.

The binding exposes no `close()`; dropping the last reference is what
frees the claim, so that is what these tests assert.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = pytest.mark.t1


class _FakeCam:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


class _UnreachableCam(_FakeCam):
    """A camera already dead from a USB error: `stop()` writes ISSD_STOP
    and throws. Cleanup is best-effort, but must still release."""

    def stop(self) -> None:
        self.stops += 1
        msg = "ISSD write_field failed at step 0"
        raise RuntimeError(msg)


def _source(
    monkeypatch: pytest.MonkeyPatch, cam: object, *, streaming: bool,
) -> object:
    monkeypatch.setitem(
        sys.modules, "imx636",
        sys.modules.get("imx636", types.ModuleType("imx636")),
    )
    from evio.io.imx636_camera import Imx636CameraSource
    source = Imx636CameraSource(serial="test")
    source.cam = cam
    source._streaming = streaming
    return source


def test_exit_releases_a_streaming_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = _FakeCam()
    source = _source(monkeypatch, cam, streaming=True)

    source.__exit__(None, None, None)

    assert cam.stops == 1, "a streaming camera must be stopped"
    assert source.cam is None, "claim leaked: the reference was kept"


def test_exit_releases_a_camera_that_never_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`__enter__` claims the device; streaming starts later, lazily. A
    source that is opened and dropped before its first `__next__` still
    holds a claim, so the early return must not skip the release."""
    cam = _FakeCam()
    source = _source(monkeypatch, cam, streaming=False)

    source.__exit__(None, None, None)

    assert cam.stops == 0, "nothing to stop; it never started"
    assert source.cam is None, "claim leaked on the never-streamed path"


def test_exit_releases_even_when_stop_throws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = _UnreachableCam()
    source = _source(monkeypatch, cam, streaming=True)

    source.__exit__(None, None, None)

    assert source.cam is None, "claim leaked when stop() failed"


def test_exit_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    cam = _FakeCam()
    source = _source(monkeypatch, cam, streaming=True)

    source.__exit__(None, None, None)
    source.__exit__(None, None, None)

    assert cam.stops == 1, "a released camera must not be stopped twice"
