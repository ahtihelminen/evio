"""Count-comparison-and-warn decision for camera serial reads.

Kept separate from `imx636_camera.py` — which imports the real `imx636`
USB driver binding at module load — specifically so this decision is
testable with plain data and no module stubbing (SN-468).
"""

from __future__ import annotations

from collections.abc import Sequence


def describe_serial_read(serials: Sequence[str]) -> str | None:
    """Warn when fewer devices reported a serial than were USB-matched.

    ``serials`` holds one entry per USB-matched device, in the shape
    `imx636.IMX636.list_cameras()` returns: an empty string means that
    device is present but its serial descriptor read came back empty.

    Returns the warning text (naming the matched/readable counts and the
    index of every affected device) or ``None`` when every matched device
    has a readable serial.
    """
    matched = len(serials)
    readable = [s for s in serials if s]
    if len(readable) == matched:
        return None
    missing_indices = [i for i, s in enumerate(serials) if not s]
    return (
        f"{matched} camera(s) matched over USB but only {len(readable)} "
        f"returned a readable serial; unreadable serial at "
        f"index {missing_indices} — camera present but not fully "
        "initialized, not absent"
    )
