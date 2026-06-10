"""List and delete footage a day at a time, for the web UI.

The recorder buckets footage as `<root>/<camera_id>/YYYY-MM-DD/`. An
operator clearing disk in the field works at that day granularity, so the
two helpers here enumerate those day directories with their on-disk size
and delete one after validating it really is a `camera/day` path inside the
given root — deletion is irreversible, so a traversal in `camera_id`/`day`
must never escape the footage tree.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def list_footage_days(roots: dict[str, Path]) -> dict[str, list[dict]]:
    """Per-root list of `{camera_id, day, bytes, files}`, newest day first."""
    return {label: _days_under(root) for label, root in roots.items()}


def _days_under(root: Path) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    days: list[dict] = []
    for camera_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for day_dir in sorted(p for p in camera_dir.iterdir() if p.is_dir()):
            if not _DAY_RE.match(day_dir.name):
                continue
            files = [f for f in day_dir.iterdir() if f.is_file()]
            days.append({
                "camera_id": camera_dir.name,
                "day": day_dir.name,
                "bytes": sum(f.stat().st_size for f in files),
                "files": len(files),
            })
    days.sort(key=lambda d: (d["day"], d["camera_id"]), reverse=True)
    return days


def delete_footage_day(root: Path, camera_id: str, day: str) -> int:
    """Delete `<root>/<camera_id>/<day>/`; return the bytes freed.

    Rejects anything that isn't a literal `camera/YYYY-MM-DD` directory
    resolving inside `root`, so a crafted `camera_id`/`day` can't reach
    outside the footage tree.
    """
    if not _DAY_RE.match(day):
        msg = f"not a YYYY-MM-DD day: {day!r}"
        raise ValueError(msg)
    if not camera_id or "/" in camera_id or camera_id in {".", ".."}:
        msg = f"invalid camera id: {camera_id!r}"
        raise ValueError(msg)
    root = Path(root).resolve()
    target = (root / camera_id / day).resolve()
    if not target.is_relative_to(root) or target.parent.parent != root:
        msg = f"{camera_id}/{day} is not a day directory under {root}"
        raise ValueError(msg)
    if not target.is_dir():
        msg = f"no footage at {camera_id}/{day}"
        raise FileNotFoundError(msg)
    freed = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    shutil.rmtree(target)
    return freed
