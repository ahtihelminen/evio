"""Per-package line coverage held to the floors in `coverage_manifest.json`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def packages(root: Path) -> set[str]:
    """The top-level package directories under `root`."""
    return {
        path.name for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    }


def manifest_mismatches(
    found: set[str], manifest: Mapping[str, object],
) -> list[str]:
    """Packages with no floor, and floors naming no package."""
    named = {key for key in manifest if not key.startswith("_")}
    unlisted = [f"{name}: no entry in the manifest" for name in found - named]
    stale = [f"{name}: no such package" for name in named - found]
    return sorted(unlisted) + sorted(stale)


def below_floor(
    measured: Mapping[str, float], floors: Mapping[str, float],
) -> list[str]:
    """Each measured package whose line coverage is under its floor."""
    return [
        f"{name}: {percent:.1f}% < {floors[name]}%"
        for name, percent in sorted(measured.items())
        if percent < floors[name]
    ]
