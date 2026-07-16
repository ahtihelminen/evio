"""JSONL manifest for footage files.

Each day-bucket directory has a `manifest.jsonl`. One line per closed
file, structured so downstream tooling (training-data fetcher, retention
sweeper) can filter without re-scanning the raw `.dat`. Lines are
append-only — never rewritten — so writers and readers can coexist
without coordination.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class ManifestEntry:
    filename: str
    start_ts_ns: int
    end_ts_ns: int
    size_bytes: int
    sha256: str
    object_ids: list[str] = field(default_factory=list)
    # Static per-file capture snapshot: sensor dims, window/step, the bias
    # vector active when the file opened, plus the autotuner's regime/scene
    # context. None on un-augmented recordings. The bias vector here can't
    # be reconstructed from the events, so it is the file's gold metadata.
    capture: dict[str, object] | None = None
    # Wall clock (epoch ns) when the recorder OPENED the file. The filename
    # encodes only the hour bucket + run start, so this is the one durable
    # record of a recording re-enable time — replay (dat_index) prefers it
    # over the name-derived start. None on pre-field manifests and on
    # recovered orphans (the open time died with the recorder).
    wall_start_ns: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> ManifestEntry:
        data = json.loads(line)
        return cls(
            filename=data["filename"],
            start_ts_ns=int(data["start_ts_ns"]),
            end_ts_ns=int(data["end_ts_ns"]),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
            object_ids=list(data.get("object_ids", [])),
            capture=data.get("capture"),
            wall_start_ns=data.get("wall_start_ns"),
        )


def append_entry(manifest_path: Path, entry: ManifestEntry) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")


def iter_manifest(manifest_path: Path) -> Iterator[ManifestEntry]:
    """Yield entries, skipping unparseable lines (warned once per file).

    ``append_entry`` is not atomic, so a power cut can tear the tail
    line; one bad line must never make a whole day's footage unlistable.
    """
    if not manifest_path.exists():
        return
    warned = False
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                yield ManifestEntry.from_json(line)
            except (KeyError, TypeError, ValueError):
                if not warned:
                    warned = True
                    _log.warning(
                        "%s: skipping unparseable manifest line(s) — torn "
                        "append?", manifest_path,
                    )


def entry_for_file(dat_path: Path) -> ManifestEntry | None:
    """Manifest entry for one footage file, if its day-dir has one."""
    for entry in iter_manifest(dat_path.parent / "manifest.jsonl"):
        if entry.filename == dat_path.name:
            return entry
    return None
