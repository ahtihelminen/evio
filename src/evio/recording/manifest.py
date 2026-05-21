"""JSONL manifest for footage files.

Each day-bucket directory has a `manifest.jsonl`. One line per closed
file, structured so downstream tooling (training-data fetcher, retention
sweeper) can filter without re-scanning the raw `.dat`. Lines are
append-only — never rewritten — so writers and readers can coexist
without coordination.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ManifestEntry:
    filename: str
    start_ts_ns: int
    end_ts_ns: int
    size_bytes: int
    sha256: str
    object_ids: list[str] = field(default_factory=list)

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
        )


def append_entry(manifest_path: Path, entry: ManifestEntry) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")


def iter_manifest(manifest_path: Path) -> Iterator[ManifestEntry]:
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                yield ManifestEntry.from_json(line)
