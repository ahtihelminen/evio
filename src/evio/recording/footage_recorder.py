"""Tees `EventsBatch` into hour-bucketed DT8 `.dat` files.

`write(batch)` appends to the file for the current hour; a new file
opens automatically when the wall-clock hour rolls over. Each emit from
the SAPIENT side calls `mark_object_id(track_id)` so the manifest entry
gets a sticky bit later retention sweeps will honour.

Path layout:

    <root>/<camera_id>/YYYY-MM-DD/HHMMSS.dat
    <root>/<camera_id>/YYYY-MM-DD/manifest.jsonl
    <root>/<camera_id>/provenance-<run_id>.jsonl

`provenance-<run_id>.jsonl` is a per-run, per-sensor change-log of the
IMX636 bias vector (and the autotuner's regime/scene context). A bias is
a hardware setting applied before events exist, so it cannot be recovered
from the `.dat` — it has to be logged at capture time, stamped with the
event-time it took effect, to be joinable to the events offline.

Files are written eagerly (no in-memory queue) — at IMX636 event rates
the codec encode is the bottleneck, not disk I/O. If that changes,
wrap `write` in a thread-pool worker; the rotation/manifest logic
doesn't depend on calling thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from evio.codec.dt8 import Dt8Codec
from evio.core.events import EventsBatch
from evio.recording.manifest import ManifestEntry, append_entry

DAT_FILE_HEADER = b"% evio footage recorder\n\x00\x08"

_log = logging.getLogger(__name__)
_TimeFn = Callable[[], float]
_Bucket = tuple[int, int, int, int]  # (year, month, day, hour) in UTC


class FootageRecorder:
    def __init__(
        self,
        *,
        root: Path,
        camera_id: str,
        width: int,
        height: int,
        window_us: int = 0,
        step_us: int = 0,
        fsync_interval_s: float = 2.0,
        time_provider: _TimeFn | None = None,
    ) -> None:
        self._root = Path(root)
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._window_us = window_us
        self._step_us = step_us
        self._fsync_interval_s = fsync_interval_s
        self._codec = Dt8Codec(width=width, height=height)
        self._now: _TimeFn = time_provider or time.time
        self._last_fsync: float = self._now()
        # Per-process suffix so a re-run inside the same UTC hour writes a
        # new file instead of clobbering the previous one. HHMMSSZ of the
        # recorder's construction wall-clock; ambiguous only if two
        # processes start within the same second on the same host.
        self._run_id = time.strftime("%H%M%SZ", time.gmtime(self._now()))
        self._file: BinaryIO | None = None
        self._file_path: Path | None = None
        self._bucket: _Bucket | None = None
        self._start_ts_ns: int | None = None
        self._end_ts_ns: int = 0
        self._size_bytes: int = 0
        self._hasher = hashlib.sha256()
        self._object_ids: set[str] = set()
        # Running capture state, carried across file rotations: the latest
        # bias vector and the autotuner's regime/scene context. Each closed
        # file's manifest entry snapshots these; the provenance log records
        # every change. Written lazily so a no-event run leaves no dir.
        self._provenance_path = (
            self._root / camera_id / f"provenance-{self._run_id}.jsonl"
        )
        self._biases: dict[str, int] = {}
        self._context: dict[str, object] = {}

    def write(self, batch: EventsBatch) -> None:
        if batch.timestamps.size == 0:
            return
        bucket = self._bucket_for(self._now())
        if bucket != self._bucket:
            self._rotate(bucket)
        self._append(batch)

    def mark_object_id(self, object_id: str) -> None:
        if self._file is not None and object_id:
            self._object_ids.add(object_id)

    def note_bias(self, name: str, value: int, ts_ns: int) -> None:
        """Record one applied IMX636 bias write, stamped with event-time."""
        self._biases[name] = value
        self._provenance(
            {"event": "bias", "ts_ns": ts_ns, "name": name, "value": value},
        )

    def note_context(self, context: dict[str, object], ts_ns: int) -> None:
        """Record the autotuner's regime/scene context + full bias vector."""
        biases = context.get("biases")
        if isinstance(biases, dict):
            self._biases.update(biases)
        self._context = {k: v for k, v in context.items() if k != "biases"}
        self._provenance({"event": "context", "ts_ns": ts_ns, **context})

    def _provenance(self, record: dict[str, object]) -> None:
        self._provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with self._provenance_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        self._finalise_current()

    @property
    def current_path(self) -> Path | None:
        return self._file_path

    def _append(self, batch: EventsBatch) -> None:
        if self._file is None:
            msg = "_append called with no open file"
            raise RuntimeError(msg)
        words = self._codec.encode_chunk(batch)
        if words.size == 0:
            return
        payload = words.tobytes()
        self._file.write(payload)
        self._size_bytes += len(payload)
        self._hasher.update(payload)
        last_us = int(batch.timestamps[-1])
        self._end_ts_ns = last_us * 1000
        if self._start_ts_ns is None:
            self._start_ts_ns = int(batch.timestamps[0]) * 1000
        self._maybe_fsync()

    def _maybe_fsync(self) -> None:
        """Force buffered events to disk every `fsync_interval_s`.

        The appliance has no UPS: on a power cut everything still in the
        OS page cache is lost. Bounding the unsynced window to a couple of
        seconds caps that loss; `recovery.recover_orphans` then re-indexes
        the file (which never got a clean manifest entry) on next boot.
        """
        if self._file is None:
            return
        now = self._now()
        if now - self._last_fsync < self._fsync_interval_s:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._last_fsync = now

    def _rotate(self, bucket: _Bucket) -> None:
        self._finalise_current()
        year, month, day, hour = bucket
        day_dir = (
            self._root
            / self._camera_id
            / f"{year:04d}-{month:02d}-{day:02d}"
        )
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{hour:02d}0000Z-{self._run_id}.dat"
        f = path.open("wb")
        f.write(DAT_FILE_HEADER)
        self._hasher = hashlib.sha256()
        self._hasher.update(DAT_FILE_HEADER)
        self._size_bytes = len(DAT_FILE_HEADER)
        self._file = f
        self._file_path = path
        self._bucket = bucket
        self._last_fsync = self._now()
        self._start_ts_ns = None
        self._end_ts_ns = 0
        self._object_ids = set()
        _log.info("footage rotated -> %s", path)

    def _finalise_current(self) -> None:
        if self._file is None or self._file_path is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        if self._start_ts_ns is None:
            self._file_path.unlink(missing_ok=True)
            self._reset_state()
            return
        manifest = self._file_path.parent / "manifest.jsonl"
        capture: dict[str, object] = {
            "width": self._width,
            "height": self._height,
            "window_us": self._window_us,
            "step_us": self._step_us,
            "biases": dict(self._biases),
            **self._context,
        }
        entry = ManifestEntry(
            filename=self._file_path.name,
            start_ts_ns=self._start_ts_ns,
            end_ts_ns=self._end_ts_ns,
            size_bytes=self._size_bytes,
            sha256=self._hasher.hexdigest(),
            object_ids=sorted(self._object_ids),
            capture=capture,
        )
        append_entry(manifest, entry)
        self._reset_state()

    def _reset_state(self) -> None:
        self._file = None
        self._file_path = None
        self._bucket = None
        self._start_ts_ns = None
        self._end_ts_ns = 0
        self._size_bytes = 0
        self._object_ids = set()

    @staticmethod
    def _bucket_for(epoch_seconds: float) -> _Bucket:
        t = time.gmtime(epoch_seconds)
        return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour)
