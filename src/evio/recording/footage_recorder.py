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

Durability model (2026-07-03): the hot path appends into a PREALLOCATED
MEMORY-MAPPED extent — a pure memcpy that cannot raise transient I/O
errors; the kernel writes dirty pages back even while the process is
busy, and a periodic msync bounds power-cut loss like the old fsync did.
The only fallible operations left are extent allocation and rotation,
and those RETRY with backoff while batches wait in a bounded pending
buffer — recording NEVER permanently stops (the 2026-07-03 incident:
one transient OSError silently disabled all six recorders mid-flight).
``degraded``/``last_error``/``dropped_bytes`` expose the state for the
status line.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import mmap
import os
import time
from collections.abc import Callable
from pathlib import Path

from evio.codec.dt8 import Dt8Codec
from evio.core.events import EventsBatch
from evio.recording.manifest import ManifestEntry, append_entry

DAT_FILE_HEADER = b"% evio footage recorder\n\x00\x08"

_log = logging.getLogger(__name__)
_TimeFn = Callable[[], float]
_Bucket = tuple[int, int, int, int]  # (year, month, day, hour) in UTC

_EXTENT_BYTES = 256 * 1024 * 1024  # preallocation quantum
_PENDING_CAP_BYTES = 512 * 1024 * 1024  # RAM buffer while disk is degraded
_RETRY_BASE_S = 1.0
_RETRY_MAX_S = 10.0


def _preallocate(fd: int, offset: int, length: int) -> None:
    """Really allocate ``[offset, offset+length)`` — never a sparse hole.

    A sparse hole would SIGBUS the mmap write on ENOSPC; a true
    allocation fails HERE, where the retry/backoff policy can handle it.
    """
    if hasattr(os, "posix_fallocate"):  # Linux (the rig)
        os.posix_fallocate(fd, offset, length)
        return
    os.lseek(fd, offset, os.SEEK_SET)  # macOS dev box: zero-fill
    block = b"\x00" * (8 << 20)
    remaining = length
    while remaining > 0:
        n = min(remaining, len(block))
        os.write(fd, block[:n])
        remaining -= n


class _MmapExtent:
    """Append-only mmap writer: the hot path is a memcpy, no syscalls."""

    def __init__(self, path: Path, initial: bytes) -> None:
        self._fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        self._capacity = 0
        self._mm: mmap.mmap | None = None
        self.length = 0
        try:
            self._grow(max(_EXTENT_BYTES, len(initial)))
            self.append(initial)
        except BaseException:
            self.abort()
            raise

    def _grow(self, need_total: int) -> None:
        cap = self._capacity
        while cap < need_total:
            cap += _EXTENT_BYTES
        _preallocate(self._fd, self._capacity, cap - self._capacity)
        if self._mm is not None:
            self._mm.close()
        self._mm = mmap.mmap(self._fd, cap)
        self._capacity = cap

    def append(self, payload: bytes) -> None:
        end = self.length + len(payload)
        if end > self._capacity:
            self._grow(end)
        assert self._mm is not None
        self._mm[self.length:end] = payload
        self.length = end

    def flush(self) -> None:
        if self._mm is not None:
            self._mm.flush()  # msync: bounds power-cut loss

    def close(self) -> None:
        """Finalize: flush, unmap, truncate the extent to the true length."""
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        os.ftruncate(self._fd, self.length)
        os.close(self._fd)
        self._fd = -1

    def abort(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            if self._mm is not None:
                self._mm.close()
        with contextlib.suppress(OSError):
            if self._fd >= 0:
                os.close(self._fd)
        self._mm = None
        self._fd = -1


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
        self._ext: _MmapExtent | None = None
        self._file_path: Path | None = None
        # Never-die policy: encoded payloads wait here (bounded) while the
        # disk path is degraded; landed only when the extent accepts them.
        self._pending: list[tuple[bytes, int, int]] = []
        self._pending_bytes = 0
        self.degraded = False
        self.last_error: str | None = None
        self.dropped_bytes = 0
        self._retry_at = 0.0
        self._backoff_s = _RETRY_BASE_S
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
        """Append a batch. NEVER raises for I/O reasons.

        Encoding happens immediately (cannot fail on I/O); the payload
        lands in the mmap extent, or — while the disk path is degraded —
        waits in the bounded pending buffer and is retried with backoff.
        Recording only loses data if the disk stays broken long enough to
        overflow the RAM buffer, and then it drops OLDEST-first with an
        exact ``dropped_bytes`` count. It never silently stops.
        """
        if batch.timestamps.size == 0:
            return
        words = self._codec.encode_chunk(batch)
        if words.size == 0:
            return
        self._pending.append((
            words.tobytes(),
            int(batch.timestamps[0]),
            int(batch.timestamps[-1]),
        ))
        self._pending_bytes += len(self._pending[-1][0])
        while self._pending_bytes > _PENDING_CAP_BYTES:
            stale, _, _ = self._pending.pop(0)
            self._pending_bytes -= len(stale)
            self.dropped_bytes += len(stale)
        now = self._now()
        if self.degraded and now < self._retry_at:
            return
        try:
            bucket = self._bucket_for(now)
            if bucket != self._bucket or self._ext is None:
                self._rotate(bucket)
            self._drain_pending()
        except OSError as exc:
            self.degraded = True
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._retry_at = now + self._backoff_s
            self._backoff_s = min(self._backoff_s * 2.0, _RETRY_MAX_S)
            _log.warning(
                "footage write degraded (%s); retrying in %.0fs with "
                "%d bytes pending", self.last_error,
                self._retry_at - now, self._pending_bytes,
            )
        else:
            if self.degraded:
                _log.warning("footage recording recovered")
            self.degraded = False
            self._backoff_s = _RETRY_BASE_S

    def _drain_pending(self) -> None:
        while self._pending:
            payload, first_us, last_us = self._pending[0]
            self._append(payload, first_us, last_us)
            self._pending.pop(0)
            self._pending_bytes -= len(payload)

    def mark_object_id(self, object_id: str) -> None:
        if self._ext is not None and object_id:
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
        # Best-effort: provenance must never take the recording down.
        with contextlib.suppress(OSError):
            self._provenance_path.parent.mkdir(parents=True, exist_ok=True)
            with self._provenance_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        self._finalise_current()

    @property
    def current_path(self) -> Path | None:
        return self._file_path

    def _append(self, payload: bytes, first_us: int, last_us: int) -> None:
        if self._ext is None:
            msg = "_append called with no open extent"
            raise RuntimeError(msg)
        self._ext.append(payload)
        self._size_bytes += len(payload)
        self._hasher.update(payload)
        self._end_ts_ns = last_us * 1000
        if self._start_ts_ns is None:
            self._start_ts_ns = first_us * 1000
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        """msync the extent every `fsync_interval_s`.

        The appliance has no UPS: on a power cut dirty pages are lost.
        Bounding the unsynced window to a couple of seconds caps that
        loss; `recovery.recover_orphans` re-indexes the file (trimming
        the preallocated zero tail) on next boot.
        """
        if self._ext is None:
            return
        now = self._now()
        if now - self._last_fsync < self._fsync_interval_s:
            return
        self._ext.flush()
        self._last_fsync = now

    def _rotate(self, bucket: _Bucket) -> None:
        # No drain here: pending payloads belong to the NEW bucket.
        self._finalise_current(drain=False)
        year, month, day, hour = bucket
        day_dir = (
            self._root
            / self._camera_id
            / f"{year:04d}-{month:02d}-{day:02d}"
        )
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{hour:02d}0000Z-{self._run_id}.dat"
        self._ext = _MmapExtent(path, DAT_FILE_HEADER)
        self._hasher = hashlib.sha256()
        self._hasher.update(DAT_FILE_HEADER)
        self._size_bytes = len(DAT_FILE_HEADER)
        self._file_path = path
        self._bucket = bucket
        self._last_fsync = self._now()
        self._start_ts_ns = None
        self._end_ts_ns = 0
        self._object_ids = set()
        _log.info("footage rotated -> %s", path)

    def _finalise_current(self, *, drain: bool = True) -> None:
        if self._ext is None or self._file_path is None:
            return
        if drain:
            with contextlib.suppress(OSError):
                self._drain_pending()  # last chance for buffered batches
        self._ext.close()  # flush + truncate the extent to true length
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
        self._ext = None
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
