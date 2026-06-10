"""Tees mono-camera frames into hour-bucketed raw `.mraw` files.

Frame-based sibling of `FootageRecorder`. A frame camera produces fixed-
size images, so the container is dead simple: a self-describing header
(magic + width/height/channels) followed by fixed-stride records, each an
8-byte little-endian capture timestamp (wall-clock ns) then the raw frame
bytes. The fixed stride is what makes a power-cut recoverable —
`recovery.recover_mono_orphans` truncates a half-written trailing frame
back to the last whole one, exactly as the DT8 path does for events.

Raw (uncompressed) on purpose: an MP4 writes its index only on a clean
close, so a plug-pull leaves an unplayable file — the opposite of the
appliance's crash-safety goal. Raw costs disk (Mono8 is W*H*fps B/s) but
recovers to the last whole frame with no codec state to rebuild.

Path layout:

    <root>/<camera_id>/YYYY-MM-DD/HH0000Z-<run>.mraw
    <root>/<camera_id>/YYYY-MM-DD/manifest.jsonl
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np

from evio.recording.manifest import ManifestEntry, append_entry

MONO_MAGIC = b"% evio mono footage\n"
MONO_GEOM = struct.Struct("<HHB")  # width, height, channels
MONO_TS = struct.Struct("<Q")  # per-frame capture timestamp, ns

_log = logging.getLogger(__name__)
_TimeFn = Callable[[], float]
_Bucket = tuple[int, int, int, int]  # (year, month, day, hour) in UTC


def mono_frame_stride(width: int, height: int, channels: int) -> int:
    """Bytes per on-disk record: the ns timestamp plus the raw frame."""
    return MONO_TS.size + width * height * channels


class MonoFootageRecorder:
    def __init__(
        self,
        *,
        root: Path,
        camera_id: str,
        width: int,
        height: int,
        channels: int = 1,
        fps: float = 0.0,
        fsync_interval_s: float = 2.0,
        time_provider: _TimeFn | None = None,
    ) -> None:
        self._root = Path(root)
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._channels = channels
        self._fps = fps
        self._payload_bytes = width * height * channels
        self._fsync_interval_s = fsync_interval_s
        self._now: _TimeFn = time_provider or time.time
        self._header = MONO_MAGIC + MONO_GEOM.pack(width, height, channels)
        # Per-process suffix so a re-run inside the same UTC hour opens a new
        # file rather than clobbering the previous one.
        self._run_id = time.strftime("%H%M%SZ", time.gmtime(self._now()))
        self._file: BinaryIO | None = None
        self._file_path: Path | None = None
        self._bucket: _Bucket | None = None
        self._start_ts_ns: int | None = None
        self._end_ts_ns = 0
        self._size_bytes = 0
        self._hasher = hashlib.sha256()
        self._last_fsync = self._now()

    def write(self, frame: np.ndarray, ts_ns: int) -> None:
        if frame.size != self._payload_bytes:
            msg = (
                f"frame has {frame.size} bytes, expected "
                f"{self._payload_bytes} for "
                f"{self._width}x{self._height}x{self._channels}"
            )
            raise ValueError(msg)
        bucket = self._bucket_for(self._now())
        if bucket != self._bucket:
            self._rotate(bucket)
        self._append(frame, ts_ns)

    def close(self) -> None:
        self._finalise_current()

    @property
    def current_path(self) -> Path | None:
        return self._file_path

    def _append(self, frame: np.ndarray, ts_ns: int) -> None:
        if self._file is None:
            msg = "_append called with no open file"
            raise RuntimeError(msg)
        pixels = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        record = MONO_TS.pack(ts_ns) + pixels
        self._file.write(record)
        self._size_bytes += len(record)
        self._hasher.update(record)
        self._end_ts_ns = ts_ns
        if self._start_ts_ns is None:
            self._start_ts_ns = ts_ns
        self._maybe_fsync()

    def _maybe_fsync(self) -> None:
        """Force buffered frames to disk every `fsync_interval_s`.

        The appliance has no UPS: on a power cut everything still in the OS
        page cache is lost. Bounding the unsynced window caps that loss;
        `recover_mono_orphans` then re-indexes the orphaned file on boot.
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
            self._root / self._camera_id / f"{year:04d}-{month:02d}-{day:02d}"
        )
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{hour:02d}0000Z-{self._run_id}.mraw"
        f = path.open("wb")
        f.write(self._header)
        self._hasher = hashlib.sha256()
        self._hasher.update(self._header)
        self._size_bytes = len(self._header)
        self._file = f
        self._file_path = path
        self._bucket = bucket
        self._last_fsync = self._now()
        self._start_ts_ns = None
        self._end_ts_ns = 0
        _log.info("mono footage rotated -> %s", path)

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
        entry = ManifestEntry(
            filename=self._file_path.name,
            start_ts_ns=self._start_ts_ns,
            end_ts_ns=self._end_ts_ns,
            size_bytes=self._size_bytes,
            sha256=self._hasher.hexdigest(),
            capture={
                "format": "mono-raw",
                "width": self._width,
                "height": self._height,
                "channels": self._channels,
                "fps": self._fps,
            },
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

    @staticmethod
    def _bucket_for(epoch_seconds: float) -> _Bucket:
        t = time.gmtime(epoch_seconds)
        return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour)
