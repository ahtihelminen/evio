"""Boot-time recovery for footage files that never got a manifest entry.

A `FootageRecorder` appends a `ManifestEntry` only when it closes a file
cleanly (`_finalise_current`). A power cut therefore leaves the in-flight
`.dat` on disk but absent from `manifest.jsonl` — invisible to footage
listing, retention sweeps, and the training-data fetcher.

`recover_orphans` runs at startup. For each `.dat` under
`<root>/<camera>/<date>/` missing from that day's manifest it validates the
header, truncates any trailing partial DT8 record (the few bytes a power cut
can leave half-written) back to the last whole 8-byte record, and appends a
reconstructed `ManifestEntry`. Object-ids and the bias/scene capture block
are lost with the recorder's in-memory state, so the recovered entry carries
`{"recovered": True}` and no object-ids.

Re-hashing a multi-GB orphan is slow, so the runner runs this on a background
thread while recording already proceeds. Pass `before=<process start time>`
so files the live recorders are still writing (mtime at or after start) are
skipped — only prior-run orphans get re-indexed, never a file another
recorder owns.

`recover_mono_orphans` is the frame-camera counterpart for `.mraw` files
(see `mono_footage_recorder`): same idea, but the record stride is read
from the file's self-describing header rather than being a fixed 8 bytes.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from pathlib import Path
from typing import BinaryIO

from evio.recording.footage_recorder import DAT_FILE_HEADER
from evio.recording.manifest import ManifestEntry, append_entry, iter_manifest
from evio.recording.mono_footage_recorder import (
    MONO_GEOM,
    MONO_MAGIC,
    MONO_TS,
    mono_frame_stride,
)

_log = logging.getLogger(__name__)

_DT8_RECORD_BYTES = 8
_HEADER_LEN = len(DAT_FILE_HEADER)
_T32 = struct.Struct("<I")  # leading uint32 of each DT8 record (µs timestamp)


def recover_orphans(root: Path, *, before: float | None = None) -> list[Path]:
    """Re-index footage `.dat` files missing from their day manifest.

    Returns the paths re-indexed (a manifest entry appended). A missing
    root, or a tree with nothing orphaned, yields `[]`. `before` (an epoch
    seconds cutoff) skips files modified at or after it: when recovery runs
    concurrently with live recording, that excludes the files the current
    run's recorders still own.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    day_dirs = {dat.parent for dat in root.glob("*/*/*.dat")}
    recovered: list[Path] = []
    for day_dir in sorted(day_dirs):
        manifest_path = day_dir / "manifest.jsonl"
        known = {entry.filename for entry in iter_manifest(manifest_path)}
        for dat in sorted(day_dir.glob("*.dat")):
            if dat.name in known:
                continue
            if before is not None and dat.stat().st_mtime >= before:
                continue
            entry = _reindex(dat)
            if entry is not None:
                append_entry(manifest_path, entry)
                recovered.append(dat)
                _log.info("recovered orphaned footage %s", dat)
    return recovered


def _trailing_zero_records(f: BinaryIO, n_whole: int) -> int:
    """Count the run of all-zero DT8 records at the end of the file."""
    block = 1 << 20  # records per backward step (8 MB reads)
    zeros = 0
    end = n_whole
    while end > 0:
        start = max(0, end - block)
        f.seek(_HEADER_LEN + start * _DT8_RECORD_BYTES)
        chunk = f.read((end - start) * _DT8_RECORD_BYTES)
        stripped = len(chunk.rstrip(b"\x00"))
        # zero bytes after the last nonzero byte, rounded DOWN to whole
        # records (a record merely ENDING in zeros is not a zero record)
        zeros += (len(chunk) - stripped) // _DT8_RECORD_BYTES if stripped \
            else (end - start)
        if stripped:
            break
        end = start
    return zeros


def _reindex(path: Path) -> ManifestEntry | None:
    """Validate, truncate-to-whole-record, and rebuild a manifest entry.

    Returns None (and may unlink the file) when there is nothing to index:
    a foreign/corrupt header, or a header-only file with no events.
    """
    if path.stat().st_size < _HEADER_LEN:
        _log.warning("skipping %s: shorter than the footage header", path)
        return None
    with path.open("r+b") as f:
        if f.read(_HEADER_LEN) != DAT_FILE_HEADER:
            _log.warning("skipping %s: not an evio footage file", path)
            return None
        data_bytes = path.stat().st_size - _HEADER_LEN
        n_whole = data_bytes // _DT8_RECORD_BYTES
        trailing = data_bytes % _DT8_RECORD_BYTES
        if trailing:
            f.truncate(_HEADER_LEN + n_whole * _DT8_RECORD_BYTES)
            _log.warning(
                "truncated %s: dropped %d trailing partial byte(s)",
                path, trailing,
            )
        # The mmap recorder preallocates 256 MB extents; a power cut leaves
        # a ZERO tail that would decode as millions of fake t=0 events at
        # (0,0). A real all-zero DT8 record is vanishingly rare, so trim the
        # trailing zero run (backward, block-wise — never loads the file).
        if n_whole:
            zeros = _trailing_zero_records(f, n_whole)
            if zeros:
                n_whole -= zeros
                f.truncate(_HEADER_LEN + n_whole * _DT8_RECORD_BYTES)
                _log.warning(
                    "truncated %s: dropped %d zero-filled record(s) "
                    "(preallocated extent tail)", path, zeros,
                )
        if n_whole == 0:
            f.close()
            path.unlink(missing_ok=True)
            _log.info("removed header-only footage %s", path)
            return None
        f.seek(_HEADER_LEN)
        first_us = _T32.unpack(f.read(_T32.size))[0]
        f.seek(_HEADER_LEN + (n_whole - 1) * _DT8_RECORD_BYTES)
        last_us = _T32.unpack(f.read(_T32.size))[0]

    return ManifestEntry(
        filename=path.name,
        start_ts_ns=min(first_us, last_us) * 1000,
        end_ts_ns=max(first_us, last_us) * 1000,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        object_ids=[],
        capture={"recovered": True},
    )


_MONO_HEADER_LEN = len(MONO_MAGIC) + MONO_GEOM.size


def recover_mono_orphans(
    root: Path, *, before: float | None = None,
) -> list[Path]:
    """Re-index mono `.mraw` files missing from their day manifest.

    Frame-camera counterpart of `recover_orphans`. Returns the paths
    re-indexed; a missing root or a tree with nothing orphaned yields `[]`.
    `before` skips files modified at or after it (the live recorder's own).
    """
    root = Path(root)
    if not root.is_dir():
        return []
    day_dirs = {mraw.parent for mraw in root.glob("*/*/*.mraw")}
    recovered: list[Path] = []
    for day_dir in sorted(day_dirs):
        manifest_path = day_dir / "manifest.jsonl"
        known = {entry.filename for entry in iter_manifest(manifest_path)}
        for mraw in sorted(day_dir.glob("*.mraw")):
            if mraw.name in known:
                continue
            if before is not None and mraw.stat().st_mtime >= before:
                continue
            entry = _reindex_mono(mraw)
            if entry is not None:
                append_entry(manifest_path, entry)
                recovered.append(mraw)
                _log.info("recovered orphaned mono footage %s", mraw)
    return recovered


def _reindex_mono(path: Path) -> ManifestEntry | None:
    """Validate, truncate-to-whole-frame, and rebuild a manifest entry.

    The frame stride comes from the file's own header, so a corrupt or
    foreign header is skipped (it would give a meaningless stride).
    """
    if path.stat().st_size < _MONO_HEADER_LEN:
        # A rotate that died mid-header (ENOSPC, power cut) leaves a stub
        # that can never be indexed: remove it instead of re-warning every
        # boot. Foreign content (magic-prefix mismatch) is left untouched.
        with path.open("rb") as f:
            prefix = f.read(len(MONO_MAGIC))
        if MONO_MAGIC.startswith(prefix):
            path.unlink(missing_ok=True)
            _log.info("removed sub-header mono stub %s", path)
        else:
            _log.warning("skipping %s: shorter than the mono header", path)
        return None
    with path.open("r+b") as f:
        if f.read(len(MONO_MAGIC)) != MONO_MAGIC:
            _log.warning("skipping %s: not an evio mono footage file", path)
            return None
        width, height, channels = MONO_GEOM.unpack(f.read(MONO_GEOM.size))
        stride = mono_frame_stride(width, height, channels)
        data_bytes = path.stat().st_size - _MONO_HEADER_LEN
        n_whole = data_bytes // stride
        trailing = data_bytes % stride
        if trailing:
            f.truncate(_MONO_HEADER_LEN + n_whole * stride)
            _log.warning(
                "truncated %s: dropped %d trailing partial byte(s)",
                path, trailing,
            )
        if n_whole == 0:
            f.close()
            path.unlink(missing_ok=True)
            _log.info("removed header-only mono footage %s", path)
            return None
        f.seek(_MONO_HEADER_LEN)
        first_ns = MONO_TS.unpack(f.read(MONO_TS.size))[0]
        f.seek(_MONO_HEADER_LEN + (n_whole - 1) * stride)
        last_ns = MONO_TS.unpack(f.read(MONO_TS.size))[0]

    return ManifestEntry(
        filename=path.name,
        start_ts_ns=min(first_ns, last_ns),
        end_ts_ns=max(first_ns, last_ns),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        object_ids=[],
        capture={
            "format": "mono-raw",
            "width": width,
            "height": height,
            "channels": channels,
            "recovered": True,
        },
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["recover_mono_orphans", "recover_orphans"]
