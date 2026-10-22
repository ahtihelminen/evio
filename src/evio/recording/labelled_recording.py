"""One HDF5 file per labelled event recording.

Events are stored as columns with the time delta-encoded, ground-truth boxes
as a second table, the sensor bias log and the reviewer-excluded time spans
as two more, and the provenance as root attributes; a per-chunk time
index makes any window readable without decoding from the start. The layout,
the time convention and the versioning rules are written down in
`docs/modules/labelled-recording.md`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import h5py
import numpy as np

from evio.core.events import EventsBatch

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

FORMAT_NAME = "evio-labelled-recording"
FORMAT_VERSION = "1.1"
FORMAT_MAJOR = 1
# Events per HDF5 chunk and per index bookmark: a seek decodes at most one
# partial chunk on each side, a writer buffers less than one chunk.
CHUNK_EVENTS = 1 << 18
LABEL_CHUNK_ROWS = 1 << 16
INDEX_CHUNK_ROWS = 1 << 12
TABLE_CHUNK_ROWS = 1 << 12

_EVENT_DTYPES = {
    "dt_us": np.uint64,
    "x": np.int16,
    "y": np.int16,
    "polarity": np.int8,
}
_LABEL_DTYPES = {
    "time_us": np.int64,
    "object_id": np.int64,
    "class_id": np.int32,
    "box": np.float64,
}
_BIAS_DTYPES = {
    "time_us": np.int64,
    "name_id": np.int32,
    "value": np.int64,
}


@dataclass(frozen=True, slots=True)
class RecordingMeta:
    """Provenance of a recording; `time_origin` and `box_frame` are
    documented in `docs/modules/labelled-recording.md`."""

    source_dataset: str
    original_id: str
    licence: str
    sensor: str
    width: int
    height: int
    time_origin: str
    box_frame: str


@dataclass(frozen=True, slots=True)
class RecordingLabels:
    """Boxes `(x0, y0, x1, y1)` per row, on the event clock."""

    time_us: NDArray[np.int64]
    object_id: NDArray[np.int64]
    class_id: NDArray[np.int32]
    box: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BiasLog:
    """Bias changes on the event clock: from `time_us[i]` on, bias
    `names[name_id[i]]` has value `value[i]`. No rows means bias unknown."""

    time_us: NDArray[np.int64]
    name_id: NDArray[np.int32]
    value: NDArray[np.int64]
    names: tuple[str, ...]

    def __len__(self) -> int:
        return self.time_us.size

    def at(self, time_us: int) -> dict[str, int]:
        """The bias vector in force at `time_us`: the last row at or before
        it per name. Names not yet set are absent."""
        rows = np.searchsorted(self.time_us, time_us, side="right")
        return {
            self.names[name_id]: int(value)
            for name_id, value in zip(
                self.name_id[:rows], self.value[:rows], strict=True,
            )
        }


@dataclass(frozen=True, slots=True)
class ExcludedSpans:
    """Disjoint half-open spans `[start_us, end_us)` that a reviewer marked
    as not trustworthy. They may be given in any order and are stored sorted
    by start; overlapping spans raise. No spans means nothing excluded."""

    start_us: NDArray[np.int64]
    end_us: NDArray[np.int64]

    def __post_init__(self) -> None:
        if (self.start_us >= self.end_us).any():
            msg = "excluded spans need start < end"
            raise ValueError(msg)
        order = np.argsort(self.start_us)
        object.__setattr__(self, "start_us", self.start_us[order])
        object.__setattr__(self, "end_us", self.end_us[order])
        if (self.start_us[1:] < self.end_us[:-1]).any():
            msg = "excluded spans must not overlap"
            raise ValueError(msg)

    def mask(self, times_us: NDArray[np.int64]) -> NDArray[np.bool_]:
        """True where a time falls inside an excluded span."""
        # Index 0 is "before every span" and never excludes.
        ends = np.concatenate([[np.iinfo(np.int64).min], self.end_us])
        span = np.searchsorted(self.start_us, times_us, side="right")
        return times_us < ends[span]

    def kept(self, start_us: int, end_us: int) -> list[tuple[int, int]]:
        """The parts of `[start_us, end_us)` outside every excluded span."""
        kept = []
        cursor = start_us
        for span_start, span_end in zip(
            self.start_us.tolist(), self.end_us.tolist(), strict=True,
        ):
            kept.append((cursor, min(span_start, end_us)))
            cursor = max(cursor, span_end)
        kept.append((cursor, end_us))
        return [(a, b) for a, b in kept if a < b]


def _create(
    group: h5py.Group,
    name: str,
    dtype: type[np.generic],
    chunk_rows: int,
    columns: tuple[int, ...] = (),
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=(0, *columns),
        maxshape=(None, *columns),
        dtype=dtype,
        chunks=(chunk_rows, *columns),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _append(dataset: h5py.Dataset, values: NDArray[np.generic]) -> None:
    size = dataset.shape[0]
    dataset.resize(size + len(values), axis=0)
    dataset[size:] = values


class LabelledRecordingWriter:
    """Streams `EventsBatch` batches and labels into a labelled recording.

    Timestamps must be non-decreasing across the whole recording; a batch
    that breaks that raises `ValueError` and is not written (events already
    written stay in the file). Less than one chunk of events is buffered:
    `buffered_events` is always below `chunk_events` between calls.
    """

    def __init__(
        self,
        path: Path,
        meta: RecordingMeta,
        *,
        chunk_events: int = CHUNK_EVENTS,
    ) -> None:
        self._path = path
        self._meta = meta
        self._chunk_events = chunk_events
        self._pending = {
            name: np.empty(0, dtype)
            for name, dtype in {"t_us": np.int64, **_EVENT_DTYPES}.items()
        }
        self._last_t: int | None = None
        self._last_bias_t: int | None = None
        self._bias_ids: dict[str, int] = {}
        self._bias_columns: dict[str, h5py.Dataset] = {}
        self._excluded = ExcludedSpans(
            np.empty(0, np.int64), np.empty(0, np.int64),
        )

    @property
    def buffered_events(self) -> int:
        return self._pending["t_us"].size

    def __enter__(self) -> LabelledRecordingWriter:
        self._file = h5py.File(self._path, "w")
        self._file.attrs.update(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "chunk_events": self._chunk_events,
                **asdict(self._meta),
            }
        )
        events = self._file.create_group("events")
        self._columns = {
            name: _create(events, name, dtype, self._chunk_events)
            for name, dtype in _EVENT_DTYPES.items()
        }
        self._index = _create(
            self._file.create_group("index"),
            "t_us",
            np.int64,
            INDEX_CHUNK_ROWS,
        )
        labels = self._file.create_group("labels")
        self._label_columns = {
            name: _create(
                labels,
                name,
                dtype,
                LABEL_CHUNK_ROWS,
                (4,) if name == "box" else (),
            )
            for name, dtype in _LABEL_DTYPES.items()
        }
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._flush(self.buffered_events)
        self._write_bias_names()
        self._write_excluded()
        self._file.close()

    def write_events(self, batch: EventsBatch) -> None:
        times = batch.timestamps
        if times.size == 0:
            return
        first = times[0] if self._last_t is None else self._last_t
        deltas = np.diff(times, prepend=first)
        if (deltas < 0).any():
            msg = "timestamps must be non-decreasing across the recording"
            raise ValueError(msg)
        self._last_t = int(times[-1])
        new = {
            "t_us": times,
            "dt_us": deltas,
            "x": batch.x_coords,
            "y": batch.y_coords,
            "polarity": batch.polarities,
        }
        dtypes = {"t_us": np.int64, **_EVENT_DTYPES}
        self._pending = {
            name: np.concatenate(
                [self._pending[name], new[name].astype(dtypes[name])]
            )
            for name in new
        }
        self._flush(self.buffered_events // self._chunk_events
                    * self._chunk_events)

    def write_labels(self, labels: RecordingLabels) -> None:
        for name, dataset in self._label_columns.items():
            _append(dataset, getattr(labels, name).astype(_LABEL_DTYPES[name]))

    def write_bias(self, rows: Iterable[tuple[int, str, int]]) -> None:
        """Appends `(time_us, name, value)` rows; times must be
        non-decreasing across the recording, like events."""
        rows = list(rows)
        if not rows:
            return
        times, names, values = zip(*rows, strict=True)
        times = np.array(times, np.int64)
        deltas = np.diff(times, prepend=times[0] if self._last_bias_t is None
                         else self._last_bias_t)
        if (deltas < 0).any():
            msg = "bias times must be non-decreasing across the recording"
            raise ValueError(msg)
        self._last_bias_t = int(times[-1])
        ids = [self._bias_ids.setdefault(name, len(self._bias_ids))
               for name in names]
        if not self._bias_columns:
            group = self._file.create_group("bias")
            self._bias_columns = {
                name: _create(group, name, dtype, TABLE_CHUNK_ROWS)
                for name, dtype in _BIAS_DTYPES.items()
            }
        new = {"time_us": times, "name_id": ids, "value": values}
        for name, dataset in self._bias_columns.items():
            _append(dataset, np.array(new[name], _BIAS_DTYPES[name]))

    def write_excluded(self, spans: ExcludedSpans) -> None:
        """Adds spans, in any order; they must not overlap each other or
        earlier ones. A rejected call adds nothing."""
        starts = np.concatenate([self._excluded.start_us, spans.start_us])
        ends = np.concatenate([self._excluded.end_us, spans.end_us])
        self._excluded = ExcludedSpans(starts, ends)

    def _write_bias_names(self) -> None:
        if self._bias_ids:
            self._file["bias"].create_dataset(
                "names",
                data=np.array([name.encode() for name in self._bias_ids]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

    def _write_excluded(self) -> None:
        if self._excluded.start_us.size:
            group = self._file.create_group("excluded")
            for name, values in (("start_us", self._excluded.start_us),
                           ("end_us", self._excluded.end_us)):
                _append(_create(group, name, np.int64, TABLE_CHUNK_ROWS),
                        values)

    def _flush(self, count: int) -> None:
        for name, dataset in self._columns.items():
            _append(dataset, self._pending[name][:count])
        _append(self._index, self._pending["t_us"][:count:self._chunk_events])
        self._pending = {
            name: values[count:].copy()
            for name, values in self._pending.items()
        }


class LabelledRecordingReader:
    """Reads a labelled recording; seeks by time through the stored index."""

    def __init__(self, path: Path) -> None:
        self._file = h5py.File(path, "r")
        try:
            self._require_readable()
        except ValueError:
            self._file.close()
            raise
        attrs = self._file.attrs
        self.chunk_events = int(attrs["chunk_events"])
        values = {f.name: attrs[f.name] for f in fields(RecordingMeta)}
        self.meta = RecordingMeta(
            **{
                **values,
                "width": int(values["width"]),
                "height": int(values["height"]),
            }
        )
        self._index = self._file["index/t_us"][:]
        self._columns = {name: self._file[f"events/{name}"]
                         for name in _EVENT_DTYPES}

    def _require_readable(self) -> None:
        name = self._file.attrs.get("format")
        version = str(self._file.attrs.get("format_version", ""))
        major = int(version.split(".")[0] or 0)
        if name != FORMAT_NAME or major > FORMAT_MAJOR:
            msg = (
                f"cannot read {name!r} format version {version!r}: "
                f"this evio reads {FORMAT_NAME} format version "
                f"{FORMAT_MAJOR}.x"
            )
            raise ValueError(msg)

    def __enter__(self) -> LabelledRecordingReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    @property
    def n_events(self) -> int:
        return self._columns["x"].shape[0]

    def labels(self) -> RecordingLabels:
        return RecordingLabels(
            **{name: self._file[f"labels/{name}"][:]
               for name in _LABEL_DTYPES}
        )

    def bias_log(self) -> BiasLog:
        """The recorded bias changes; empty when bias was not recorded."""
        if "bias" not in self._file:
            return BiasLog(
                np.empty(0, np.int64), np.empty(0, np.int32),
                np.empty(0, np.int64), (),
            )
        group = self._file["bias"]
        return BiasLog(
            *(group[name][:] for name in _BIAS_DTYPES),
            tuple(name.decode() for name in group["names"][:]),
        )

    def bias_at(self, time_us: int) -> dict[str, int]:
        return self.bias_log().at(time_us)

    def excluded_spans(self) -> ExcludedSpans:
        """The reviewer-excluded spans; empty when nothing is excluded."""
        if "excluded" not in self._file:
            return ExcludedSpans(np.empty(0, np.int64), np.empty(0, np.int64))
        group = self._file["excluded"]
        return ExcludedSpans(group["start_us"][:], group["end_us"][:])

    def events(self, start_us: int, end_us: int) -> EventsBatch:
        """Events with `start_us <= t < end_us`, reading only the chunks
        that can hold them."""
        first = max(np.searchsorted(self._index, start_us) - 1, 0)
        last = max(np.searchsorted(self._index, end_us), first)
        batch = self._read_chunks(first, last)
        keep = (batch.timestamps >= start_us) & (batch.timestamps < end_us)
        return EventsBatch(
            batch.timestamps[keep],
            batch.x_coords[keep],
            batch.y_coords[keep],
            batch.polarities[keep],
        )

    def iter_events(self) -> Iterator[EventsBatch]:
        """Every event in order, one batch per stored chunk."""
        for chunk in range(len(self._index)):
            yield self._read_chunks(chunk, chunk + 1)

    def _read_chunks(self, first: int, last: int) -> EventsBatch:
        rows = slice(first * self.chunk_events, last * self.chunk_events)
        deltas = self._columns["dt_us"][rows].astype(np.int64)
        # The bookmark, not the delta to the previous chunk, anchors the time.
        deltas[:1] = 0
        times = (self._index[first:first + 1] + np.cumsum(deltas)
                 if deltas.size else deltas)
        return EventsBatch(
            times,
            self._columns["x"][rows],
            self._columns["y"][rows],
            self._columns["polarity"][rows],
        )
