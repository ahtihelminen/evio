"""One HDF5 file per labelled event recording.

Events are stored as columns with the time delta-encoded, ground-truth boxes
as a second table and the provenance as root attributes; a per-chunk time
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
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

FORMAT_NAME = "evio-labelled-recording"
FORMAT_VERSION = "1.0"
FORMAT_MAJOR = 1
# Events per HDF5 chunk and per index bookmark: a seek decodes at most one
# partial chunk on each side, a writer buffers less than one chunk.
CHUNK_EVENTS = 1 << 18
LABEL_CHUNK_ROWS = 1 << 16
INDEX_CHUNK_ROWS = 1 << 12

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
