# Labelled Recording Format (`src/evio/recording/labelled_recording.py`)

One HDF5 file holds one labelled event recording: its events, its
ground-truth boxes and its provenance. Every source (public datasets, OPS
captures) is converted to this, so consumers handle one layout and one time
convention. Plain `h5py` reads it; the only filters are HDF5's built-in
shuffle and gzip.

Files covered:

- `src/evio/recording/labelled_recording.py`
- `tests/test_labelled_recording.py`, `tests/test_labelled_recording_footage.py`

Public names (all exported from `evio.recording`): `LabelledRecordingWriter`,
`LabelledRecordingReader`, `RecordingMeta`, `RecordingLabels`,
`CHUNK_EVENTS`, `FORMAT_NAME`, `FORMAT_VERSION`.

## Layout (format `evio-labelled-recording`, version `1.0`)

Root attributes (HDF5 attributes on `/`):

| Attribute | Type | Meaning |
|---|---|---|
| `format` | str | always `evio-labelled-recording` |
| `format_version` | str | `major.minor`, here `1.0` |
| `chunk_events` | int64 | events per chunk and per index bookmark in this file |
| `source_dataset` | str | e.g. `FRED`, `Ev-Flying`, `NeRDD`, `ops-capture` |
| `original_id` | str | the recording's id in the source |
| `licence` | str | licence of the source data |
| `sensor` | str | sensor name |
| `width`, `height` | int64 | sensor resolution in pixels |
| `time_origin` | str | what `t = 0` of the event clock is (see below) |
| `box_frame` | str | pixel frame of the boxes: `event` or `rgb` |

Datasets (all chunked, `gzip` level 4 with `shuffle`, resizable, nothing
else; no extra attributes):

| Dataset | dtype | Shape | Meaning |
|---|---|---|---|
| `events/dt_us` | uint64 | (n,) | time delta to the previous event, in microseconds; `0` for event 0 |
| `events/x` | int16 | (n,) | pixel column |
| `events/y` | int16 | (n,) | pixel row |
| `events/polarity` | int8 | (n,) | polarity as given (evio uses 0/1); stored verbatim |
| `index/t_us` | int64 | (c,) | absolute time of event `k * chunk_events`, for chunk `k` |
| `labels/time_us` | int64 | (m,) | label time, on the event clock |
| `labels/object_id` | int64 | (m,) | object id, opaque to the format |
| `labels/class_id` | int32 | (m,) | class id, opaque to the format |
| `labels/box` | float64 | (m, 4) | `x0, y0, x1, y1` in `box_frame` pixels, exactly as the source gives them |

`events/*` are chunked `(chunk_events,)`, `labels/*` `(65536,)` rows,
`index/t_us` `(4096,)`. `n` and `m` are the dataset lengths. `CHUNK_EVENTS`
is `2**18` = 262144 events: a seek decodes at most one extra chunk at either
end (about 0.26 s of a 1 Mev/s camera), the writer buffers under 5 MiB, and
the index of an hour of 1 Mev/s events is under 14k entries. Tests pass a
small `chunk_events`; readers use the value stored in the file.

Events are stored in the order given and are non-decreasing in time, so
absolute time is `index/t_us[0] + cumsum(dt_us)`. Labels are stored in the
order given (not necessarily sorted). An empty recording has zero-length
datasets, and a recording without labels has zero label rows.

## Time convention

Time is int64 microseconds on one clock, for events and labels alike.
`time_origin` records what `t = 0` of that clock is:

- `first_event`: the first event of the recording is at `t = 0`.
- `sensor_power_on`: the sensor's own clock, zero at stream start (OPS).
- `dataset`: the source dataset's own time zero.

The writer does not rebase or interpret times: the converter supplies events
and labels already on one clock, and the field says which. A reader never
converts. Times are the unwrapped int64 clock: DT8 `.dat` files wrap their
32-bit timestamp every 2**32 us, and `evio.core.recording` unwraps it before
the data reaches this format. Timestamps may start anywhere (also negative)
and gaps may exceed 32 bit.

## Seeking

`index/t_us[k]` is the absolute time of the first event of chunk `k`. To read
`[start_us, end_us)`:

1. `first = max(searchsorted(index, start_us, 'left') - 1, 0)`. The `- 1`
   matters: duplicate timestamps equal to `start_us` can sit at the end of
   the chunk before the bookmark.
2. `last = searchsorted(index, end_us, 'left')`: chunks `first .. last - 1`.
3. Read those rows, set the first delta to 0 and add `index[first]`: the
   bookmark, not the first delta (which is relative to the previous chunk),
   anchors the time.
4. Keep `start_us <= t < end_us`.

`LabelledRecordingReader.events(start_us, end_us)` does exactly this, and
HDF5 decompresses only the chunks it touches.

## Versioning

`format_version` is `major.minor`. A minor bump only adds datasets or
attributes, so an older reader still reads the file. Any change to an
existing name, dtype or meaning bumps the major. A reader refuses another
format name or a newer major with a `ValueError` naming both versions.

## Reading with h5py only

The test suite executes this block (with `path`, `start_us` and `end_us`
defined), so it cannot rot.

<!-- docs-example -->
```python
import h5py
import numpy as np

with h5py.File(path, "r") as f:
    chunk = f.attrs["chunk_events"]
    index = f["index/t_us"][:]
    first = max(np.searchsorted(index, start_us) - 1, 0)
    last = max(np.searchsorted(index, end_us), first)
    rows = slice(first * chunk, last * chunk)
    dt = f["events/dt_us"][rows].astype(np.int64)
    dt[:1] = 0
    t = index[first:first + 1] + np.cumsum(dt)
    keep = (t >= start_us) & (t < end_us)
    x, y, p = (f[f"events/{c}"][rows][keep] for c in ("x", "y", "polarity"))
    t = t[keep]
```

## Writing from an OPS capture

```python
with LabelledRecordingWriter(path, meta) as writer:
    for batch in batches:           # EventsBatch, time non-decreasing
        writer.write_events(batch)
    writer.write_labels(RecordingLabels(time_us, object_id, class_id, box))
```

The caller builds the label arrays from its own rows, for example
`(timestamp_us, track_id, class, x0, y0, x1, y1)` per detection or rig track.
SAPIENT track ids are ULID strings: the caller maps them to ints (and class
labels to ints) because the format stores ids as int64. A writer buffers less
than one chunk (`buffered_events`); a batch that goes back in time raises
`ValueError` and is not written. `tests/test_labelled_recording_footage.py`
runs footage recorded by `FootageRecorder`, decoded with the DT8 codec,
through the writer.

## Known limits

- Boxes only: per-event class labels (Ev-Flying) are not stored in 1.0.
- No class-name table and no free-form attributes; ids are agreed per source.
- No frames or RGB data, only the boxes' frame is named.
- Timestamps must fit int64; a single delta must stay below 2**63 us.
