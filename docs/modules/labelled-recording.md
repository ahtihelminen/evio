# Labelled Recording Format (`src/evio/recording/labelled_recording.py`)

One HDF5 file holds one labelled event recording: its events, its
ground-truth boxes, its sensor bias log, the spans a reviewer excluded and its
provenance. Every source (public datasets, OPS
captures) is converted to this, so consumers handle one layout and one time
convention. Plain `h5py` reads it; the only filters are HDF5's built-in
shuffle and gzip.

Files covered:

- `src/evio/recording/labelled_recording.py`
- `tests/test_labelled_recording.py`, `tests/test_labelled_recording_footage.py`

Public names (all exported from `evio.recording`): `LabelledRecordingWriter`,
`LabelledRecordingReader`, `RecordingMeta`, `RecordingLabels`, `BiasLog`,
`ExcludedSpans`, `CHUNK_EVENTS`, `FORMAT_NAME`, `FORMAT_VERSION`.

## Layout (format `evio-labelled-recording`, version `1.1`)

Root attributes (HDF5 attributes on `/`):

| Attribute | Type | Meaning |
|---|---|---|
| `format` | str | always `evio-labelled-recording` |
| `format_version` | str | `major.minor`, here `1.1` |
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

Format 1.1 adds two optional groups, each present only when it has rows (all
`int` columns are chunked `(4096,)`, the same filters):

| Dataset | dtype | Shape | Meaning |
|---|---|---|---|
| `bias/time_us` | int64 | (r,) | from this time on, on the event clock, non-decreasing |
| `bias/name_id` | int32 | (r,) | row index into `bias/names` |
| `bias/value` | int64 | (r,) | the bias value from that time on |
| `bias/names` | `S<k>` | (u,) | bias names as UTF-8, in order of first appearance; `k` fits the longest |
| `excluded/start_us` | int64 | (s,) | start of an excluded span, inclusive; sorted |
| `excluded/end_us` | int64 | (s,) | end of the span, exclusive; `start_us < end_us`, spans disjoint |

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

## Bias log

Sensor biases (for the IMX636 e.g. `bias_diff`, `bias_diff_on`,
`bias_diff_off`, `bias_fo`, `bias_hpf`, `bias_refr`; the format does not fix
the set, names are the caller's strings) are part of what a recording means
and change during a bias sweep. The log holds one row `(time_us, name, value)`
per change: from `time_us` on, bias `name` has `value`. Rows are stored in the
order written, with non-decreasing time like events (the writer raises
`ValueError` and writes nothing from a batch that goes back); several rows may
share a time and the last one wins. The initial state is rows at or before the
first event. `bias/names` is written when the writer closes (so a file whose
writer crashed has bias rows but no names table); `name_id` indexes it, so a
name that is a prefix of another is no problem.

No `bias` group, or zero rows, means bias was not recorded: unknown, not
"default". A consumer tells recorded from unknown by `len(reader.bias_log())
> 0`. `reader.bias_at(t)` (or `BiasLog.at`) returns `{name: value}` in force
at `t`: the last row at or before `t` per name. Before the first row it is
`{}`; a name set only later is absent until then.

```python
writer.write_bias([(0, "bias_diff", 10), (5_000_000, "bias_diff", 12)])
```

Called repeatedly, like `write_labels`.

## Excluded spans

A human reviewer marks time ranges where the labels are wrong or missing. The
format only records them as half-open `[start_us, end_us)` on the event clock;
the consumer decides what to do. To treat them the same everywhere, use
`reader.excluded_spans()`, an `ExcludedSpans` with:

- `mask(times_us)`: True where a time (an event or a label) is excluded,
  `labels.time_us[~spans.mask(labels.time_us)]` keeps the trusted rows.
- `kept(start_us, end_us)`: the `[a, b)` pieces of a window outside every
  span, for example to score only trusted time.

`write_excluded(ExcludedSpans(starts, ends))` takes spans in any order and may
be called repeatedly. `ExcludedSpans` raises `ValueError` for `start >= end`
or overlapping spans (touching spans are fine); a rejected call adds nothing.
Spans are stored sorted. No `excluded` group means nothing is excluded.

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

- `1.1` added the `bias` and `excluded` groups and nothing else. This reader
  reads `1.0` files as having no bias log and no excluded spans; a `1.0`
  reader reads `1.1` files and ignores the new groups (the h5py-only example
  below is unchanged and is tested on such a file).

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

Bias in force at `time_us`, h5py only (the `bias` group may be absent, which
means unknown):

<!-- docs-example-bias -->
```python
import h5py

with h5py.File(path, "r") as f:
    names = [name.decode() for name in f["bias/names"][:]]
    rows = f["bias/time_us"][:] <= time_us
    bias = {}
    for name_id, value in zip(f["bias/name_id"][rows], f["bias/value"][rows]):
        bias[names[name_id]] = int(value)
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
- Bias is stored from 1.1 on; a recording without a bias log has unknown
  bias, not default bias. Biases are integers, nothing says which sensor
  setting a name denotes, and no other sensor settings are stored.
- No class-name table and no free-form attributes; ids are agreed per source.
- No frames or RGB data, only the boxes' frame is named.
- Timestamps must fit int64; a single delta must stay below 2**63 us.
