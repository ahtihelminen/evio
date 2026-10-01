# Codec Module (`src/evio/codec`)

This document describes the codec layer that turns raw file words into
`EventsBatch` objects (and, for DT8, back to words).

Files covered:

- `src/evio/codec/codec_protocol.py`
- `src/evio/codec/dt8.py`
- `src/evio/codec/__init__.py`

DT8 (Prophesee DAT, 8-byte CD events) is the only codec in this repo.

## Responsibilities

The codec layer is responsible for:

1. Defining a common streaming decoder interface.
2. Parsing codec-specific binary payload words.
3. Keeping decoder state across chunk boundaries.
4. Returning decoded events as `EventsBatch`.

It does not own rolling windows or playback pacing; those live in IO/window
modules.

## Common interface (`codec_protocol.py`)

### `CodecInput`

- Alias: `NDArray[np.uint16]`
- Decoders consume streams as little-endian 16-bit words.

### `EventCodec` protocol

Any codec must implement:

- `reset() -> None`
- `decode_chunk(words, *, final=False) -> Iterable[EventsBatch]`

Notes:

- `decode_chunk` can return zero, one, or many batches.
- Decoders are expected to be stateful across chunk calls.
- `final` exists for flush behavior, but `Dt8Codec` does not use it.

## DT8 codec (`dt8.py`)

`dt8.py` handles Prophesee DAT files with 8-byte CD events.

### Header offset helper

`find_dat_data_offset(path)` scans DAT header bytes:

1. Skip ASCII lines beginning with `%`.
2. Read two marker bytes (`event_type`, `decoded_size`).
3. Require `decoded_size == 8`.
4. Return byte offset of first payload word.

This is used by DAT readers before decoding.

### DT8 word layout

Each event is two `uint32` values:

- `t32`: timestamp low 32 bits
- `w32`: packed event word

Packed event field in `w32`:

- bits `0..13`: `x`
- bits `14..27`: `y`
- bit `28`: polarity

### Numba primitives

`decode_dt8_pairs(timestamps, words, width, height)`:

- Unpacks `uint32` arrays into `(timestamps, x, y, polarity)`.
- Drops events whose coordinates fall outside `width/height`.

`encode_dt8_pairs(timestamps, x, y, polarity, width, height)`:

- Inverse packing path for DT8 writing.
- Drops events with out-of-bounds coordinates.
- Writes low 32-bit timestamp values.

### `Dt8Codec` class

`Dt8Codec(width, height)` is `EventCodec`-compatible and stateful.

State behavior:

- Input is `uint16` words.
- If a chunk ends on an odd number of words, one trailing word is buffered in
  `_buf` and prepended to the next chunk.
- After alignment, words are viewed as `uint32` and split into `(t32, w32)`
  pairs.

Decode output:

- Returns `()` for empty/no-event chunks.
- Otherwise returns a single-element tuple `(EventsBatch,)`.

Encode path:

- `encode_chunk(batch)` packs an `EventsBatch` back into a `uint16` stream
  compatible with the DAT payload layout.

## Integration points in the repo

- `src/evio/io/dat_file_mp.py` wires `Dt8Codec` with DAT header offsets.
- `src/evio/io/mp_file.py` provides the generic reader/worker process model
  that repeatedly calls `codec.decode_chunk(...)`.
- `src/evio/recording/footage_recorder.py` uses `Dt8Codec.encode_chunk(...)`
  to write recorded windows as DAT payload.
- `tests/test_dt8_codec_roundtrip.py` pins encode/decode as inverses.

## Performance and design notes

- Hot loops (`decode_dt8_pairs`, `encode_dt8_pairs`) are `@njit`.
- The protocol keeps codecs chunk-based so multiprocessing workers can decode
  incrementally with bounded memory.
- DT8 uses chunk buffering to survive odd-alignment word boundaries.

## Current limitations and assumptions

- Codec input type is fixed to `uint16` words.
- `decode_chunk(..., final=True)` is not currently used for extra flush logic.
- DT8 timestamps are 32-bit values promoted to `int64`; no unwrapping is
  applied in the codec itself (`core.recording.open_dat_dt8` unwraps for the
  whole-file loader).
- `src/evio/codec/__init__.py` is empty and does not re-export codec symbols.
