# Codec Module (`src/evio/codec`)

This document describes the codec layer that turns raw file words into
`EventsBatch` objects (and, for DT8, back to words).

Files covered:

- `src/evio/codec/codec_protocol.py`
- `src/evio/codec/dt8.py`
- `src/evio/codec/evt3.py`
- `src/evio/codec/__init__.py`

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
- All decoders consume streams as little-endian 16-bit words.

### `EventCodec` protocol

Any codec must implement:

- `reset() -> None`
- `decode_chunk(words, final=False) -> Iterable[EventsBatch]`

Notes:

- `decode_chunk` can return zero, one, or many batches.
- Decoders are expected to be stateful across chunk calls.
- `final` exists for flush behavior, but current concrete codecs do not use it.

## DT8 codec (`dt8.py`)

`dt8.py` handles Prophesee DAT files with 8-byte CD events.

### Header offset helper

`find_dat_data_offset(path)` scans DAT header bytes:

1. Skip ASCII lines beginning with `%`.
2. Read two marker bytes (`event_type`, `decoded_size`).
3. Require `decoded_size == 8`.
4. Return byte offset of first payload word.

This is used by DAT readers and converters before decoding.

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
- Applies coordinate validity checks against `width/height`.

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

- `encode_chunk(batch)` exists for conversion/writing workflows.
- Produces `uint16` stream compatible with DAT payload layout.

## EVT3 codec (`evt3.py`)

`evt3.py` handles EVT3 RAW payload decoding.

### Header offset helper

`find_data_offset(path)` skips ASCII header lines starting with `%` and returns
payload start offset. It also supports `% end` termination.

### EVT3 decode model

EVT3 is decoded as a stream of 16-bit words with high-nibble tags.

`decode_evt3_chunk(...)` is a numba-compiled function that:

- Parses event tags (`Y`, `X`, `BASE_X`, `VECT_12`, `VECT_8`,
  `TIME_HIGH`, `TIME_LOW`).
- Reconstructs event timestamps from high/low time registers.
- Tracks wraparound in the high time base for long recordings.
- Emits arrays `(timestamps, x, y, polarity)` plus updated decoder registers.

Important behavior:

- Decoding does not emit events until an initial `TIME_HIGH` is seen.
- Vector events (`VECT_12`, `VECT_8`) expand bitmasks into multiple events.
- Time state is carried across chunk boundaries via returned registers.

### `Evt3Codec` class

`Evt3Codec` wraps `decode_evt3_chunk` and stores persistent registers:

- position/polarity registers (`current_y`, `current_base_x`,
  `current_polarity`)
- timestamp registers (`current_time_base_us`, `current_time_low_us`,
  `current_time_us`)
- rollover tracking (`time_high_loops`, plus constants)
- initialization flag (`first_time_base_set`)

`decode_chunk`:

- Returns `()` when chunk has no decodable CD events.
- Otherwise returns `(EventsBatch(...),)`.

`reset` clears all registers to initial state.

## Integration points in the repo

### Multiprocessing readers

- `src/evio/io/dat_file_mp.py` wires `Dt8Codec` with DAT header offsets.
- `src/evio/io/raw_file_mp.py` wires `Evt3Codec` with RAW header offsets.
- `src/evio/io/mp_file.py` provides the generic reader/worker process model
  that repeatedly calls `codec.decode_chunk(...)`.

### Conversion

`src/evio/io/converter.py` uses codecs for stream conversion:

- input DAT: `Dt8Codec` decode path
- input RAW: `Evt3Codec` decode path
- output DAT: `Dt8Codec.encode_chunk(...)`

### UI playback helpers

`src/evio/ui/sources/dat_playback.py` uses DAT offset parsing and DT8 codec
helpers when building playback/export operations.

## Performance and design notes

- Hot decode loops (`decode_dt8_pairs`, `decode_evt3_chunk`) are `@njit`.
- The protocol keeps codecs chunk-based so multiprocessing workers can decode
  incrementally with bounded memory.
- EVT3 is inherently stateful; preserving register state across chunks is
  mandatory for correct decode.
- DT8 is simpler but still uses chunk buffering to survive odd-alignment word
  boundaries.

## Current limitations and assumptions

- Codec input type is fixed to `uint16` words for both formats.
- `decode_chunk(..., final=True)` is not currently used for extra flush logic.
- DT8 timestamps are 32-bit values promoted to `int64`; no extra unwrapping is
  applied in the codec itself.
- `src/evio/codec/__init__.py` is currently empty and does not re-export codec
  symbols.

