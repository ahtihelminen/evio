# evio

Minimal Python library for standardized handling of event camera data.

**evio** provides a single abstraction for event streams, whether coming from a `.dat` file or a live device. Each source yields standardized event packets containing `(x, y, t, p)` arrays. This makes algorithms and filters source-agnostic.

---

## Features
- Unified async interface for event streams
- Read `.dat` recordings with optional real-time pacing
- Extensible to live cameras via adapter classes (e.g., libcaer, Metavision)
- No dependencies beyond `numpy` and `asyncio`
- Optional blocking wrapper for synchronous scripts

---

## Repository Structure

```
.
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ src/
│  └─ evio/
│     ├─ __init__.py
│     ├─ core/
│     │  ├─ __init__.py
│     │  ├─ types.py        # EventArray/EventPacket Protocols
│     │  ├─ packet.py       # NumPy-backed Packet
│     │  ├─ clock.py        # pacing utilities
│     │  └─ mmap.py         # minimal .dat memmap reader
│     ├─ sources/
│     │  ├─ __init__.py
│     │  ├─ base.py         # Source protocol
│     │  ├─ file_dat.py     # FileSource (with pacing)
│     │  ├─ usb_base.py     # USBSource protocol
│     │  └─ synthetic.py    # SyntheticSource for tests
│     ├─ io/
│     │  ├─ __init__.py
│     │  └─ dat_spec.py     # .dat header parser
│     └─ util/
│        ├─ __init__.py
│        └─ blocking.py     # run_async_iter_sync()
├─ scripts/
│  ├─ play_dat.py           # playback example
│  ├─ dump_stats.py         # print event rate stats
│  ├─ to_npz.py             # export .dat to npz
│  └─ live_stub.py          # template for live adapters
└─ tests/
   ├─ test_file_source.py
   ├─ test_timing.py
   └─ data/
```

---

## Quick start

```bash
# install in editable mode (using uv or pip)
uv pip install -e .

# play back a .dat file in real time
uv run python scripts/play_dat.py recordings/demo.dat
```

---

## License
MIT

