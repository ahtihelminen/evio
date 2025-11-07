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
│     ├── core/
│     │   ├── __init__.py
│     │   ├── recording.py
│     │   ├── policy.py
│     │   ├── index_scheduler.py
│     │   ├── pacer.py
│     │   └── render.py
│     ├── source/
│     │   ├── __init__.py
│     │   ├── dat_file.py
│     │   └── live_stub.py
│     └── transforms/
│         ├── __init__.py
│         └── basic.py
├─ scripts/
│  ├─ play_dat.py           
│  └─ convert_raw_to_dat.py
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

