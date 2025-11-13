# evio

Minimal Python library for standardized handling of event camera data.

**evio** provides a single abstraction for event streams. Each source yields standardized event packets containing `(x, y, t, p)` arrays. This makes algorithms and filters source-agnostic.

---

## Features
- Unified async interface for event streams
- Read `.dat` recordings with optional real-time pacing
- Extensible to live cameras via adapter classes (requires Metavision SDK)

---

## Repository Structure

```
.
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ scripts/
│  └─ play_dat.py    
└─ src/
   └─ evio/
      ├─ __init__.py
      ├── core/
      │   ├── __init__.py
      │   ├── index_scheduler.py
      │   ├── mmap.py
      │   ├── pacer.py
      │   └── recording.py
      └─── source/
          ├── __init__.py
          └── dat_file.py
       
```

---

## Quick start using UV
Clone the repo and in the repo root run

```bash
# install in editable mode (using uv or pip)
uv sync

# play back a .dat file in real time
uv run scripts/play_dat.py path/to/dat/file.dat
```

---

## Istalling openeb on macOS


## License
MIT

