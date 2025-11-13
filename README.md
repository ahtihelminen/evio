# evio

Minimal Python library for standardized handling of event camera data.

**evio** provides a single abstraction for event streams. Each source yields standardized event packets containing `x_coords, y_coords, timestamps, polarities` arrays. This makes algorithms and filters source-agnostic.

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
# create venv and install dependencies.
uv sync

# play a .dat file in real time
uv run scripts/play_dat.py path/to/dat/file.dat
```

Adjust window duration in ms using `--window` argument and playback speed factor with `--speed` argument. When event data is constructed to frames we take all events between t and t + window and display them in the frame. With very short windows the rendering of the frames can take longer than the actual window duration and the player falls behind (depends on the playback speed), you can see this by comparing the wall clock to the recording clock in the GUI. In such cases you can force the playback speed with a `--force-speed` argument. This drops enough frames to make the recording play according to the set speed.
---

## License
MIT

