# Focus Module (`src/evio/focus`)

This document describes the event-based lens-focus tooling under
`src/evio/focus`.

Files covered:

- `src/evio/focus/metric.py`
- `src/evio/focus/sources.py`
- `src/evio/focus/render.py`
- `src/evio/focus/session_log.py`
- `src/evio/focus/__init__.py`

## What this module owns

The focus package owns the focus audit of an event camera:

1. The sharpness metric: the ratio of high-spatial-frequency energy to total
   AC energy in a binned event-count image (`metric.py`).
2. The pane workers that score every rolling window from a live IMX636, a
   `.dat` replay or a synthetic scene (`sources.py`).
3. The grid rendering of the focus meter UI, as pure functions (`render.py`).
4. The JSONL session log and end-of-session summary table (`session_log.py`).

It does not own:

- reading `.dat` files or opening cameras (`src/evio/io/*`),
- the event container and rolling windows (`src/evio/core/*`),
- detection, tracking or classification,
- any lens or iris control: it scores focus and logs it, it does not move
  anything.
