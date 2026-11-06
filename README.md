# evio

The base event-camera library: everything that works on a laptop (Linux or
macOS) with or without an event camera plugged in. Read and write event
data, open the cameras, project pixels to the sky, render frames, and run
the detection chain.

Two repos build on it, and the dependency only points one way:

- **ops** (`git@git.sensofusion.com:ops/ops.git`, package `evio_ops`) — the
  OPS-unit appliance: the camera ring's worker processes, the bias
  autotuner, SAPIENT egress, the web UI and the unit's boot lifecycle. It
  pins an evio release tag.
- **evior** (`git@git.sensofusion.com:ops/evior.git`) — model and method
  development tools on top of evio.

Nothing here imports either of them; `tests/test_repo_boundary.py` enforces
it.

## What this repo contains

- `codec/` — the DT8 event codec.
- `core/` — `EventsBatch`, rolling windows, pacing, `.dat` recording
  helpers, ids.
- `io/` — `.dat` readers, the IMX636 camera source, the mono-camera facade
  with its U3V and (deprecated) IDS Peak backends.
- `streaming/` — the IMX636 stuck-pixel scan and the bias register
  constants and types.
- `detection/` — pipelines, Kalman tracker, re-id, LLR arms, the
  temporal-transformer classifier (torch and OpenVINO IR) and the verdict
  gate.
- `geometry/` — intrinsics/extrinsics, az/el projection, FOV, geodesy and
  the rig/site config schema.
- `dtc/` — the detect-track-classify pipeline contract, chain and
  conformance kit.
- `rendering/` — event-to-frame rendering and overlays.
- `recording/` — hour-bucketed footage capture and crash recovery, and the
  labelled recording format (one HDF5 file of events, boxes and metadata).
- `focus/` — the event-based lens-focus metric.
- `time/` — the time-source abstraction.

`docs/modules/` has per-package notes: [`focus`](docs/modules/focus.md),
[`geometry`](docs/modules/geometry.md),
[`recording`](docs/modules/recording.md),
[`streaming`](docs/modules/streaming.md) and
[`time`](docs/modules/time.md), next to the existing ones.
`docs/modules/labelled-recording.md` specifies the labelled recording file
format.

## Setup

```bash
make install     # uv sync (with the dev extra) + link any built drivers
make check       # ruff + the t1/t2 test tiers + clean tree (make clean-tree)
```

That is enough for everything except live cameras. To open real cameras,
clone the native drivers as siblings and build them:

```bash
git clone git@git.sensofusion.com:ops/imx636.git ../IMX636
git clone git@git.sensofusion.com:ops/u3v.git ../U3V
make drivers     # cmake-build both if missing, link them into .venv
```

The drivers are libusb-based C++ extensions, not PyPI packages; `make
drivers` writes `.pth` files into the venv so `import imx636` and `import
u3v` survive `uv sync`. The mono camera's backend is picked by
`EVIO_MONO_BACKEND` (`ids_peak` by default, or `u3v`). IDS Peak is deprecated: U3V replaces it completely.

To run the hardware tests against attached cameras, see "Hardware tests" in
[`docs/modules/io.md`](docs/modules/io.md#hardware-tests).

## Optional extras

- `evio[openvino]` — OpenVINO IR inference for the transformer classifier.
- `evio[torch]` — torch checkpoints and the IR export tooling
  (`scripts/convert_checkpoint_to_ir.py`).
- `evio[ids-peak]` — the IDS Peak mono-camera SDK (Linux only; deprecated, U3V replaces it).

The gate is `uv run --frozen --extra dev ruff check .` plus
`uv run --frozen --extra dev pytest -q`, or `make check` for both.

## No network in the laptop suite

`make test` (and so `make check`) runs with sockets blocked: the autouse
`_no_sockets` fixture in `tests/conftest.py` makes any test that opens a
socket fail with `SocketBlocked`. `tests/test_socket_guard.py` proves it with
a strict expected-failure test. A `t2` test that needs a local socket opts out
with `pytest.mark.sockets`; nothing else may.

## History and coverage

This repo's history was filtered out of ops (the former evioF) at the
split: it keeps every commit that touched a file still here, under its
current path. Commits made to a file under an older name, and the per-file
coverage ratchet the monolith kept, stayed behind in ops (`f3c0845` on its
`dev` is the last pre-split commit). Here the floor is per top-level package
under `src/evio`, in `tests/fixtures/coverage_manifest.json`: `make test` and
`make check` fail when a package's line coverage drops below its floor.

## Releases

ops installs evio from a git tag (`vX.Y.Z`), never from a branch. To ship a
change to ops: merge it here, tag the merge commit, push the tag, then bump
the tag in ops's `pyproject.toml` and `uv lock` there. A pushed tag is never
moved; a fix gets the next patch version.
