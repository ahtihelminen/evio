# Driver build and link guide

How to build and link the two native camera drivers, IMX636 and U3V, from a
clean Linux install. This guide is the only place the steps are written down;
`tests/test_driver_setup_doc.py` fails when it drifts from the `Makefile`.

The drivers are libusb-based C++ extensions, not PyPI packages. Their sources
live in their own repositories and are not part of evio. evio imports them
lazily, so everything except live cameras works without this guide.

## Prerequisites

Install these first. A version marked "to be confirmed at the walkthrough" is
not determined by any file in this repository and is deliberately not guessed.

| Prerequisite | Version | Why |
| --- | --- | --- |
| git | to be confirmed at the walkthrough | clone the driver repositories |
| Python | 3.12 (`requires-python >=3.12`, `PY ?= 3.12` in the Makefile) | the venv and the driver bindings |
| uv | to be confirmed at the walkthrough | `make install` runs `uv sync` |
| make | to be confirmed at the walkthrough | every step below is a make target |
| C++ compiler | to be confirmed at the walkthrough | builds the driver extensions |
| cmake | to be confirmed at the walkthrough | configures and builds each driver |
| libusb (with headers) | to be confirmed at the walkthrough | both drivers talk to the camera over USB |

Python development headers matching the venv's Python are also needed to
build the pybind11 bindings; their version follows the Python row above.

## Steps

Run these from the root of the evio checkout.

1. Install the environment. This needs no driver at all:

   ```bash
   make install
   ```

2. Check the prerequisites. This tests only that git, python3, uv, make, a
   C++ compiler and cmake are on `PATH` (not their versions, nor libusb or the
   Python headers), names every one that is missing and exits non-zero:

   ```bash
   make drivers-prereqs
   ```

3. Clone each driver as a sibling of evio, as described in the two sections
   below. To use other locations, override `IMX636_DIR` and `U3V_DIR`:

   ```bash
   make drivers IMX636_DIR=/abs/IMX636 U3V_DIR=/abs/U3V
   ```

4. Build and link both drivers:

   ```bash
   make drivers
   ```

   `make drivers` runs `make build-externals` (cmake-builds each driver only
   when its `.so` is missing, so it is safe to re-run) and then links them
   into the venv.

5. Check that both import in the project venv. No camera is opened:

   ```bash
   make drivers-check
   ```

   It names every driver that fails to import and exits non-zero.

After `uv sync` or a pull, run `make sync` to rewrite the link files.

## IMX636

- Repository: `git@git.sensofusion.com:ops/imx636.git`
- Clone directory: `../IMX636` (the `IMX636_DIR` default, relative to the
  evio root)
- Build command: `make build-imx636` (part of `make drivers`)
- Linked output: `imx636.pth` in the venv's `site-packages`, pointing at
  `../IMX636/src/build`, which holds `imx636.cpython-*.so`

```bash
git clone git@git.sensofusion.com:ops/imx636.git ../IMX636
make drivers
```

## U3V

- Repository: `git@git.sensofusion.com:ops/u3v.git`
- Clone directory: `../U3V` (the `U3V_DIR` default, relative to the evio
  root)
- Build command: `make build-u3v` (part of `make drivers`)
- Linked output: `u3v.pth` in the venv's `site-packages`, pointing at
  `../U3V/src/build`, which holds `u3v.cpython-*.so`

```bash
git clone git@git.sensofusion.com:ops/u3v.git ../U3V
make drivers
```

The mono camera's backend is picked by `EVIO_MONO_BACKEND` (`ids_peak` by
default, or `u3v`).
