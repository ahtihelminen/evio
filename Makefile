# evio dev environment bootstrap.
#
# Goal: clone the three repos as siblings (evio, sapient, IMX636),
# cd evio, run `make install`. Done.
#
# Why a Makefile + .pth shim instead of declaring IMX636 as a uv
# dependency: it isn't on PyPI; it's a locally-built C++ extension whose
# Python bindings land as a bare .so file. uv sync would strip it on
# every invocation. The .pth file re-adds the build dir to sys.path so
# `import imx636` keeps working across syncs.
#
# Override the location via an env var if your tree isn't sibling-shaped:
#
#   make install IMX636_DIR=/abs/IMX636

PY        ?= 3.12
EVIO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
IMX636_DIR ?= $(EVIO_ROOT)/../IMX636
VENV      := $(EVIO_ROOT)/.venv
SITE_PKGS := $(VENV)/lib/python$(PY)/site-packages

.PHONY: install sync link-locals build-externals build-imx636 \
        lint test test-hardware check clean-venv help

help:
	@echo "Targets:"
	@echo "  install         clone-and-go: build externals, uv sync, link .pth"
	@echo "  sync            uv sync + relink the .pth file (use after pulling)"
	@echo "  link-locals     just (re)write the IMX636 .pth file"
	@echo "  build-externals build IMX636 if its .so is missing"
	@echo "  lint            uv run ruff check ."
	@echo "  test            uv run pytest -q -m \"not hardware\""
	@echo "  test-hardware   uv run pytest -q -m hardware (real OPS-unit camera required)"
	@echo "  check           lint && test"
	@echo "  clean-venv      blow away .venv (next sync will rebuild)"
	@echo ""
	@echo "Env overrides:"
	@echo "  IMX636_DIR=$(IMX636_DIR)"

install: build-externals sync

sync:
	uv sync
	$(MAKE) link-locals

# Builds IMX636 only when the Python .so is missing. Idempotent — safe to
# re-run; cmake handles incremental rebuilds.
build-externals: build-imx636

build-imx636:
	@if [ ! -f "$(IMX636_DIR)/src/build/imx636.cpython-$(subst .,,$(PY))-x86_64-linux-gnu.so" ]; then \
		echo ">>> building IMX636 at $(IMX636_DIR)"; \
		test -d "$(IMX636_DIR)/src" || { \
			echo "IMX636 missing at $(IMX636_DIR); clone git@git.sensofusion.com:ops/imx636.git" >&2; \
			exit 1; \
		}; \
		cd "$(IMX636_DIR)/src" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
		    cmake --build build -j; \
	else \
		echo ">>> IMX636 already built"; \
	fi

# The .pth file survives `uv sync` only as long as we re-write it after.
# Hooked into `make sync` so a `git pull && make sync` is the standard
# update loop.
link-locals:
	@test -d "$(SITE_PKGS)" || { echo "no venv at $(VENV); run 'make sync' first" >&2; exit 1; }
	@if [ -d "$(IMX636_DIR)/src/build" ]; then \
		echo "$(IMX636_DIR)/src/build" > "$(SITE_PKGS)/imx636.pth"; \
		echo "linked imx636 → $(SITE_PKGS)/imx636.pth"; \
	else \
		echo "skip imx636.pth (no $(IMX636_DIR)/src/build)"; \
	fi

lint:
	uv run ruff check .

test:
	uv run pytest -q -m "not hardware"

# Runs only the hardware-marked subset, against a real OPS-unit camera.
test-hardware:
	uv run pytest -q -m hardware

check: lint test

clean-venv:
	rm -rf "$(VENV)"
