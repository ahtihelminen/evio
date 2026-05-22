# evio dev environment bootstrap.
#
# Goal: clone the four repos as siblings (evio, sapient, openeb, IMX636),
# cd evio, run `make install`. Done.
#
# Why a Makefile + .pth shim instead of declaring openeb/IMX636 as uv
# dependencies: neither is on PyPI; they're locally-built C++ extensions
# whose Python bindings land as bare .so files. uv sync would strip
# them on every invocation. The .pth files re-add the build dirs to
# sys.path so `import metavision_hal` and `import imx636` keep working
# across syncs.
#
# Override locations via env vars if your tree isn't sibling-shaped:
#
#   make install OPENEB_DIR=/abs/openeb IMX636_DIR=/abs/IMX636

PY        ?= 3.12
EVIO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
OPENEB_DIR ?= $(EVIO_ROOT)/../openeb
IMX636_DIR ?= $(EVIO_ROOT)/../IMX636
VENV      := $(EVIO_ROOT)/.venv
SITE_PKGS := $(VENV)/lib/python$(PY)/site-packages

.PHONY: install sync link-locals build-externals build-openeb build-imx636 \
        lint test check clean-venv help

help:
	@echo "Targets:"
	@echo "  install         clone-and-go: build externals, uv sync, link .pth"
	@echo "  sync            uv sync + relink .pth files (use after pulling)"
	@echo "  link-locals     just (re)write the openeb/IMX636 .pth files"
	@echo "  build-externals build openeb and IMX636 if their .so is missing"
	@echo "  lint            uv run ruff check ."
	@echo "  test            uv run pytest -q"
	@echo "  check           lint && test"
	@echo "  clean-venv      blow away .venv (next sync will rebuild)"
	@echo ""
	@echo "Env overrides:"
	@echo "  OPENEB_DIR=$(OPENEB_DIR)"
	@echo "  IMX636_DIR=$(IMX636_DIR)"

install: build-externals sync

sync:
	uv sync
	$(MAKE) link-locals

# Builds openeb / IMX636 only when the Python .so is missing. Idempotent —
# safe to re-run; cmake handles incremental rebuilds.
build-externals: build-openeb build-imx636

build-openeb:
	@if [ ! -f "$(OPENEB_DIR)/build/py3/metavision_hal.cpython-$(subst .,,$(PY))-x86_64-linux-gnu.so" ]; then \
		echo ">>> building openeb at $(OPENEB_DIR)"; \
		test -d "$(OPENEB_DIR)" || { \
			echo "openeb missing at $(OPENEB_DIR); clone https://github.com/prophesee-ai/openeb" >&2; \
			exit 1; \
		}; \
		cd "$(OPENEB_DIR)" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
		    -DCOMPILE_PYTHON3_BINDINGS=ON && cmake --build build -j; \
	else \
		echo ">>> openeb already built"; \
	fi

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

# .pth files survive `uv sync` only as long as we re-write them after.
# Hooked into `make sync` so a `git pull && make sync` is the standard
# update loop.
link-locals:
	@test -d "$(SITE_PKGS)" || { echo "no venv at $(VENV); run 'make sync' first" >&2; exit 1; }
	@if [ -d "$(OPENEB_DIR)/build/py3" ]; then \
		echo "$(OPENEB_DIR)/build/py3" > "$(SITE_PKGS)/openeb.pth"; \
		echo "linked openeb → $(SITE_PKGS)/openeb.pth"; \
	else \
		echo "skip openeb.pth (no $(OPENEB_DIR)/build/py3)"; \
	fi
	@if [ -d "$(IMX636_DIR)/src/build" ]; then \
		echo "$(IMX636_DIR)/src/build" > "$(SITE_PKGS)/imx636.pth"; \
		echo "linked imx636 → $(SITE_PKGS)/imx636.pth"; \
	else \
		echo "skip imx636.pth (no $(IMX636_DIR)/src/build)"; \
	fi

lint:
	uv run ruff check .

test:
	uv run pytest -q

check: lint test

clean-venv:
	rm -rf "$(VENV)"
