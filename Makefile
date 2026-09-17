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
GATE      := uv run --frozen --extra dev
CASE      ?= simplified
COVJSON   := artifacts/coverage.json
EVIO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
IMX636_DIR ?= $(EVIO_ROOT)/../IMX636
U3V_DIR   ?= $(EVIO_ROOT)/../U3V
VENV      := $(EVIO_ROOT)/.venv
SITE_PKGS := $(VENV)/lib/python$(PY)/site-packages

.PHONY: install sync link-locals build-externals build-imx636 build-u3v \
        lint fast test test-hardware bench coverage check clean-venv help

help:
	@echo "Targets:"
	@echo "  install         clone-and-go: build externals, uv sync, link .pth"
	@echo "  sync            uv sync + relink the .pth file (use after pulling)"
	@echo "  link-locals     just (re)write the IMX636/U3V .pth files"
	@echo "  build-externals build IMX636 and U3V if their .so is missing"
	@echo "  lint            $(GATE) ruff check ."
	@echo "  fast            lint + the t1 unit tier (the every-edit loop)"
	@echo "  bench           measure a replay case and diff it against the baseline"
	@echo "  coverage        measure coverage and check the per-file ratchet"
	@echo "  test            $(GATE) pytest -q -m \"not hardware\""
	@echo "  test-hardware   $(GATE) pytest -q -m hardware (real OPS-unit camera required)"
	@echo "  check           lint && test"
	@echo "  clean-venv      blow away .venv (next sync will rebuild)"
	@echo ""
	@echo "Env overrides:"
	@echo "  IMX636_DIR=$(IMX636_DIR)"
	@echo "  U3V_DIR=$(U3V_DIR)"

install: build-externals sync

sync:
	uv sync
	$(MAKE) link-locals

# Builds IMX636 and U3V only when their Python .so is missing. Idempotent —
# safe to re-run; cmake handles incremental rebuilds.
build-externals: build-imx636 build-u3v

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

# Builds U3V's pybind11 extension only when its .so is missing. Unlike
# build-imx636, the .so name isn't hardcoded to one platform tag (pybind11
# names it e.g. u3v.cpython-312-darwin.so or ...-x86_64-linux-gnu.so
# depending on the build host) — glob for any u3v.cpython-*.so instead.
build-u3v:
	@if ! ls "$(U3V_DIR)/src/build"/u3v.cpython-*.so >/dev/null 2>&1; then \
		echo ">>> building U3V at $(U3V_DIR)"; \
		test -d "$(U3V_DIR)/src" || { \
			echo "U3V missing at $(U3V_DIR); clone git@git.sensofusion.com:ops/u3v.git" >&2; \
			exit 1; \
		}; \
		cd "$(U3V_DIR)/src" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
		    cmake --build build -j; \
	else \
		echo ">>> U3V already built"; \
	fi

# The .pth files survive `uv sync` only as long as we re-write them after.
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
	@if [ -d "$(U3V_DIR)/src/build" ]; then \
		echo "$(U3V_DIR)/src/build" > "$(SITE_PKGS)/u3v.pth"; \
		echo "linked u3v → $(SITE_PKGS)/u3v.pth"; \
	else \
		echo "skip u3v.pth (no $(U3V_DIR)/src/build)"; \
	fi

# --frozen keeps uv.lock untouched and --extra dev is where ruff and pytest
# actually live; tests/test_gate_spelling.py fails if any doc drifts from this
# spelling. Deliberately NOT applied to sync/install: the 2026-09-10 Decisions
# entry rejected --frozen there, where it would drag the 9.5 GB development
# environment onto a field unit.
lint:
	$(GATE) ruff check .

# The inner loop: the pure tier, no process, socket, sleep, clock or browser.
fast: lint
	$(GATE) pytest -q -m t1 --tier-manifest=t1

test:
	$(GATE) pytest -q -m "not hardware" --tier-manifest=t1,t2

# Runs only the hardware-marked subset, against a real OPS-unit camera.
# pytest exits 5 when nothing carries the marker; that is "nothing to run",
# not a failure, and it is reported rather than hidden.
test-hardware:
	@$(GATE) pytest -q -m hardware; rc=$$?; \
	if [ $$rc -eq 5 ]; then echo ">>> no hardware-marked tests collected"; exit 0; fi; \
	exit $$rc

# The scoreboard: "for identical output, is this cheaper". Writes an
# artifact and compares it against the committed baseline; exit 2 is VOID,
# which is not a pass -- it means measure again on a quiet host.
bench:
	$(GATE) python scripts/bench_replay.py --case $(CASE)
	$(GATE) python scripts/bench_diff.py \
	    tests/fixtures/bench/$(CASE).baseline.json \
	    artifacts/bench/$(CASE).json

# Coverage as a ratchet: no file may end with more uncovered statements
# than the committed record. On demand only -- it costs 18 % on the unit
# tier, whose whole point is that it takes seconds. pytest-cov is pulled in
# for the invocation rather than added to the lockfile.
coverage:
	$(GATE) --with pytest-cov pytest -q -m "not hardware" \
	    --cov=src/evio --cov-report=json:$(COVJSON) >/dev/null
	$(GATE) python scripts/coverage_ratchet.py $(COVJSON) $(if $(WRITE),--write,)

check: lint test

clean-venv:
	rm -rf "$(VENV)"
