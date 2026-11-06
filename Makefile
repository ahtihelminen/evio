# evio dev environment bootstrap.
#
# `make install` gives a working environment with no camera repos at all:
# evio imports the native drivers lazily. To talk to real cameras, clone
# IMX636 and U3V as siblings of this repo and run `make drivers`.
#
# Why a Makefile + .pth shim instead of declaring the drivers as uv
# dependencies: they aren't on PyPI; they're locally-built C++ extensions
# whose Python bindings land as bare .so files. uv sync would strip them on
# every invocation. The .pth files re-add the build dirs to sys.path so
# `import imx636` / `import u3v` keep working across syncs.
#
# Override the locations via env vars if your tree isn't sibling-shaped:
#
#   make drivers IMX636_DIR=/abs/IMX636 U3V_DIR=/abs/U3V

PY        ?= 3.12
GATE      := uv run --frozen --extra dev
EVIO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPEAT    ?= 3
SEED_A    ?= 1
SEED_B    ?= 2
IMX636_DIR ?= $(EVIO_ROOT)/../IMX636
U3V_DIR   ?= $(EVIO_ROOT)/../U3V
DRIVERS   ?= imx636 u3v
VENV     := $(EVIO_ROOT)/.venv
SITE_PKGS := $(VENV)/lib/python$(PY)/site-packages

.PHONY: install sync drivers drivers-check link-locals build-externals build-imx636 \
        build-u3v lint fast test test-repeat test-random-order test-hardware clean-tree check clean-venv help

help:
	@echo "Targets:"
	@echo "  install         uv sync + link any already-built drivers"
	@echo "  sync            uv sync + relink the .pth files (use after pulling)"
	@echo "  drivers         build IMX636 and U3V if missing, then link them"
	@echo "  drivers-check   import imx636 and u3v in the venv, name any that fail"
	@echo "  link-locals     just (re)write the IMX636/U3V .pth files"
	@echo "  lint            $(GATE) ruff check ."
	@echo "  fast            lint + the t1 unit tier (the every-edit loop)"
	@echo "  test            $(GATE) pytest -q -m \"not hardware\""
	@echo "  test-repeat     run the laptop suite $(REPEAT) times, fail if outcomes differ"
	@echo "  test-random-order  laptop suite in random order, seeds $(SEED_A) and $(SEED_B)"
	@echo "  test-hardware   $(GATE) pytest -q -m hardware (skips without a camera)"
	@echo "  clean-tree      test, then fail if git status --porcelain prints anything"
	@echo "  check           lint && clean-tree"
	@echo "  clean-venv      blow away .venv (next sync will rebuild)"
	@echo ""
	@echo "Env overrides:"
	@echo "  IMX636_DIR=$(IMX636_DIR)"
	@echo "  U3V_DIR=$(U3V_DIR)"

install: sync

sync:
	uv sync --extra dev
	$(MAKE) link-locals

drivers: build-externals link-locals

# Builds IMX636 and U3V only when their Python .so is missing. Idempotent —
# safe to re-run; cmake handles incremental rebuilds. pybind11 names the .so
# after the build host (e.g. imx636.cpython-312-darwin.so), so glob for it.
build-externals: build-imx636 build-u3v

build-imx636:
	@if ! ls "$(IMX636_DIR)/src/build"/imx636.cpython-*.so >/dev/null 2>&1; then \
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

# Smoke check: imports each driver inside the project venv. No camera is
# opened. Every failing driver is named on stderr, then the target fails.
define DRIVERS_CHECK_PY
import importlib, sys
bad = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(name)
        print(f"driver {name} failed to import: {exc!r}", file=sys.stderr)
sys.exit(1 if bad else 0)
endef
export DRIVERS_CHECK_PY

drivers-check:
	@$(GATE) python -c "$$DRIVERS_CHECK_PY" $(DRIVERS)
	@echo "drivers import: $(DRIVERS)"

# --frozen keeps uv.lock untouched and --extra dev is where ruff and pytest
# actually live; tests/test_gate_spelling.py fails if any doc drifts from this
# spelling.
lint:
	$(GATE) ruff check .

# The inner loop: the pure tier, no process, socket, sleep, clock or browser.
fast: lint
	$(GATE) pytest -q -m t1 --tier-manifest=t1

test:
	$(GATE) pytest -q -m "not hardware" --tier-manifest=t1,t2 \
		--coverage-manifest

# Runs the laptop suite REPEAT times and fails if any run's outcome set (every
# PASSED/FAILED/SKIPPED/... line with its test id) differs from the first run's.
test-repeat:
	@dir="$$(mktemp -d)"; trap 'rm -rf "$$dir"' EXIT; \
	for i in $$(seq 1 $(REPEAT)); do \
		echo ">>> run $$i of $(REPEAT)"; \
		$(GATE) pytest -q -m "not hardware" -rA -p no:cacheprovider \
			> "$$dir/out.$$i" || true; \
		grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ' "$$dir/out.$$i" \
			| sort > "$$dir/outcomes.$$i"; \
		test -s "$$dir/outcomes.$$i" || { \
			echo "run $$i produced no outcomes" >&2; tail -n 20 "$$dir/out.$$i" >&2; exit 1; }; \
		diff -u "$$dir/outcomes.1" "$$dir/outcomes.$$i" || { \
			echo "run $$i differs from run 1" >&2; exit 1; }; \
	done; \
	echo "$(REPEAT) runs, identical outcomes"

# Runs the laptop suite in random order once per fixed seed, echoing the seed
# before each run; a failure reproduces with `pytest --random-order-seed=<seed>`.
test-random-order:
	@echo ">>> random order, seed $(SEED_A)"
	$(GATE) pytest -q -m "not hardware" --random-order-seed=$(SEED_A)
	@echo ">>> random order, seed $(SEED_B)"
	$(GATE) pytest -q -m "not hardware" --random-order-seed=$(SEED_B)

# Runs only the hardware-marked subset. Without a camera every test skips,
# naming the missing driver or device (tests/hardware/conftest.py).
test-hardware:
	$(GATE) pytest -q -m hardware --tier-manifest=hardware

# The laptop suite must leave the repository as it found it. Run it from a
# committed tree: any uncommitted edit is reported as a leftover too.
clean-tree: test
	@leftovers="$$(git status --porcelain)"; \
	if [ -n "$$leftovers" ]; then \
		echo "working tree not clean after the laptop run:" >&2; \
		echo "$$leftovers" >&2; \
		exit 1; \
	fi

check: lint clean-tree

clean-venv:
	rm -rf "$(VENV)"
