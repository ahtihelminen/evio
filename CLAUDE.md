## Readability

The code should remain as human readable as possible in the development stage.

Before finishing:
- Examine the code you added and make sure it uses clear variable names.
- Remove redundant type checking and error handling.
- Minimize the number of added code lines.
- Try to keep scripts under ./scripts as tidy as possible, preferably just a main() function that gets executed on running the script. Refactor all helper functions and classes under appropriate modules.

## Linting (mandatory)

All Python code MUST pass Ruff without warnings.

Before finishing:
- Run: `uv run --frozen --extra dev ruff check .`
- If auto-fixable issues exist, run: `uv run --frozen --extra dev ruff check . --fix`
- The test half of the gate is `uv run --frozen --extra dev pytest -q`, or
  `make check`, which runs both halves.
- `ruff` and `pytest` both live in the opt-in `dev` extra, so `--extra dev` is
  required — without it `uv run` cannot find either binary on a clean checkout.
  `tests/test_gate_spelling.py` fails if this file, AGENTS.md, PROJECT.md and
  the Makefile ever disagree again.
- Do not consider the task complete if Ruff reports any errors.
- Never disable Ruff rules with `# noqa` unless explicitly requested.
- Follow the project's existing import ordering and formatting style.
- Ruff config is defined in `pyproject.toml`. Follow it strictly.
