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
- Run: `uv run --extra dev ruff check .`
- If auto-fixable issues exist, run: `uv run --extra dev ruff check . --fix`
- `ruff` lives in the opt-in `dev` extra, so `--extra dev` is
  required — a bare `uv run ruff` cannot find the binary on a clean checkout.
- Do not consider the task complete if Ruff reports any errors.
- Never disable Ruff rules with `# noqa` unless explicitly requested.
- Follow the project's existing import ordering and formatting style.
- Ruff config is defined in `pyproject.toml`. Follow it strictly.
