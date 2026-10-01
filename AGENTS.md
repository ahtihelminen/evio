## Repo boundary (evio)

This is the base library. `ops` (the OPS-unit appliance, package `evio_ops`)
and `evior` (research tooling) depend on it, never the reverse.

- Nothing here may import or name `evio_ops` or `evior`.
  `tests/test_repo_boundary.py` enforces it.
- A module belongs here if it works on a laptop with nothing but evio and,
  optionally, a camera plugged in. Anything that only makes sense on an OPS
  unit — worker orchestration, the web UI, SAPIENT, boot staging — belongs
  in ops.
- Promoting a detection method from evior: move the module into
  `src/evio/detection/`, register it in `detection/pipeline/registry.py`,
  delete it from evior, tag a release, and bump the tag in ops.

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
- Do not consider the task complete if Ruff reports any errors.
- Never disable Ruff rules with `# noqa` unless explicitly requested.
- Follow the project's existing import ordering and formatting style.
- Ruff config is defined in `pyproject.toml`. Follow it strictly.

## Git

Feature branches off `dev`, the integration branch, merged into `dev` via a GitLab merge
request with SN-IDs in the MR title. A human promotes `dev` to `main` on their own
schedule; `main` stays the default branch and receives no direct pushes.

Before creating a branch:
- Format: `<type>/<short-slug>` — `feat/`, `fix/`, `refactor/`, `chore/`, `docs/`, `test/`.
- Slug: 2-5 hyphenated words, e.g. `feat/evio-detection-emitter`.
- Do not use Linear's auto-generated `feature/sn-NN-<long-title>` names.
- Branch from `dev`, not from `main` and not from another feature branch.
- Multi-issue branches are fine — list all SN-IDs in the MR title.

Before committing:
- Conventional prefix: `feat:`, `fix:`, `refactor:`, `chore:`, `test:`, `docs:`.
- Subject: lowercase after the colon, ~5-10 words.
- Body: optional, at most a couple of lines. Long rationale belongs in Linear.
- No SN-IDs in subjects or bodies — they live on the branch name and the MR title.
- No `Closes: SN-X` / `Fixes: SN-X` trailers (GitHub convention, unused here).
- Aim for 2-5 cohesive commits per branch.
- `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer is fine when Claude authored the changes.

Before opening a merge request:
- Run `make lint && make test` (or repo equivalents). Must pass.
- Gate with `uv run --frozen --extra dev`. A plain `uv run` re-resolves `uv.lock`
  for the host platform and strips this repo's pinned Linux/xpu package set.
- Target `dev`, never `main`.
- MR title: imperative, with SN-IDs in parentheses — multiple issues: `(SN-XXX, SN-YYY)`.
- Delete the local feature branch once the MR has merged: `git branch -d <branch>`.

Before pushing:
- Wait for an explicit "push" or "go ahead" from the user. Show `git log --oneline -10` and stop.
- Push the feature branch only: `git push -u origin <branch>`. No `--force`,
  `--no-verify`, or `--no-gpg-sign` unless asked.
- Never push to `main`. Never push to `dev` directly — `dev` takes work through
  merge requests only.

Releasing (ops pins evio by tag):
- Tag `vX.Y.Z` on a commit of `dev` once it is green, and push the tag.
- Never move or delete a pushed tag; a fix is the next patch version.

## Linear

Scope: OPS project, team `team-samuli`, prefix `SN-`. Never touch issues outside this scope.

Before changing issue state:
- Work flows Todo → In Progress → In Review → Done. `team-samuli` does have an
  `In Review` state.
- Start of work: Todo → In Progress (when the feature branch is first pushed).
- MR opened against `dev`: In Progress → In Review.
- After the MR merges into `dev`: In Review → Done.
- Use `mcp__claude_ai_Linear__save_issue` with `state: "In Progress"`, `"In Review"` or `"Done"`.

Before writing or editing a description:
- One paragraph or a short bullet list. No "Steps to Reproduce" / "Expected/Actual" headings.
- Sub-issues of an epic open with `Stage N of <epic>. <one-line scope>.`
- Inline code with backticks; fenced blocks only for meaningful snippets or tracebacks.
- Bug descriptions: paste a traceback or attach an image, don't write prose.
- If a Todo / In Progress issue has an empty description, ask the user first.

Before creating an issue (only when explicitly asked):
- Required: `title`, `team: "team-samuli"`.
- Set `priority`: High (2) for blocking work, Medium (3) for normal features.
- Optional `label`, pick one: `Feature`, `Bug`, `Test`, `Documentation`, `Theory`, `Improvement`.
- Set `milestone` from the existing list — do not invent new ones.
- Set `parentId` if it's a sub-issue of an epic.
- Title: imperative verb or noun phrase. No `[WIP]`, `[blocked]`, `[draft]` markers.
- Do not set `estimate`, `dueDate`, or `cycle` — unused in OPS.
- Do not create retroactive issues for completed work.