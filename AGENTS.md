## Repo boundary (evio)

`PROJECT.md` is the contract for what evio is and is not (its Scope and Non-goals); change
it only through a deliberate edit, never in passing.

- Nothing here may import or name `evio_ops` or `evior`: ops and evior depend on evio,
  never the reverse. `tests/test_repo_boundary.py` enforces it.
- Promoting a detection method from evior: move the module into `src/evio/detection/`,
  register it in `detection/pipeline/registry.py`, delete it from evior, tag a release, and
  bump the tag in ops.

## Code

Readable first. Before finishing:
- Check the names you added are clear, remove redundant type checking and error handling,
  and keep the added lines to a minimum.
- Keep scripts under `./scripts` tidy: preferably just a `main()`. Helper functions and
  classes belong in modules.

All Python must pass Ruff without warnings; never disable a rule with `# noqa` unless asked.
- The gate is `make check` (lint plus the laptop tests). `make fast` is lint plus the quick
  tier, for the edit loop.
- Spelled out: `uv run --frozen --extra dev ruff check .` (add `--fix` for the fixable ones)
  and `uv run --frozen --extra dev pytest -q`. `--extra dev` is required because ruff and
  pytest live in that extra; `--frozen` keeps `uv.lock` pinned, since a plain `uv run`
  re-resolves it for the host and strips this repo's Linux/xpu package set.
  `tests/test_gate_spelling.py` fails if this file, `README.md` and the `Makefile` disagree.
- The task is not complete while Ruff reports errors. Follow the existing import order and
  formatting; the config is in `pyproject.toml`.

## Tests

- Every test module declares one tier with `pytestmark = pytest.mark.t1`, `t2` or `hardware`;
  `tests/test_tier_enforcement.py` checks it. `t1` is the quick tier and must stay pure: at
  run time a `t1` test that reaches for a process, a subprocess or the clock fails and
  belongs in `t2`. `hardware` needs a camera: `make test` leaves it out, `make test-hardware`
  runs it.
- `tests/fixtures/tier_manifest.json` holds lower bounds on the test count per tier. Raise
  it in the same merge request that adds tests, and never lower it to make a run pass.
- A new test double lives in `tests/fakes/`, not inline in a test
  (`tests/test_no_adhoc_fakes.py`); the list of older inline doubles may only shrink. A
  double must be able to express the property its test checks.
- The laptop suite runs with no camera and no network, and leaves no files or processes
  behind (the leak gate in `tests/conftest.py`). Write only under `tmp_path`.

## Git

Feature branches off `dev`, the integration branch, land through GitLab merge requests, and
a human promotes `dev` to `main` on their own schedule. The GitLab project enforces the
rest of the flow (merge method, passing pipeline, protected branches, squash): read the
project settings, not this file, for those. The `gitlab-usage` skill shows how.

Branches:
- Name a branch after its Linear issue: use the branch name Linear generates for it
  (`feature/sn-NN-<title>`). With no issue, use `<type>/<short-slug>` (`feat/`, `fix/`,
  `refactor/`, `chore/`, `docs/`, `test/`).
- Branch from `dev`, not from `main` and not from another feature branch. A branch may
  cover several issues; list all the SN-IDs in the merge request title.

Commits:
- Conventional prefix (`feat:`, `fix:`, `refactor:`, `chore:`, `test:`, `docs:`), then a
  lowercase subject of about 5-10 words. A body is optional, a couple of lines at most.
- No SN-IDs in subjects or bodies: the branch name and the MR title carry them. No
  `Closes: SN-X` / `Fixes: SN-X` trailers.
- Aim for 2-5 cohesive commits per branch.
- A `Co-Authored-By` trailer naming the model is fine when Claude authored the changes.

Merge requests:
- Target `dev`, never `main`. Title: imperative, with the SN-IDs in parentheses, for
  example `(SN-XXX, SN-YYY)`.
- Bring the branch up to date with `dev` the way the project requires and run the gate on
  it before opening the merge request.
- Delete the local feature branch once the merge request has merged: `git branch -d <branch>`.

Pushing:
- Push the feature branch only: `git push -u origin <branch>`. After a rebase, update an
  already-pushed branch with `git push --force-with-lease`; no other `--force`, and no
  `--no-verify` or `--no-gpg-sign` unless asked. Never push to `main` or `dev` directly.
- In an interactive session, show `git log --oneline -10` and wait for the user's go-ahead
  before pushing. A run dispatched by the orchestrator pushes its own feature branch and
  opens the merge request; it never waits.

Releasing (ops pins evio by tag):
- Tag `vX.Y.Z` on a commit of `dev` once it is green, and push the tag.
- Never move or delete a pushed tag; a fix is the next patch version.

## Linear

Project evio, team `team-samuli`, prefix `SN-`. Never touch issues outside it. Issue
structure, workflow states and labels are defined by the project-workflow plugin, not here.
