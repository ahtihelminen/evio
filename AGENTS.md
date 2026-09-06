## Repo boundary (evioF)

This is the field repo. It must stay runnable on a mast with nothing else
installed.

- Nothing here may import or reference `evior` — evioR depends on evioF,
  never the reverse. `tests/test_repo_boundary.py` enforces it.
- Before adding a dependency, check it is needed by the *appliance*.
  Training, evaluation, plotting, notebooks, Qt and the test-site edge
  integration belong in evioR.
- Promoting a detection method from evioR: move the module into
  `src/evio/detection/`, register it in `detection/pipeline/registry.py`,
  and delete it from evioR.

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
- Run: `uv run ruff check .`
- If auto-fixable issues exist, run: `uv run ruff check . --fix`
- Do not consider the task complete if Ruff reports any errors.
- Never disable Ruff rules with `# noqa` unless explicitly requested.
- Follow the project's existing import ordering and formatting style.
- Ruff config is defined in `pyproject.toml`. Follow it strictly.

## Git

Feature branches off `main`, merged back with `--no-ff` and tagged with SN-IDs in the merge subject.

Before creating a branch:
- Format: `<type>/<short-slug>` — `feat/`, `fix/`, `refactor/`, `chore/`, `docs/`, `test/`.
- Slug: 2-5 hyphenated words, e.g. `feat/evio-detection-emitter`.
- Do not use Linear's auto-generated `feature/sn-NN-<long-title>` names.
- Branch from `main`, not from another feature branch.
- Multi-issue branches are fine — list all SN-IDs in the merge subject.

Before committing:
- Conventional prefix: `feat:`, `fix:`, `refactor:`, `chore:`, `test:`, `docs:`.
- Subject: lowercase after the colon, ~5-10 words.
- Body: optional, at most a couple of lines. Long rationale belongs in Linear.
- No SN-IDs in subjects or bodies — they live on the branch name and merge commit.
- No `Closes: SN-X` / `Fixes: SN-X` trailers (GitHub convention, unused here).
- Aim for 2-5 cohesive commits per branch.
- `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer is fine when Claude authored the changes.

Before merging:
- Run `make lint && make test` (or repo equivalents). Must pass.
- `git checkout main && git merge --no-ff <branch>`.
- Merge subject: `Merge branch '<branch>' into 'main' (SN-XXX)` — multiple issues: `(SN-XXX, SN-YYY)`.
- No body on the merge commit.
- Delete the local feature branch: `git branch -d <branch>`.

Before pushing:
- Wait for an explicit "push" or "go ahead" from the user. Show `git log --oneline -10` and stop.
- `git push origin main`. No `--force`, `--no-verify`, or `--no-gpg-sign` unless asked.
- GitLab MRs are not used for OPS work — direct push to `main` is the pattern.

## Linear

Scope: OPS project, team `team-samuli`, prefix `SN-`. Never touch issues outside this scope.

Before changing issue state:
- Work flows Todo → In Progress → Done. No "In Review" state.
- Start of work: Todo → In Progress (when the feature branch is first pushed).
- After push to main: In Progress → Done.
- Use `mcp__claude_ai_Linear__save_issue` with `state: "In Progress"` or `state: "Done"`.

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