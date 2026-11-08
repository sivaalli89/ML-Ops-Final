# Contributing

## Workflow

1. Pick an issue (or make one) and say you're taking it.
2. Branch from an up-to-date `main`: `git checkout -b feature/short-name` (or `fix/...`, `docs/...`).
3. Make small commits with a clear first line, e.g. `Add Evidently drift report`.
4. Run `pytest` before you push. It takes a few seconds and needs no data or Docker.
5. Open a pull request into `main`. Describe what changed and how you checked it.
6. Someone else reviews and approves. Nobody pushes straight to `main`, and you don't merge your own PR.

## Setup

See the README (section A). You only need Docker if you work on the API container or the full stack.

## Rules of thumb

- **No data, models or secrets in git.** `data/`, `artifacts/`, `mlruns/` and `.env` are ignored on purpose.
  Share data with DVC; share settings through `.env.example`.
- **Keep the model code in `src/name_that_llm/`.** The notebook is for explanation; the package is what gets
  tested, trained and served.
- **Test what you add.** Use the toy data in `tests/conftest.py` so tests stay fast and need nothing else.
- **Bump the version when the model changes.** If you change the features or the model code, raise the minor
  number of `__version__` in `src/name_that_llm/__init__.py` (0.1.0 -> 0.2.0) in the same PR. Retrains
  under the same number get their last digit automatically. See *Model versioning* in the README.
- **Honest scores only.** Always evaluate on whole held-out prompts (grouped CV), never random rows.
  Say which numbers are local estimates.
- **Keep PRs small.** One idea per PR is easier to review.
- **Note AI help.** The course requires listing assisting tools in submitted work; add yours to the PR
  description or the README.

## Who owns what

| Area | Owner |
|---|---|
| Data, grouping, pipeline (DVC, Airflow) | _name_ |
| Model, experiments, MLflow, tests | _name_ |
| API, Docker, monitoring | _name_ |

(Fill in the names. If you add a `CODEOWNERS` file, GitHub will request reviews from these people automatically.)
