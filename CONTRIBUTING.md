# Contributing

## Workflow

1. Pick a task and say you are taking it.
2. Branch from an up-to-date `main`: `git checkout -b feature/short-name` (or `fix/...`, `docs/...`).
3. Make small commits with a clear first line, for example `Add a chart of scores per fold`.
4. Run the notebook top to bottom before you push, so it still works and its saved results are current. If the data or the
   model changed, version them with DVC and a tag (README) and `dvc push` so teammates can `dvc pull`.
5. Open a pull request into `main` and describe what changed and how you checked it.
6. Someone else reviews and approves. Nobody pushes straight to `main`, and you don't merge your own PR.

## Setup

See the README: either create a virtual environment and `pip install -r requirements.txt`, or run `docker compose up --build` for JupyterLab in a container. Either way, put `train.csv` in `data/` first.

## Rules of thumb

- **No data, models or secrets in git.** Data and the model file are versioned with DVC (git keeps only the small `.dvc`
  pointer files); `.dvc/config.local` and `.env` stay on your computer. See *Data and model versioning* in the README.
- **Keep the notebook simple and top-to-bottom.** Short cells, plain names, a sentence or two of explanation before each
  step. Someone should be able to read it without knowing the code.
- **Honest scores only.** Always evaluate on whole held-out prompts (split by group), never random rows. Say which numbers
  are local estimates and which are not.
- **Don't save private paths in the notebook.** Check that output cells don't contain folders from your computer.
- **Keep PRs small.** One idea per PR is easier to review.
- **Note AI help.** The course requires listing assisting tools in submitted work; add yours to the PR description or the README.
