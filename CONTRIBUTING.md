# Contributing

## Workflow

1. Pick a task and say you are taking it.
2. Branch from an up-to-date `main`: `git checkout -b feature/short-name` (or `fix/...`, `docs/...`).
3. Make small commits with a clear first line, for example `Add a chart of scores per fold`.
4. Run the notebook top to bottom before you push, so it still works and its saved results are current.
5. Open a pull request into `main` and describe what changed and how you checked it.
6. Someone else reviews and approves. Nobody pushes straight to `main`, and you don't merge your own PR.

## Setup

See the README: create a virtual environment, `pip install -r requirements.txt`, put `train.csv` in `data/`.

## Rules of thumb

- **No data or secrets in git.** `data/` is ignored on purpose, because the competition data is not ours to share.
- **Keep the notebook simple and top-to-bottom.** Short cells, plain names, a sentence or two of explanation before each
  step. Someone should be able to read it without knowing the code.
- **Honest scores only.** Always evaluate on whole held-out prompts (split by group), never random rows. Say which numbers
  are local estimates and which are not.
- **Don't save private paths in the notebook.** Check that output cells don't contain folders from your computer.
- **Keep PRs small.** One idea per PR is easier to review.
- **Note AI help.** The course requires listing assisting tools in submitted work; add yours to the PR description or the README.
