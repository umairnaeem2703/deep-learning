# CLAUDE.md

Root guidance for the `deep-learning` repo. Each project has its own folder; project-specific rules live in that folder's own `CLAUDE.md` and take precedence over this file.

## Purpose

A monorepo of small, educational deep-learning projects (class assignments and experiments) built mainly in PyTorch. The goal is to understand training behaviour, not to chase state-of-the-art results.

## Layout

- One folder per project (e.g. `mnist/`), each with its own `README.md`, `CLAUDE.md`, and `requirements.txt` listing only what that project adds.
- Projects are independent: don't import code across sibling folders. Copy a small helper instead if it's needed.
- To start a new project, create a sibling folder (ideally on its own branch) rather than editing an existing one.

## Shared conventions

- **Environment:** one shared virtual environment at the repo root, `.venv/` (gitignored). Don't create per-project venvs unless a project needs conflicting package versions.
- **Gitignored everywhere:** `.venv/`, `data/`, `runs/`, `__pycache__/`, `*.pyc`. Also keep model checkpoints (`*.pt`) out of git.
- **Logging:** training metrics go to TensorBoard under each project's `runs/`, one subfolder per run.
- **Style:** keep code simple and readable, with short comments that teach. Don't add frameworks or dependencies without being asked.
- **Explain changes:** say what changed, why, and what to look for in the results. Assume the reader is learning.
- **Run from the project folder:** `cd <project>` before running scripts, so relative paths like `data/` and `runs/` resolve inside it.
