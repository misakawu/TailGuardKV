# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains planning and research documentation for TailGuardKV, a KV-cache management project for edge LLM serving. Keep the root focused on project-level files such as `README.md`, `AGENTS.md`, and future configuration files. Detailed research notes and paper-planning materials live under `论文规划/`:

- `论文规划/王祯祥_论文工作规划.md`: main work plan.
- `论文规划/进度规划.md`: progress planning.
- `论文规划/实验提取.md`: experiment extraction notes.

If code is added later, place runnable source in a clearly named top-level directory such as `src/`, scripts in `scripts/`, experiments in `experiments/`, and generated outputs in `out/` or `results/`.

## Build, Test, and Development Commands

No build system or automated test runner is currently present. Useful repository checks are:

- `git status`: inspect local changes before editing or committing.
- `find . -maxdepth 2 -type f | sort`: review the current file layout.
- `git log --oneline -5`: check recent commit style and project history.

When future scripts are introduced, document the exact commands here, for example `python3 scripts/run_experiment.py` or `pytest tests/`.

## Coding Style & Naming Conventions

For Markdown, use concise headings, short paragraphs, and repository-relative links where useful. Keep Chinese research notes in Chinese unless a file is explicitly intended for English readers. Prefer descriptive filenames that reflect the document purpose, following the existing Chinese naming pattern in `论文规划/`.

For future Python code, prefer 4-space indentation, `snake_case` for functions and modules, and clear experiment names that encode the model, dataset, and cache policy where applicable.

## Testing Guidelines

There are no tests yet. For documentation-only changes, verify Markdown renders cleanly and links point to existing files. When code is added, include focused tests under `tests/`, name files `test_*.py`, and keep small fixtures or synthetic workloads separate from large datasets.

## Commit & Pull Request Guidelines

Recent commits use short Chinese summaries such as `论文规划文档` and `论文工作规划`. Continue using concise, imperative commit messages that describe the changed artifact or result.

Pull requests should include a brief summary, the motivation for the change, affected files or directories, and any validation performed. For experiment-related changes, include key parameters, input data paths, and output locations.

## Agent-Specific Instructions

Before creating or editing repository guidance files, check whether the target file already exists. Do not overwrite user-authored planning notes unless the request explicitly asks for that edit.
