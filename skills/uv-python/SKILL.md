---
name: uv-python
description: Use uv to run Python scripts and modules, manage Python dependencies, and create environments. Apply whenever Python execution or package operations are needed.
---

Use uv for Python execution and dependency operations. Preserve the repository's Python version requirements, declared dependencies and lock files. The user has chosen uv; use it for inline Python as well as saved scripts.

## Choose the execution context

Inspect `uv --version` and applicable `pyproject.toml`, `uv.lock`, `.python-version` and script metadata. For a Python project, run commands in its project context:

```sh
uv run python script.py
uv run python -m pytest
```

Use `--locked` when validation must preserve an existing lock file. Do not add `--no-project` to commands that need the project's dependencies.

For independent utility scripts in a Rust or other non-Python repository, avoid installing an unrelated Python project:

```sh
uv run --no-project python scripts/check_reference.py
uv run --no-project python - <<'PYTHON'
from pathlib import Path
print(Path.cwd())
PYTHON
```

Supply one-off external dependencies through uv rather than installing them globally:

```sh
uv run --no-project --with PyYAML python path/to/quick_validate.py path/to/skill
```

For a maintained project's dependency change, use `uv add` / `uv remove` and review the resulting manifest and lock changes. For a maintained standalone script, use its existing inline dependency metadata or `uv add --script`; do not create project manifests just to run a utility. Use `uv tool run` for standalone Python command-line applications.

## Environments and constrained filesystems

Use `uv venv` when a persistent virtual environment is actually needed. If operating on an existing environment with requirements files, use `uv pip install --python <environment-python> -r requirements.txt`; specify the intended interpreter instead of relying on an unrelated active environment. Keep virtual environments and caches out of commits.

When the default cache is unwritable, set `UV_CACHE_DIR` to a writable location. Preserve Python selection requirements; use `--no-python-downloads` when an available interpreter suffices and no download is needed. If a managed Python installation is required, ensure its installation directory is writable rather than falling back to global package installation.

Validate through the actual uv invocation. Report the relevant script/test result; successful dependency resolution alone is not proof that the Python operation succeeded. No broader environment migration or dependency upgrade is implied by a request to run a script.

Consult the official [script guide](https://docs.astral.sh/uv/guides/scripts/), [project guide](https://docs.astral.sh/uv/guides/projects/), and [cache documentation](https://docs.astral.sh/uv/concepts/cache/) when flags or behavior need checking.
