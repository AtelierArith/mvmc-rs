#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -x /opt/mpich/bin/mpiexec ]] && command -v docker >/dev/null 2>&1; then
    mapfile -t containers < <(docker ps -q --filter "label=devcontainer.local_folder=$repo_root")
    if [[ ${#containers[@]} != 1 ]]; then
        echo 'Open this repository in its Linux Dev Container, or start exactly one matching container.' >&2
        exit 1
    fi
    exec docker exec -w /workspaces/mvmc-rs "${containers[0]}" bash bench/run.sh "$@"
fi
cd -- "$repo_root"
exec uv run --no-project python bench/run.py "$@"
