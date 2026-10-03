#!/usr/bin/env bash
set -euo pipefail

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'This development environment requires Linux x86_64.' >&2
    exit 1
fi

# Docker initializes empty named volumes as root. Make them usable by remoteUser.
sudo mkdir -p "$CARGO_HOME" /home/vscode/.cache/mvmc
sudo chown "$(id -u):$(id -g)" "$CARGO_HOME" /home/vscode/.cache/mvmc
mkdir -p "$CARGO_TARGET_DIR" "$UV_CACHE_DIR"

# macOS bind mounts retain the host's UID. Trust this exact workspace only.
workspace_root=$(pwd -P)
git config --global --fixed-value --get safe.directory "$workspace_root" >/dev/null \
    || git config --global --add safe.directory "$workspace_root"

# Historical integration tests read pinned Julia DATA input files. They require
# reference data, but do not invoke Julia or C oracle runtimes.
git submodule update --init --recursive -- extern/Julia-mVMC

rustc --version
cargo --version
cargo nextest --version
kache --version
uv --version
mpicc -show
pkg-config --modversion openblas
