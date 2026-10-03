#!/usr/bin/env bash
set -euo pipefail

# Optional reference tooling. Ordinary Rust tests do not execute Julia.
version=1.13.1
archive="julia-${version}-linux-x86_64.tar.gz"
tools_dir=/home/vscode/.cache/mvmc/tools
mkdir -p "$tools_dir"
if [[ ! -x "$tools_dir/julia-${version}/bin/julia" ]]; then
    download_dir=$(mktemp -d)
    trap 'rm -rf "$download_dir"' EXIT
    curl --fail --location --proto '=https' --tlsv1.2 \
        "https://julialang-s3.julialang.org/bin/linux/x64/1.13/${archive}" \
        -o "$download_dir/$archive"
    curl --fail --location --proto '=https' --tlsv1.2 \
        "https://julialang-s3.julialang.org/bin/checksums/julia-${version}.sha256" \
        -o "$download_dir/checksums"
    awk -v file="$archive" '$2 == file {print}' "$download_dir/checksums" \
        > "$download_dir/selected.sha256"
    test -s "$download_dir/selected.sha256"
    (cd "$download_dir" && sha256sum -c selected.sha256)
    tar xzf "$download_dir/$archive" -C "$tools_dir"
fi
"$tools_dir/julia-${version}/bin/julia" --version
printf 'Reference command: JULIA_DEPOT_PATH=%s/julia-depot %s/julia-%s/bin/julia --project=extern/Julia-mVMC\n' \
    /home/vscode/.cache/mvmc "$tools_dir" "$version"
