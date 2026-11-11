#!/usr/bin/env bash
# Optional, isolated provider experiment. Never runs from a Cargo test/build script.
set -euo pipefail
pf_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
pf_probe="$pf_repo/benchmark/pfaffian_faer/provider-prototype"
pf_work="$(mktemp -d /tmp/mvmc731-provider.XXXXXXXX)"
pf_source="${MVMC_PFAFFIAN_FAER_SOURCE:-https://github.com/tensor4all/faer.git}"
printf 'Retaining isolated provider experiment in %s\n' "$pf_work" >&2
git clone --no-checkout --filter=blob:none "$pf_source" "$pf_work/faer" >&2
git -C "$pf_work/faer" checkout --detach 144621059bca3f0348c4ddb64d68428185528daf >&2
git -C "$pf_work/faer" apply "$pf_probe/shared-factor.patch"
mkdir -p "$pf_work/probe/src"
cp "$pf_probe/main.rs" "$pf_work/probe/src/main.rs"
pf_lock=Cargo.lock
if [ "${MVMC_FAER_PROBE_TEAM:-0}" = 1 ]; then pf_lock=Cargo.team.lock; fi
cp "$pf_probe/$pf_lock" "$pf_work/probe/Cargo.lock"
cat > "$pf_work/probe/Cargo.toml" <<EOF
[package]
name = "mvmc731-faer-probe"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
faer = { package = "t4a-faer", path = "$pf_work/faer/faer", default-features = false, features = ["std", "linalg"] }
pfapack = { path = "$pf_repo/crates/pfapack", features = ["blas-backend"] }
num-complex = "0.4"
[features]
default = ["faer-backend"]
faer-backend = []
provider-team = ["faer/threadteam"]
[profile.release]
opt-level = 3
codegen-units = 1
lto = "thin"
EOF
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
export CARGO_TARGET_DIR="$pf_work/target"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
rustc -Vv >&2
pf_features=()
if [ "${MVMC_FAER_PROBE_TEAM:-0}" = 1 ]; then pf_features=(--features provider-team); fi
cargo run --locked --release --manifest-path "$pf_work/probe/Cargo.toml" "${pf_features[@]}"
