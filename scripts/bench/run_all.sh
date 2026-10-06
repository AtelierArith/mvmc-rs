#!/usr/bin/env bash
# Portable function-level GPU/CPU benchmark and numerical-validation suite (issue #450).
#
#   scripts/bench/run_all.sh [--native|--docker] [--quick|--full] [--out DIR] [--gpu N]
#
#   --native   build and run on this host (CUDA toolkit >= 12.6, Rust >= 1.96, OpenBLAS)
#   --docker   build and run in an nvidia/cuda image with --gpus (host rustup/cargo mounted;
#              override the image with MVMC_RS_CUDA_IMAGE). Default: native when a CUDA
#              toolkit is found, else docker when docker exists.
#   --quick    ~10 minutes (small grids)           [default]
#   --full     ~2 hours on one GPU (A100-class)
#   --out DIR  output directory (default ./bench-out)
#   --gpu N    GPU index (default 0)
#   --preflight-only  run the checks and exit
#
# Primary result: a numerical verdict (PASS/FAIL per function and size) of every GPU/tenferro
# variant against the C-order CPU oracle with explicit bounds. Timings are reference columns.
# The script exits non-zero when any family reports FAIL/ERROR (all families still run).
# Output: DIR/results-<host>-<date>.tar.gz (CSV per family, report.md, logs, metadata.txt).
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode=""
profile=quick
out="./bench-out"
gpu=0
preflight_only=0
inner=0
while [ $# -gt 0 ]; do
  case "$1" in
    --native) mode=native ;;
    --docker) mode=docker ;;
    --quick) profile=quick ;;
    --full) profile=full ;;
    --out) out="$2"; shift ;;
    --gpu) gpu="$2"; shift ;;
    --preflight-only) preflight_only=1 ;;
    --inner) inner=1 ;;
    -h|--help) sed -n 2,22p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

die() { echo "ERROR: $*" >&2; exit 1; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; } # ver_ge have need

# ---------------------------------------------------------------- toolkit detection
cuda_toolkit_version() {
  local v=""
  if command -v nvcc >/dev/null 2>&1; then
    v="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
  fi
  if [ -z "$v" ]; then
    for d in /usr/local/cuda /usr/local/cuda-*; do
      if [ -f "$d/version.json" ]; then
        v="$(sed -n 's/.*"cuda"[^{]*{[^}]*"version"[^"]*"\([0-9][0-9.]*\)".*/\1/p' "$d/version.json" | head -1)"
        [ -z "$v" ] && v="$(grep -o '"version" *: *"[0-9.]*"' "$d/version.json" | sed -n 's/.*"\([0-9.]*\)"/\1/p' | head -1)"
      fi
      [ -n "$v" ] && break
    done
  fi
  if [ -z "$v" ] && command -v ldconfig >/dev/null 2>&1; then
    v="$(ldconfig -p 2>/dev/null | sed -n 's/.*libnvrtc\.so\.\([0-9][0-9]*\(\.[0-9]*\)\?\).*/\1/p' | sort -V | tail -1)"
  fi
  echo "$v"
}

have_cuda_libs() {
  command -v ldconfig >/dev/null 2>&1 || return 1
  local l; l="$(ldconfig -p 2>/dev/null)"
  for lib in libnvrtc libcublas libcusolver; do
    echo "$l" | grep -q "$lib\.so" || return 1
  done
}

have_openblas_dev() { # the unversioned libopenblas.so is needed for linking (-lopenblas)
  pkg-config --exists openblas 2>/dev/null && return 0
  local d
  for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/lib64 \
           /opt/OpenBLAS/lib ${LIBRARY_PATH//:/ } ${LD_LIBRARY_PATH//:/ }; do
    [ -e "$d/libopenblas.so" ] && return 0
  done
  return 1
}

host_preflight() { # checks needed in both modes on the machine that starts the script
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no NVIDIA driver on this host"
  nvidia-smi -L | grep -q "GPU $gpu:" || die "GPU $gpu not found (nvidia-smi -L: $(nvidia-smi -L | tr '\n' ' '))"
  command -v rustc >/dev/null 2>&1 || die "rustc not found (need Rust >= 1.96; install with rustup)"
  local rv; rv="$(rustc --version | sed -n 's/rustc \([0-9.]*\).*/\1/p')"
  ver_ge "$rv" 1.96 || die "rustc $rv is too old (need >= 1.96; run: rustup update stable)"
  command -v cargo >/dev/null 2>&1 || die "cargo not found"
  [ -f "$root/extern/mVMC-1.3.0/CMakeLists.txt" ] || \
    echo "note: extern/ is empty; run 'git submodule update --init --recursive' if the build asks for it" >&2
}

native_preflight() {
  host_preflight
  local tv; tv="$(cuda_toolkit_version)"
  [ -n "$tv" ] || die "no CUDA toolkit found (need >= 12.6: nvcc, or libnvrtc/libcublas/libcusolver in the loader path). Use --docker, or install the toolkit."
  ver_ge "$tv" 12.6 || die "CUDA toolkit $tv is too old (need >= 12.6)"
  have_cuda_libs || die "CUDA toolkit $tv found but libnvrtc/libcublas/libcusolver are not in the loader path (ldconfig -p); set LD_LIBRARY_PATH or install the libraries"
  if ! have_openblas_dev; then
    die "OpenBLAS development files (libopenblas.so for linking) not found (Debian/Ubuntu: apt install libopenblas-dev liblapack-dev; RHEL: dnf install openblas-devel lapack-devel)"
  fi
  command -v python3 >/dev/null 2>&1 || echo "warning: python3 not found; report.md will not be generated" >&2
  echo "preflight OK: CUDA toolkit $tv, rustc $(rustc --version | cut -d' ' -f2), OpenBLAS present"
}

if [ -z "$mode" ]; then
  if [ -n "$(cuda_toolkit_version)" ] && have_cuda_libs; then mode=native
  elif command -v docker >/dev/null 2>&1; then mode=docker
  else mode=native; fi
fi

# ---------------------------------------------------------------- docker mode: re-run inside
if [ "$mode" = docker ] && [ "$inner" = 0 ]; then
  host_preflight
  command -v docker >/dev/null 2>&1 || die "docker not found (use --native, or install docker + nvidia-container-toolkit)"
  image="${MVMC_RS_CUDA_IMAGE:-nvidia/cuda:12.9.1-devel-ubuntu24.04}"
  mkdir -p "$out"; out_abs="$(cd "$out" && pwd)"
  [ "$preflight_only" = 1 ] && {
    docker run --rm --gpus "device=$gpu" "$image" nvidia-smi -L || die "docker cannot access the GPU (install nvidia-container-toolkit)"
    echo "preflight OK (docker image $image)"; exit 0; }
  hostos="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}") / kernel $(uname -r)"
  extra=()
  [ -d "$HOME/.rustup" ] && extra+=(-v "$HOME/.rustup:$HOME/.rustup")
  [ -d "$HOME/.cargo" ] && extra+=(-v "$HOME/.cargo:$HOME/.cargo")
  exec docker run --rm --gpus "device=$gpu" \
    -e HOME="$HOME" -e MVMC_BENCH_HOST_OS="$hostos" -e MVMC_BENCH_HOST_NAME="$(hostname)" \
    -e MVMC_BENCH_IMAGE="$image" -e MVMC_RS_REVISION="$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)" \
    -e CARGO_TARGET_DIR=/work/target-bench-docker \
    "${extra[@]}" -v "$root:/work" -v "$out_abs:/out" -w /work "$image" \
    bash -lc "(( [ -e /usr/lib/x86_64-linux-gnu/libopenblas.so ] || [ -e /usr/lib/aarch64-linux-gnu/libopenblas.so ] ) && command -v pkg-config >/dev/null ) || { apt-get update -qq >/dev/null && apt-get install -y -qq libopenblas-dev liblapack-dev pkg-config >/dev/null; } \
      && exec setpriv --reuid=$(id -u) --regid=$(id -g) --clear-groups bash -c \
      'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd /work && scripts/bench/run_all.sh --native --inner --$profile --out /out --gpu 0'"
fi

native_preflight
[ "$preflight_only" = 1 ] && exit 0

# In docker the GPU was already selected by --gpus; natively select it by index.
[ "$inner" = 0 ] && export CUDA_VISIBLE_DEVICES="$gpu"

# ---------------------------------------------------------------- setup
host="${MVMC_BENCH_HOST_NAME:-$(hostname)}"
date_tag="$(date +%Y%m%d-%H%M%S)"
name="results-${host}-${date_tag}"
mkdir -p "$out"
work="$(cd "$out" && pwd)/$name"
mkdir -p "$work/csv" "$work/logs" "$work/inputs"
rev="${MVMC_RS_REVISION:-$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)}"
gpudir="$root/gpu/mvmc-gpu-cuda"
flag="--$profile"
echo "== mvmc-rs function suite: profile=$profile gpu=$gpu mode=$mode out=$work"

run_step() { # run_step <name> <cmd...>: log to logs/<name>.log, record exit status
  local n="$1"; shift
  echo "-- $n"
  "$@" >"$work/logs/$n.log" 2>&1
  local rc=$?
  echo "$n $rc" >>"$work/status.txt"
  if [ $rc -ne 0 ]; then echo "   $n exited with $rc (see logs/$n.log)"; tail -n 5 "$work/logs/$n.log" | sed 's/^/   | /'; fi
  return 0
}
: >"$work/status.txt"

# ---------------------------------------------------------------- build (fail loudly)
echo "== build (release)"
{
  (cd "$gpudir" && cargo build --release --locked --example function_suite) &&
  (cd "$root" && cargo build --release --locked -p mvmc-cli)
} >"$work/logs/build.log" 2>&1 || { tail -n 30 "$work/logs/build.log"; die "build failed (logs/build.log)"; }
gpu_target="${CARGO_TARGET_DIR:-$gpudir/target}"
# CARGO_TARGET_DIR (docker) is shared by both workspaces; both are built into it
suite="$gpu_target/release/examples/function_suite"
mvmc_bin="${CARGO_TARGET_DIR:-$root/target}/release/mvmc"
[ -x "$suite" ] || die "built binary not found: $suite"
[ -x "$mvmc_bin" ] || die "built binary not found: $mvmc_bin"

# ---------------------------------------------------------------- inputs (Rust StdFace port)
if [ "$profile" = full ]; then ls_list="16 32 64 128 256"; else ls_list="16 32"; fi
echo "== generating Hubbard-chain inputs with mvmc --dry-run: L = $ls_list"
for l in $ls_list; do
  d="$work/inputs/hubbard_chain_L$l"; mkdir -p "$d"
  cat >"$d/StdFace.def" <<EOF
L             = $l
Lsub          = 4
model         = "Hubbard"
lattice       = "chain"
U             = 4.0
t             = 1.0
Ncond         = $l
NSROptItrStep = 300
NVMCSample    = 300
2Sz           = 0
NSPGaussLeg   = 8
NSPStot       = 0
NSplitSize    = 1
NStore        = 1
NSRCG         = 0
DSROptRedCut  = 1e-8
DSROptStaDel  = 1e-2
DSROptStepDt  = 3e-3
RndSeed       = 1
EOF
  "$mvmc_bin" --dry-run "$d/StdFace.def" --out-dir "$d" >"$work/logs/dryrun_L$l.log" 2>&1 \
    || { tail -n 10 "$work/logs/dryrun_L$l.log"; die "mvmc --dry-run failed for L=$l"; }
  [ -f "$d/namelist.def" ] || die "mvmc --dry-run produced no namelist.def for L=$l"
done

# ---------------------------------------------------------------- metadata
meta="$work/metadata.txt"
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$host"
  echo "profile=$profile"
  echo "git_rev=$rev"
  echo "git_dirty=$(git -C "$root" status --porcelain 2>/dev/null | grep -vc '^??' || true)"
  echo "mode=$mode"
  [ -n "${MVMC_BENCH_IMAGE:-}" ] && echo "docker_image=$MVMC_BENCH_IMAGE"
  gname="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  [ -z "$gname" ] && gname="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "gpu_model=$gname"
  echo "gpu_index=$gpu"
  echo "gpu_compute_capability=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
  echo "gpu_memory_total=$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
  echo "gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpu_power_limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit --format=csv,noheader 2>/dev/null | head -1)"
  case "$gname" in
    *A100*) peak="9.7 (19.5 with FP64 tensor cores)" ;;
    *H100*PCIe*|*H100*PCIE*) peak="25.6 (51.2 tensor cores)" ;;
    *H100*|*H200*) peak="34 (67 tensor cores)" ;;
    *V100*) peak="7.8" ;;
    *"RTX 3060"*) peak="0.2 (FP32/64 = 64:1)" ;;
    *"RTX 3090"*) peak="0.56" ;;
    *"RTX 4090"*) peak="1.3" ;;
    *L40S*) peak="1.4" ;;
    *A10*) peak="0.98" ;;
    *) peak="unknown (not in the table)" ;;
  esac
  echo "gpu_fp64_peak_tflops=$peak"
  echo "cpu_model=$(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo | head -1)"
  echo "cpu_logical_cores=$(nproc --all)"
  echo "cpu_available_cores=$(nproc)"
  echo "mem_total_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1e6}' /proc/meminfo)"
  echo "os=${MVMC_BENCH_HOST_OS:-$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}") / kernel $(uname -r)}"
  echo "container_os=$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "cuda_toolkit=$(cuda_toolkit_version)"
  echo "openblas=$(ldconfig -p 2>/dev/null | sed -n 's/.*\(libopenblas[^ ]*\) .*/\1/p' | head -1)"
  for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS RAYON_NUM_THREADS MKL_NUM_THREADS CUDA_VISIBLE_DEVICES; do
    echo "env_$v=${!v:-<unset>}"
  done
  for c in tenferro-gpu tenferro-cpu tenferro-ad tenferro-linalg cudarc faer; do
    echo "crate_$c=$(awk -v c="$c" '$0=="name = \""c"\""{getline; gsub(/version = |"/,""); print; exit}' "$gpudir/Cargo.lock")"
  done
  echo "--- device report (tenferro/cuda libraries)"
  "$suite" meta 2>&1
} >"$meta"

# ---------------------------------------------------------------- families
S="$suite"
common=("$flag")
echo "== family 1: batched Pfaffian/inverse"
run_step pfaffian "$S" pfaffian "${common[@]}" --out "$work/csv/pfaffian.csv"
echo "== family 2: SR stages (all-cores oracle + GPU; then 1-core baseline)"
run_step sr "$S" sr "${common[@]}" --cpu-label allcores --out "$work/csv/sr.csv"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  run_step sr_cpu_1core "$S" sr "${common[@]}" --cpu-only --cpu-label 1core --out "$work/csv/sr_cpu_1core.csv"
echo "== family 2b: device-resident SR (#447 hook)"
if [ -f "$gpudir/examples/bench_sr_resident.rs" ]; then
  # contract: emits the unified CSV schema (benchmark/function_suite/README.md), exit != 0 on FAIL
  run_step sr_resident bash -c "cd '$gpudir' && cargo run --release --locked --example bench_sr_resident -- $flag --out '$work/csv/sr_resident.csv'"
else
  run_step sr_resident "$S" na --name sr_resident --reason "device-resident SR (#447) is not in this checkout (examples/bench_sr_resident.rs absent)" --out "$work/csv/sr_resident.csv"
fi
echo "== family 3: sampler (numerical verdicts, then timings)"
run_step sampler "$S" sampler "${common[@]}" ${MVMC_BENCH_WORK_CAP:+--work-cap "$MVMC_BENCH_WORK_CAP"} --inputs "$work/inputs" --out "$work/csv/sampler.csv"
echo "== family 4: transfers"
run_step transfers "$S" transfers "${common[@]}" --out "$work/csv/transfers.csv"

# ---------------------------------------------------------------- report
echo "== report"
if command -v python3 >/dev/null 2>&1; then py=(python3); elif command -v uv >/dev/null 2>&1; then py=(uv run --no-project python); else py=(); fi
if [ ${#py[@]} -gt 0 ]; then
  "${py[@]}" "$root/scripts/bench/analyze.py" "$work" --out "$work/report.md" >"$work/logs/analyze.log" 2>&1 \
    || { cat "$work/logs/analyze.log"; echo "ERROR: analyze.py failed" >&2; echo "analyze 1" >>"$work/status.txt"; }
else
  echo "no python available: report.md not generated (CSV files are complete)" | tee "$work/report.md"
fi

# ---------------------------------------------------------------- archive and verdict
tar -C "$(dirname "$work")" -czf "$(dirname "$work")/$name.tar.gz" "$name"
echo
echo "== summary"
[ -f "$work/report.md" ] && sed -n '/^## 1\. Numerical validation/,/^## 2\./p' "$work/report.md" | sed '$d' | head -40
echo
echo "archive: $(dirname "$work")/$name.tar.gz"
bad="$(awk '$2 != 0 {print $1}' "$work/status.txt" | tr '\n' ' ')"
if [ -n "$bad" ]; then
  echo "FAILED steps (numerical FAIL/ERROR or run error): $bad" >&2
  exit 1
fi
echo "all families passed their numerical verdicts"
