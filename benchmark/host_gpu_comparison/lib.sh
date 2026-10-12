#!/usr/bin/env bash
# Shared helpers for the host E2E comparison (issue #544). Source after env.sh.

# prepare_input <dest_dir> <samples_per_chain> <calmode> [<nsmp>]
prepare_input() {
    local dest=$1 samples=$2 calmode=$3 nsmp=${4:-1}
    rm -rf "$dest"
    cp -r "$INPUT_SRC" "$dest"
    sed -i -E "s/^NVMCSample[ \t]+.*/NVMCSample      ${samples}/" "$dest/modpara.def"
    sed -i -E "s/^NVMCWarmUp[ \t]+.*/NVMCWarmUp      ${WARMUP_SWEEPS}/" "$dest/modpara.def"
    sed -i -E "s/^NVMCInterval[ \t]+.*/NVMCInterval    ${INTERVAL_SWEEPS}/" "$dest/modpara.def"
    sed -i -E "s/^NSROptItrStep[ \t]+.*/NSROptItrStep  ${STEPS}/" "$dest/modpara.def"
    sed -i -E "s/^NSROptItrSmp[ \t]+.*/NSROptItrSmp   ${nsmp}/" "$dest/modpara.def"
    sed -i -E "s/^RndSeed[ \t]+.*/RndSeed        ${SEED}/" "$dest/modpara.def"
    sed -i -E "s/^NVMCCalMode[ \t]+.*/NVMCCalMode    ${calmode}/" "$dest/modpara.def"
    sed -i -E "s/^NDataQtySmp[ \t]+.*/NDataQtySmp    ${PHYSCAL_RUNS}/" "$dest/modpara.def"
}

# now <var> : store the current nanosecond timestamp in the named variable.
now() { printf -v "$1" '%s' "$(date +%s%N)"; }

# elapsed <start_ns> <end_ns> : print seconds with microsecond resolution.
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.6f", (b-a)/1e9}'; }

# all_timer <out_dir> : the internal "All [0]" seconds, or empty when absent.
all_timer() {
    local f=$1/zvo_CalcTimer.dat
    [ -f "$f" ] || { printf ''; return; }
    awk '$1 == "All" { print $NF; exit }' "$f"
}

# run_timed <csv> <case> <phase> <rep> <out_dir> <cmd...>
# Appends "case,phase,rep,wall_s,all_s" and captures stdout/stderr to a log.
run_timed() {
    local csv=$1 case=$2 phase=$3 rep=$4 out_dir=$5
    shift 5
    local log="$out_dir/${phase}-rep${rep}.log" start end wall all
    mkdir -p "$out_dir"
    now start
    "$@" >"$log" 2>&1 || { echo "FAILED: $* (see $log)" >&2; tail -n 5 "$log" >&2; return 1; }
    now end
    wall=$(elapsed "$start" "$end")
    all=$(all_timer "$out_dir")
    printf '%s,%s,%d,%s,%s\n' "$case" "$phase" "$rep" "$wall" "$all" >>"$csv"
    printf '  %-8s rep=%d wall=%-10s all=%s\n' "$phase" "$rep" "$wall" "${all:-n/a}"
}

# record_env <file> : environment and provenance block.
record_env() {
    local out=$1
    {
        echo "# $(date -u)"
        uname -a
        command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total \
            --format=csv,noheader
        command -v nvcc >/dev/null && nvcc --version | tail -1
        rustc --version; cargo --version
        command -v "$JULIA_BIN" >/dev/null && "$JULIA_BIN" --version
        git -C "$ROOT" rev-parse HEAD
        git -C "$ROOT" status --short
        for f in "$INPUT_SRC"/*; do sha256sum "$f"; done
    } >"$out" 2>&1
}

# make_checkpoint <bin> <work_dir> <out_dir> [<mpi_prefix...>]
# Runs a two-point Opt so the writer emits parameter triples the PhysCal reader
# accepts. Returns the fixed-parameter file path on stdout.
make_checkpoint() {
    local bin=$1 work=$2 out=$3
    shift 3
    prepare_input "$work" "$SAMPLES" 0 2
    ( cd "$work" && "$@" "$bin" namelist.def --nsteps 2 --nsmp 2 --seed "$SEED" --out-dir "$out" >/dev/null ) || return 1
    printf '%s\n' "$out/zqp_opt.dat"
}
