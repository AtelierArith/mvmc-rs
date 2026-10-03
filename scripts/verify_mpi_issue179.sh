#!/usr/bin/env bash
# Optional live-MPI gate; run inside the Linux container from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/mpi_issue179_environment.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
limit=${MPI179_TIMEOUT:-30}
out=$(mktemp -d /tmp/mvmc-issue179-evidence.XXXXXX)
printf '%s\n' "$out"
{ git -c safe.directory="$PWD" rev-parse HEAD; git -c safe.directory="$PWD" diff --stat; rustc -Vv; mpirun --version; uname -a; pkg-config --modversion openblas || true; } > "$out/provenance.txt"
git -c safe.directory="$PWD" ls-files --cached --others --exclude-standard -z -- crates scripts Cargo.toml Cargo.lock rust-toolchain.toml .cargo third_party | sort -zu | xargs -0 sha256sum > "$out/source-sha256.txt"
sha256sum "$out/source-sha256.txt" >> "$out/provenance.txt"
if [[ ${MPI179_PREPARE_ONLY:-0} != 1 ]]; then
    timeout --kill-after=10s 600s cargo build --locked --profile test-fast -p mvmc-cli --features mpi > "$out/build.log" 2>&1
    timeout --kill-after=10s 600s cargo test --locked --profile test-fast -p mvmc-core --features mpi --test mpi_issue179_mapping --no-run --message-format=json > "$out/preflight-build.json" 2> "$out/preflight-build.log"
    preflight_bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_mapping" and .executable!=null) | .executable' "$out/preflight-build.json")
    [[ -x $preflight_bin ]]
    source scripts/mpi_issue179_selection.sh
    mpi179_require_test "$preflight_bin" issue179_group_width_endpoints
    for ranks in 2 4; do
        # Multiple launcher processes are not proof of a real MPI world.
        if ! MPI179_EXPECT_RANKS="$ranks" timeout --kill-after=5s "${limit}s" "${mpi179_mpirun[@]}" -n "$ranks" "$preflight_bin" --ignored --exact issue179_group_width_endpoints --nocapture > "$out/preflight-$ranks.log" 2>&1; then
            echo "MPI_PREFLIGHT_FAILED: expected real world size $ranks; see $out/preflight-$ranks.log" >&2
            exit 1
        fi
    done
fi
bin="$CARGO_TARGET_DIR/test-fast/mvmc"
if [[ ${MPI179_PREPARE_ONLY:-0} != 1 ]]; then ldd "$bin" > "$out/linked-libraries.txt"; fi
printf 'cell\tranks\tmode\tsplit\tcg\tstore\tprojection\texpected\tstatus\texit\n' > "$out/matrix.tsv"
bad=0
run_cell() {
    local id=$1 ranks=$2 mode=$3 split=$4 cg=$5 store=$6 projection=$7 expected=$8 pattern=$9
    local dir="$out/$id" rc=0 status fixture="heisenberg_chain_$mode"
    mkdir -p "$dir"
    cp -R "extern/Julia-mVMC/test/integration/reference/$fixture/inputs" "$dir/inputs"
    # Fixtures are independent Julia inputs. InterAll is explicitly outside scope.
    if grep -qi '^[[:space:]]*InterAll' "$dir/inputs/namelist.def"; then
        echo "InterAll fixture forbidden: $fixture" >&2; exit 2
    fi
    awk -v split_size="$split" -v cg="$cg" -v store="$store" -v projection="$projection" -v mode="$mode" '
        $1=="NSROptItrStep" || $1=="NSROptItrSmp" || $1=="NVMCWarmUp" {print $1,1;next}
        $1=="NVMCSample" {print $1,3;next}
        $1=="NSplitSize" {print $1,split_size;next}
        $1=="NSRCG" {print $1,cg;next}
        $1=="NStore" {print $1,store;next}
        projection!="standard" && ($1=="NSPGaussLeg" || $1=="NMPTrans") {if($1=="NSPGaussLeg") seen_sp=1;print $1,1;next}
        projection=="standard" && $1=="NSPGaussLeg" {seen_sp=1;print $1,(mode=="fsz"?1:3);next}
        projection=="standard" && $1=="NMPTrans" {print $1,3;next}
        {print}
        END {if(mode=="fsz" && !seen_sp) print "NSPGaussLeg",1}
    ' "$dir/inputs/modpara.def" > "$dir/modpara.def"
    mv "$dir/modpara.def" "$dir/inputs/modpara.def"
    if [[ $projection != standard ]]; then
        awk 'BEGIN {print "====================\nNQPTrans 1\n====================\nindices\n====================\n0 1.0"; for(i=0;i<6;i++) print 0,i,i,1}' > "$dir/inputs/qptransidx.def"
    else
        awk 'BEGIN {print "====================\nNQPTrans 3\n====================\nindices\n===================="; for(t=0;t<3;t++) print t,1.0; for(t=0;t<3;t++) for(i=0;i<6;i++) print t,i,(i+t)%6,1}' > "$dir/inputs/qptransidx.def"
    fi
    local args=("$dir/inputs/namelist.def" --nsteps 1 --nsmp 1 --mode "$mode" --out-dir "$dir/output")
    if [[ $projection == opttrans ]]; then
        printf 'OptTrans opttrans.def\n' >> "$dir/inputs/namelist.def"
        awk 'BEGIN {print "====================\nNQPOptTrans 2\nComplexType 0\n====================\n====================\n0 1.0\n1 0.5"; for(t=0;t<2;t++) for(i=0;i<6;i++) print t,i,(i+t)%6,1}' > "$dir/inputs/opttrans.def"
        args+=(--opt-trans)
        # Adding two coefficient slots changes the overlay width. Explicitly
        # use randomized initialization for this distinct OptTrans scenario.
        args+=(--initial-def none)
    fi
    if [[ $expected == missing ]]; then args[0]="$dir/absent.def"; fi
    if [[ $expected == output-failure ]]; then touch "$dir/blocked"; args+=(--out-dir "$dir/blocked"); fi
    if [[ ${MPI179_PREPARE_ONLY:-0} == 1 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tNOT_RUN\tNA\n' "$id" "$ranks" "$mode" "$split" "$cg" "$store" "$projection" "$expected" >> "$out/matrix.tsv"
        return
    fi
    sha256sum "$dir"/inputs/* > "$dir/input-sha256.txt"
    timeout --kill-after=5s "${limit}s" "${mpi179_mpirun[@]}" -n "$ranks" "$bin" "${args[@]}" > "$dir/launch.log" 2>&1 || rc=$?
    if (( rc == 124 || rc == 137 )); then status=TIMEOUT; bad=1
    elif [[ $expected != success ]]; then
        if (( rc != 0 )) && grep -Eiq "$pattern" "$dir/launch.log"; then status=REJECTION; else status=FAIL; bad=1; fi
    elif (( rc == 0 )) && [[ -s "$dir/output/zvo_out.dat" ]]; then status=EXECUTION_ONLY
    elif (( rc != 0 )) && grep -Eiq 'unsupported|not supported|NSplitSize.*(requires|with|not)' "$dir/launch.log"; then status=UNSUPPORTED; bad=1
    else status=FAIL; bad=1; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$ranks" "$mode" "$split" "$cg" "$store" "$projection" "$expected" "$status" "$rc" | tee -a "$out/matrix.tsv"
    if [[ -n ${MPI179_JULIA:-} && $expected == success ]]; then
        local jrc=0
        timeout --kill-after=5s "${limit}s" "$MPI179_JULIA" --compiled-modules=existing --project=extern/Julia-mVMC scripts/verify_mpi_issue179_julia_launch.jl "$ranks" scripts/verify_mpi_issue179_reference.jl "$dir/inputs/namelist.def" "$mode" "$dir/julia-output" > "$dir/julia.log" 2>&1 || jrc=$?
        printf '%s\t%s\n' "$id" "$jrc" >> "$out/julia-exits.tsv"
        if (( jrc != 0 )); then bad=1; fi
        # An explicitly requested reference must have an executable comparison
        # gate. Bounds are supplied only after investigating the first divergence;
        # root-output agreement alone never upgrades EXECUTION_ONLY to parity.
        if [[ -s "$dir/output/zvo_out.dat" && -s "$dir/julia-output/zvo_out.dat" ]]; then
            local cmp_rc=0
            if [[ -z ${MPI179_ROOT_ATOL:-} || -z ${MPI179_ROOT_RTOL:-} || ! -s ${MPI179_JUSTIFICATION:-/dev/null} ]]; then
                echo 'root output numerical bounds and justification required' > "$dir/root-comparison-status.txt"
                bad=1
            fi
            awk -v atol="${MPI179_ROOT_ATOL:-invalid}" -v rtol="${MPI179_ROOT_RTOL:-invalid}" 'function number(s, v) {v=s+0;return s ~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ && v<=1.7976931348623157e308 && v>=-1.7976931348623157e308}
                BEGIN {if(!number(atol) || !number(rtol) || atol<0 || rtol<0) {bad=1;exit 1}}
                NR==FNR {if(!NF) {bad=1;exit 1}; for(i=1;i<=NF;i++) {if(!number($i)) {bad=1;exit 1}; x[FNR,i]=$i}; cols[FNR]=NF; rows=FNR;next}
                {if (NF!=cols[FNR]) {print "shape mismatch",FNR;exit 1}
                 for(i=1;i<=NF;i++) {if(!number($i)) {bad=1;exit 1}; d=$i-x[FNR,i]; if(d<0)d=-d; ax=x[FNR,i];if(ax<0)ax=-ax;ay=$i;if(ay<0)ay=-ay;print FNR,i,x[FNR,i],$i,d; if(d>atol+rtol*(ax>ay?ax:ay)) bad=1}}
                END {if(bad || FNR!=rows || rows!=1) {print "malformed/row mismatch";exit 1}}' "$dir/output/zvo_out.dat" "$dir/julia-output/zvo_out.dat" > "$dir/root-output-deltas.txt" || cmp_rc=$?
            if (( cmp_rc != 0 )); then bad=1; fi
        else
            echo 'missing root output comparison' > "$dir/root-output-deltas.txt"
            bad=1
        fi
    fi
}
for ranks in 2 4; do
    for mode in real cmp fsz; do
        for split in 1 2; do
            for cg in 0 1; do
                for store in 0 1; do
                    expected=success pattern=''
                    if (( split > 1 && cg == 1 )); then expected=rejection; pattern='NSplitSize|SR-CG'; fi
                    run_cell "r${ranks}-${mode}-s${split}-cg${cg}-store${store}" "$ranks" "$mode" "$split" "$cg" "$store" identity "$expected" "$pattern"
                done
            done
            expected=success pattern=''
            if [[ $mode == fsz ]] && (( split > 1 )); then expected=rejection; pattern='FSZ|standard-projection'; fi
            run_cell "r${ranks}-${mode}-s${split}-standard" "$ranks" "$mode" "$split" 0 1 standard "$expected" "$pattern"
            expected=success pattern=''
            if (( split > 1 )); then expected=rejection; pattern='OptTrans|NQPOptTrans'; fi
            run_cell "r${ranks}-${mode}-s${split}-opttrans" "$ranks" "$mode" "$split" 0 1 opttrans "$expected" "$pattern"
        done
    done
    run_cell "r${ranks}-invalid-cg" "$ranks" real 1 2 1 identity rejection NSRCG
    # C/Julia permit non-divisible communicator widths with a warning. A Rust
    # restriction is a failed expected-success cell, not a supported rejection.
    run_cell "r${ranks}-uneven-split3" "$ranks" real 3 0 1 identity success ''
    run_cell "r${ranks}-missing" "$ranks" real 1 0 1 identity missing 'absent|read|open|exist'
    run_cell "r${ranks}-output-failure" "$ranks" real 1 0 1 identity output-failure 'directory|exists|output|Not a directory'
    if (( ranks == 4 )); then
        run_cell r4-real-s4-empty-rank 4 real 4 0 1 identity success ''
    fi
done
printf 'Trajectory/624-word RNG/local accumulators/SR parity: NOT OBSERVED by CLI; run verify_mpi_issue179_states.sh.\nEmpty weights and per-move proposals/acceptance traces: NOT COVERED by CLI matrix.\n' > "$out/coverage-gaps.txt"
git -c safe.directory="$PWD" ls-files --cached --others --exclude-standard -z -- crates scripts Cargo.toml Cargo.lock rust-toolchain.toml .cargo third_party | sort -zu | xargs -0 sha256sum > "$out/source-sha256-after.txt"
if ! cmp -s "$out/source-sha256.txt" "$out/source-sha256-after.txt"; then
    echo 'SOURCE_CHANGED: shared source changed during verification; rebuild/reverify before milestone validation' >> "$out/provenance.txt"
    bad=1
fi
echo "Evidence: $out"
exit "$bad"
