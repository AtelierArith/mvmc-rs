#!/usr/bin/env bash
# Grouped normal Green CLI positives: execution evidence, not numerical parity.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/mpi_issue179_environment.sh
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
limit=${MPI179_TIMEOUT:-30}
out=$(mktemp -d /tmp/mvmc-issue179-physcal.XXXXXX)
echo "Evidence: $out"
git -c safe.directory="$PWD" rev-parse HEAD > "$out/head.txt"
git -c safe.directory="$PWD" ls-files -co --exclude-standard -z -- crates scripts Cargo.toml Cargo.lock .cargo third_party | sort -zu | xargs -0 sha256sum > "$out/source-sha256.txt"
sha256sum "$out/source-sha256.txt" > "$out/source-manifest.sha256"
timeout --kill-after=10s 600s cargo build --locked --profile test-fast -p mvmc-cli --features mpi > "$out/build.log" 2>&1
bin="$CARGO_TARGET_DIR/test-fast/mvmc"
sha256sum "$bin" > "$out/executable.sha256"
{ rustc -Vv; mpirun --version; uname -a; ldd "$bin"; } > "$out/platform.txt"
printf 'cell\texit\tstatus\n' > "$out/matrix.tsv"
bad=0
for ranks in 2 4; do
    for scenario in real cmp fsz-multiqp real-lanczos; do
        mode=${scenario%%-*}
        expected=success
        [[ $scenario == *-* ]] && expected=rejection
        id="r${ranks}-${scenario}-s2-physcal-green"
        dir="$out/$id"
        mkdir -p "$dir"
        # Keep independently authored Green definitions; no unsupported keyword
        # is silently removed. Fixed parameters are an independent fixture.
        cp -R "extern/Julia-mVMC/test/integration/reference/heisenberg_chain_$mode/inputs" "$dir/inputs"
        cp "tests/fixtures/physcal_181/heisenberg_chain_$mode/zqp_opt.dat" "$dir/fixed.dat"
        if grep -qi '^[[:space:]]*InterAll' "$dir/inputs/namelist.def"; then exit 2; fi
        awk -v scenario="$scenario" '
            $1=="NVMCCalMode" {print $1,1;next}
            $1=="NSplitSize" {print $1,2;next}
            $1=="NVMCSample" {print $1,3;next}
            $1=="NVMCWarmUp" || $1=="NDataQtySmp" || $1=="NSPGaussLeg" || $1=="NMPTrans" {print $1,1;next}
            $1=="NLanczosMode" && scenario=="real-lanczos" {print $1,1;next}
            {print}
        ' "$dir/inputs/modpara.def" > "$dir/modpara.def"
        mv "$dir/modpara.def" "$dir/inputs/modpara.def"
        awk 'BEGIN {print "====================\nNQPTrans 1\n====================\nindices\n====================\n0 1.0";for(i=0;i<6;i++)print 0,i,i,1}' > "$dir/inputs/qptransidx.def"
        if [[ $scenario == fsz-multiqp ]]; then
            awk '$1=="NMPTrans" {print $1,3;next} {print}' "$dir/inputs/modpara.def" > "$dir/modpara.def"
            mv "$dir/modpara.def" "$dir/inputs/modpara.def"
            awk 'BEGIN {print "====================\nNQPTrans 3\n====================\nindices\n====================";for(t=0;t<3;t++)print t,1.0;for(t=0;t<3;t++)for(i=0;i<6;i++)print t,i,(i+t)%6,1}' > "$dir/inputs/qptransidx.def"
        fi
        sha256sum "$dir"/inputs/* "$dir/fixed.dat" > "$dir/input-sha256.txt"
        rc=0 status=FAIL
        timeout --kill-after=5s "${limit}s" "${mpi179_mpirun[@]}" -n "$ranks" "$bin" "$dir/inputs/namelist.def" --mode "$mode" --physcal "$dir/fixed.dat" --out-dir "$dir/output" > "$dir/launch.log" 2>&1 || rc=$?
        if (( rc == 124 || rc == 137 )); then status=TIMEOUT
        elif [[ $expected == rejection ]]; then
            if (( rc != 0 )) && grep -Eiq 'FSZ.*(QP|projection|group)|Lanczos.*(unsupported|support|NSplit)|NSplitSize.*(FSZ|Lanczos)' "$dir/launch.log"; then status=REJECTION; fi
        elif (( rc == 0 )) && [[ -s "$dir/output/zvo_cisajs_001.dat" && -s "$dir/output/zvo_cisajscktalt_001.dat" ]]; then status=EXECUTION_ONLY
        fi
        [[ $status == EXECUTION_ONLY || $status == REJECTION ]] || bad=1
        printf '%s\t%s\t%s\n' "$id" "$rc" "$status" | tee -a "$out/matrix.tsv"
    done
done
git -c safe.directory="$PWD" ls-files -co --exclude-standard -z -- crates scripts Cargo.toml Cargo.lock .cargo third_party | sort -zu | xargs -0 sha256sum > "$out/source-sha256-after.txt"
if ! cmp -s "$out/source-sha256.txt" "$out/source-sha256-after.txt"; then
    echo SOURCE_CHANGED > "$out/source-changed.txt"
    bad=1
fi
exit "$bad"
