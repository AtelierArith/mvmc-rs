#!/usr/bin/env bash
# Explicit synthetic quadrature-size interaction gate, not a reference model oracle.
set -euo pipefail
cd "$(dirname "$0")/.."
binary=$(realpath "${1:?state binary required}")
original=$(realpath "${2:?original generated inventory required}")
out=${3:?NEW evidence directory required}
[[ ! -e $out ]] || { echo 'evidence already exists' >&2; exit 2; }
mkdir -p "$out/inventory"
out=$(realpath "$out")
sha256sum "$0" "$binary" > "$out/generator-binary.sha256"
printf 'id\tranks\tmode\tsplit\tcg\tstore\tprojection\texpected\tstatus\texitcode\n' > "$out/inventory/matrix.tsv"
for ranks in 2 4; do
    source="$original/r${ranks}-cmp-s2-cg0-store0/inputs"
    [[ -f $source/namelist.def && -f $source/modpara.def ]] || exit 2
    sha256sum "$source/"* >> "$out/original-inputs.sha256"
    for qp in 31 32 33; do
        id="r${ranks}-cmp-s2-cg0-store0-qp${qp}"
        mkdir -p "$out/inventory/$id/inputs"
        cp "$source/"* "$out/inventory/$id/inputs/"
        # A declared synthetic projector parameter change, not preservation of
        # the old model's quadrature or an independently derived expectation.
        awk -v qp="$qp" '
            $1=="NSPGaussLeg" {if (NF!=2) exit 2; $2=qp; seen++}
            {print}
            END {if (seen!=1) exit 2}
        ' "$source/modpara.def" > "$out/inventory/$id/inputs/modpara.def"
        printf '%s\t%s\tcmp\t2\t0\t0\tstandard\tsuccess\tNOT_RUN\tNA\n' "$id" "$ranks" >> "$out/inventory/matrix.tsv"
    done
done
export MPI179_STEPS_LIST=1 MPI179_CELL_REGEX='^r[24]-cmp-s2-cg0-store0-qp(31|32|33)$'
export MPI179_EXPECTED_CELLS=6 MPI179_EXPECTED_TOTAL=18 MPI179_TIMEOUT=${MPI179_TIMEOUT:-90}
rc=0
bash scripts/verify_mpi_issue179_repeat.sh "$binary" "$out/inventory" "$out/repeats" || rc=$?
printf '%s\n' "$rc" > "$out/repeat.exit"
# Both actual kernel entry evidence and all wrapped collective main-thread
# assertions are required; pool provisioning alone is not activation evidence.
for ranks in 2 4; do
    for qp in 31 32 33; do
        for workers in 1 2 4; do
            for repeat in 1 2; do
                dir="$out/repeats/r${ranks}-cmp-s2-cg0-store0-qp${qp}/prefix1/w$workers/repeat$repeat"
                for ((rank=0; rank<ranks; rank++)); do
                    awk '$1=="d:main-thread-collective-calls" && NF==2 && $2>0 {ok++} END{exit ok!=1}' "$dir/rank-$rank.txt" || rc=1
                    if (( workers>1 )); then
                        grep -Eq 'kernel_parallel_qp_items=[1-9][0-9]*' "$dir/workers-rank-$rank.txt" || rc=1
                    fi
                done
            done
        done
    done
done
sha256sum -c "$out/generator-binary.sha256" > "$out/generator-check.txt" || rc=1
sha256sum -c "$out/original-inputs.sha256" > "$out/original-input-check.txt" || rc=1
printf 'QP_INTERACTION expected_pairs=18 bad=%s synthetic_projector=1 threshold_override=1 transfer_activation=NOT_COVERED\n' "$rc"
exit "$rc"
