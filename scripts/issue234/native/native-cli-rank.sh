#!/usr/bin/env bash
# SOURCE candidate only; reviewed fixture staging/launcher supplies every path.
set -uo pipefail
binary=$1; cell=$2; trial=$3; width=$4; case_name=$5; phys=$6; opt_trans=$7; initial=$8
rank=${PMI_RANK:?Hydra PMI_RANK is required}
world=${PMI_SIZE:?Hydra PMI_SIZE is required}
unset MVMC_C_TIMER MVMC_TIMER MVMC_NSTEPS
[[ $rank =~ ^(0|[1-9][0-9]*)$ && $world =~ ^(2|4)$ ]] || exit 2
[[ $trial =~ ^(0|1)$ && $width =~ ^(1|2)$ && $case_name =~ ^[a-z][a-z0-9-]*$ ]] || exit 2
[[ $phys =~ ^(0|1)$ && $opt_trans =~ ^(0|1)$ ]] || exit 2
(( rank < world )) || exit 2
input="$cell/rank-$rank/inputs/namelist.def"
fixed="$cell/rank-$rank/fixed.dat"
output="$cell/rank-$rank/output"
if test -f "$cell/rank-$rank/select-missing-namelist"; then input="$cell/rank-$rank/missing.def"; fi
if test -f "$cell/rank-$rank/select-missing-fixed"; then fixed="$cell/rank-$rank/missing-fixed.dat"; fi
args=("$input" --seed 11272 --nsteps 1 --nsmp 1 --initial-def "$initial" --out-dir "$output")
if test "$phys" = 1; then args+=(--physcal "$fixed"); fi
if test "$opt_trans" = 1; then args+=(--opt-trans); fi
# Initial-path controls must be equal in kind across ranks, even when only the
# fault rank is missing its explicit file. Absolute paths may differ.
if test "$initial" = path; then
 args=("$input" --seed 11272 --nsteps 1 --nsmp 1 --initial-def "$cell/rank-$rank/initial.def" --out-dir "$output")
 if test "$opt_trans" = 1; then args+=(--opt-trans); fi
 if test "$phys" = 1; then args+=(--physcal "$fixed"); fi
fi
(set -o noclobber; printf '%q ' "$binary" "${args[@]}" > "$cell/rank-$rank/argv-receipt.txt") || exit 2
printf 'LC_ALL=%s OPENBLAS_NUM_THREADS=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s timer=disabled\n' "${LC_ALL-}" "${OPENBLAS_NUM_THREADS-}" "${OMP_NUM_THREADS-}" "${MKL_NUM_THREADS-}" > "$cell/rank-$rank/execution-settings.txt"
printf 'ISSUE178_CLI_START rank=%s world=%s trial=%s width=%s case=%s\n' "$rank" "$world" "$trial" "$width" "$case_name"
source "${BASH_SOURCE[0]%/*}/wait-cli.sh"
source "${BASH_SOURCE[0]%/*}/inventory-builtins.sh"
install_notification_handler "$cell/rank-$rank/notifications.txt" || exit 2
if "$binary" "${args[@]}"; then status=0; else status=$?; fi
inventory_status=0
collect_inventory "$output" "$cell/rank-$rank/output-inventory.txt" || inventory_status=$?
output_exists=false
if test -e "$output"; then output_exists=true; fi
printf 'ISSUE178_CLI_RETURN rank=%s world=%s trial=%s width=%s case=%s status=%s output_exists=%s\n' "$rank" "$world" "$trial" "$width" "$case_name" "$status" "$output_exists"
printf 'ISSUE178_CLI_DONE rank=%s world=%s trial=%s width=%s case=%s status=%s\n' "$rank" "$world" "$trial" "$width" "$case_name" "$status"
if ((inventory_status)); then
 printf 'CLI inventory artifact failure=%s actual_cli_status=%s\n' "$inventory_status" "$status" >&2
 exit 2
fi
exit "$status"
