#!/usr/bin/env bash
# SOURCE only: requires a reviewed MPI CLI binary/source/native preflight manifest.
set -euo pipefail
checkout=$1;binary=$2;pins=$3
here=$(cd "$(dirname "$0")" && pwd)
root=$(mktemp -d "${TMPDIR:?exclusive artifact parent}/mvmc-234-cli-native.XXXXXX")
printf 'Native CLI output root=%s\n' "$root"
aggregate=0
budget(){
 available=$(df -PB1 "$root" | awk 'NR==2{print $4}')
 size=$(du -sb "$root" | awk '{print $1}')
 if ((available<384*1024*1024 || size>20*1024*1024));then
  printf 'CLI capacity guard available=%s payload=%s reserve=384MiB budget=20MiB\n' "$available" "$size" >&2;return 1
 fi
}
finish(){ status=$?;trap - EXIT;if ((status));then aggregate=1;fi
 if ! sha256sum -c --quiet "$pins" > "$root/pins-post.log" 2>&1;then aggregate=1;fi
 printf '%s\n' "$status" > "$root/launcher.status";printf '%s\n' "$aggregate" > "$root/aggregate.status";exit "$aggregate"; }
trap finish EXIT
budget
sha256sum -c --quiet "$pins" > "$root/pins-pre.log" 2>&1
for required in "$binary" "$here/native-cli-stage.sh" "$here/native-cli-rank.sh" "$here/native-cli-validate.awk" "$here/native-cli-inventory-validate.sh" "$here/full-matrix-SOURCE.sh" "$here/wait-cli.sh" "$here/inventory-builtins.sh" "$here/native-cli-validator-controls.sh" "$here/native-cli-inventory-controls.sh";do
 awk -v path="$required" '$2==path {n++} END{exit(n!=1)}' "$pins" || { printf 'Unbound required native input %s\n' "$required" >&2;exit 2; }
done
bash "$here/native-cli-validator-controls.sh" > "$root/marker-controls.log" 2>&1
bash "$here/native-cli-inventory-controls.sh" > "$root/inventory-controls.log" 2>&1
printf 'Actual selected binary and all execution scripts required in preflight pins; staged fixture records additionally bound per cell before launch.\n' > "$root/association.txt"
mpiexec -help > "$root/hydra-help.txt" 2>&1
for flag in -disable-auto-cleanup -outfile-pattern -errfile-pattern;do rg -q -- "$flag" "$root/hydra-help.txt";done
cell_run(){
 budget
 world=$1;width=$2;bad=$3;trial=$4;name=$5
 cell="$root/$name-w$world-g$width-b$bad-t$trial"
 bash "$here/native-cli-stage.sh" "$checkout" "$binary" "$cell" "$world" "$width" "$bad" "$trial" "$name"
 phys=$(<"$cell/phys");opt=$(<"$cell/opt-trans");initial=$(<"$cell/initial-kind")
 find "$cell" -type f -print0 | sort -z | xargs -0 sha256sum > "$cell-inputs.sha256"
 for ((rank=0;rank<world;rank++));do (set -o noclobber; : > "$cell/rank-$rank.stderr");done
 printf 'Exclusively precreated stderr; zero bytes do not prove Hydra opened it.\n' > "$cell/stderr-origin.txt"
 expected=$(<"$cell/rank-0/expected-status")
 set +e
 LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout -k 10s 120s \
  mpiexec -disable-auto-cleanup -outfile-pattern "$cell/rank-%r.stdout" -errfile-pattern "$cell/rank-%r.stderr" -n "$world" \
  bash "$here/native-cli-rank.sh" "$binary" "$cell" "$trial" "$width" "$name" "$phys" "$opt" "$initial" \
  > "$cell/launcher.stdout" 2> "$cell/launcher.stderr"
 status=$?;set -e;printf '%s\n' "$status" > "$cell/native.status"
 if ((status!=expected));then printf 'Wrong launcher status actual=%s expected=%s cell=%s\n' "$status" "$expected" "$cell" >&2;return 1;fi
 sha256sum -c --quiet "$cell-inputs.sha256" > "$cell/input-post.log" 2>&1
 for ((rank=0;rank<world;rank++));do
  dir="$cell/rank-$rank";test -s "$cell/rank-$rank.stdout"
  cmp "$dir/expected-argv.txt" "$dir/argv-receipt.txt" > "$dir/argv-cmp.log" 2>&1
  awk -v rank="$rank" -v world="$world" -v width="$width" -v trial="$trial" -v case_name="$name" \
   -v expected_status="$(<"$dir/expected-status")" -v expected_output="$(<"$dir/expected-output")" \
   -v diagnostic="$(<"$dir/expected-diagnostic")" -f "$here/native-cli-validate.awk" \
   "$cell/rank-$rank.stdout" "$cell/rank-$rank.stderr" > "$dir/validation.log" 2>&1
  bash "$here/native-cli-inventory-validate.sh" "$dir/output-inventory.txt" "$rank" "$bad" "$expected" "$phys" "$name" > "$dir/inventory-validation.log" 2>&1
 done
 printf 'CELL_VALIDATED %s\n' "$cell"
 budget
}
for world in 2 4;do
 for width in 1 2;do
  # Complete C-supported overlays must succeed before asymmetric corruption.
  for trial in 0 1;do
   for name in positive-overlay-opt positive-overlay-phys;do cell_run "$world" "$width" 0 "$trial" "$name";done
  done
  for bad in 0 "$((world-1))";do for trial in 0 1;do
   for name in parse-opt parse-phys fixed-missing fixed-short fixed-token initial-opt overlay-opt overlay-phys output-opt output-phys;do
    cell_run "$world" "$width" "$bad" "$trial" "$name"
   done
  done;done
  for trial in 0 1;do
   for name in positive-opt positive-phys positive-phys-cg positive-phys-ap positive-phys-initial;do cell_run "$world" "$width" 0 "$trial" "$name";done
  done
 done
 for trial in 0 1;do
  for name in grouped-cg grouped-phys-fsz grouped-opt-fsz-spin grouped-opt-fsz-trans grouped-opt-fsz-ap grouped-opt-opttrans grouped-phys-opttrans grouped-phys-lanczos1 grouped-phys-lanczos2;do
   cell_run "$world" 2 0 "$trial" "$name"
  done
  for name in opt-lanczos1 opt-lanczos2;do cell_run "$world" 1 0 "$trial" "$name";done
 done
done
printf 'Completed native CLI matrix root=%s\n' "$root"
