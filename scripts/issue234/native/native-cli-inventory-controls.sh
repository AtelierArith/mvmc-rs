#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(mktemp -d "${TMPDIR:?exclusive artifact parent}/mvmc-234-cli-inventory.XXXXXX")
trap 'printf "%s\n" "$?" > "$root/status"' EXIT
printf '' > "$root/empty"
printf '%s\n' 'zvo_out.dat d' > "$root/obstruction-opt"
printf '%s\n' 'zvo_out_001.dat d' > "$root/obstruction-phys"
printf '%s\n' 'zqp_opt.dat f' 'zvo_out.dat f' 'zvo_var.dat f' > "$root/root-opt"
printf '%s\n' 'zvo_cisajs_001.dat f' 'zvo_cisajscktalt_001.dat f' 'zvo_cisajscktaltex_001.dat f' 'zvo_out_001.dat f' 'zvo_var_001.dat f' > "$root/root-phys"
check(){ bash "$here/native-cli-inventory-validate.sh" "$@"; }
check "$root/empty" 1 0 0 0 positive-opt > "$root/peer-positive.log"
check "$root/obstruction-opt" 1 1 0 0 output-opt > "$root/peer-obstruction-opt.log"
check "$root/obstruction-phys" 0 0 1 1 output-phys > "$root/root-obstruction-phys.log"
check "$root/root-opt" 0 0 0 0 positive-opt > "$root/root-opt.log"
check "$root/root-phys" 0 0 0 1 positive-phys > "$root/root-phys.log"
printf '%s\n' 'unexpected f' > "$root/extra-peer"
printf '%s\n' 'zvo_out.dat f' > "$root/wrong-obstruction"
sed '/zvo_var/d' "$root/root-opt" > "$root/missing-root"
cp "$root/root-phys" "$root/extra-root";printf '%s\n' 'unknown f' >> "$root/extra-root"
printf '%s\n' 'nested d' 'nested/unknown f' > "$root/nested-peer"
ln -s "$root/empty" "$root/link"
negative(){ name=$1;shift;if check "$@" > "$root/$name.log" 2>&1;then printf 'FAIL accepted %s\n' "$name" >&2;exit 1;fi; }
negative missing "$root/absent" 1 0 0 0 positive-opt
negative peer-extra "$root/extra-peer" 1 0 0 0 positive-opt
negative obstruction-type "$root/wrong-obstruction" 1 1 0 0 output-opt
negative root-missing "$root/missing-root" 0 0 0 0 positive-opt
negative root-extra "$root/extra-root" 0 0 0 1 positive-phys
negative nested-peer "$root/nested-peer" 1 0 0 0 positive-opt
negative symlink "$root/link" 1 0 0 0 positive-opt
printf 'INVENTORY_CONTROLS positive=5 negative=7 root=%s\n' "$root"
