#!/usr/bin/env bash
# Harness-only collector: no post-CLI external children to receive Hydra's
# process-group notification. Parent's caught USR1 handler remains installed.
# Exact controlled fixture paths exclude newline characters.
inventory_walk() {
 local base=$1 path relative kind
 [[ -r $base && -x $base ]] || return 2
 local -a entries=("$base"/* "$base"/.[!.]* "$base"/..?*)
 for path in "${entries[@]}"; do
  [[ -e $path || -L $path ]] || continue
  relative=${path#"$inventory_root"/}
  [[ $relative != *$'\n'* ]] || return 2
  if [[ -L $path ]]; then kind=l
  elif [[ -d $path ]]; then kind=d
  elif [[ -f $path ]]; then kind=f
  elif [[ -p $path ]]; then kind=p
  elif [[ -S $path ]]; then kind=s
  elif [[ -b $path ]]; then kind=b
  elif [[ -c $path ]]; then kind=c
  else return 2
  fi
  inventory_rows+=("$relative $kind")
  if [[ $kind = d ]]; then inventory_walk "$path" || return $?; fi
 done
}

collect_inventory() {
 local inventory_root=$1 destination=$2 row i j saved_noclobber
 local LC_ALL=C
 local -a inventory_rows=()
 if [[ -d $inventory_root ]]; then inventory_walk "$inventory_root" || return $?; fi
 # Small fixture inventories: insertion sort uses only shell builtins, with
 # the same C-locale whole-record lexical ordering as the previous sort.
 for ((i=1;i<${#inventory_rows[@]};i++)); do
  row=${inventory_rows[i]}; j=$i
  while ((j>0)) && [[ ${inventory_rows[j-1]} > "$row" ]]; do
   inventory_rows[j]=${inventory_rows[j-1]}; j=$((j-1))
  done
  inventory_rows[j]=$row
 done
 saved_noclobber=0
 [[ $- != *C* ]] || saved_noclobber=1
 set -C
 if exec 3> "$destination"; then :; else
  ((saved_noclobber)) || set +C
  return 2
 fi
 ((saved_noclobber)) || set +C
 for row in "${inventory_rows[@]}"; do
  if printf '%s\n' "$row" >&3; then :; else exec 3>&-; return 2; fi
 done
 exec 3>&-
}
