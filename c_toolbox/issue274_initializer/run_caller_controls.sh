#!/usr/bin/env bash
# One optional independent C caller-control acquisition. No Cargo/oracle model.
set -euo pipefail
unset BASH_ENV ENV
p=$(realpath "${1:?exclusive copied source}")
r=$(realpath "${2:?exclusive receipt}")
original=$(realpath "${3:?pinned original vmcmake.c}")
test -d "$p";test -d "$r"
(( $(df -Pk "$r" | awk 'NR==2 {print $4}') >= 392*1024 ))
ulimit -f 16384
finish() {
 status=$?;trap - EXIT TERM INT;set +e
 printf '%s\n' "$status" > "$r/acquisition.status"
 bad=0
 for manifest in source tools providers headers products runtime;do
  if [[ -s $r/$manifest.before.sha256 ]];then
   sha256sum -c --quiet "$r/$manifest.before.sha256" > "$r/$manifest.post.log" 2>&1
   code=$?;printf '%s\n' "$code" > "$r/$manifest.post.status";((code==0)) || bad=1
  else printf 'NOT_STARTED\n' > "$r/$manifest.post.status";fi
 done
 (( $(du -sk "$r" | awk 'NR==1 {print $1}') <= 8192 )) || bad=1
 (( $(df -Pk "$r" | awk 'NR==2 {print $4}') >= 384*1024 )) || bad=1
 ((bad==0)) || exit 74
 exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
sha256sum "$original" "$p/extract_shared_initializer.pl" "$p/caller_control.c" "$p/run_caller_controls.sh" > "$r/source.before.sha256"
cc=$(readlink -f "$(command -v cc)")
cc1=$("$cc" -print-prog-name=cc1);collect2=$("$cc" -print-prog-name=collect2)
for tool in "$cc" "$cc1" "$collect2" bash env timeout perl sha256sum readlink realpath awk du df ldd readelf sort cat;do
 if [[ $tool != /* ]];then tool=$(readlink -f "$(command -v "$tool")");fi
 sha256sum "$tool"
done > "$r/tools.before.sha256"
: > "$r/providers.before.sha256"
while read -r hash tool;do
 if readelf -h "$tool" >/dev/null 2>&1;then
  if ldd "$tool" > "$r/provider.ldd" 2>&1;then
   ! awk '/not found/ {bad=1} END {exit !bad}' "$r/provider.ldd"
   awk '/=>/ && $3 ~ /^\// {print $3} /^[[:space:]]*\// {print $1}' "$r/provider.ldd" |
    while IFS= read -r provider;do sha256sum "$(readlink -f "$provider")";done >> "$r/providers.before.sha256"
  else ! readelf -l "$tool" | awk '/INTERP/ {bad=1} END {exit !bad}';fi
 fi
done < "$r/tools.before.sha256"
"$cc" --version > "$r/compiler.version"
"$cc" -dumpspecs > "$r/compiler.specs"
mkdir "$r/extracted"
perl "$p/extract_shared_initializer.pl" "$original" "$r/extracted"
sha256sum "$r/extracted/shared-initializer.inc" "$r/extracted/source-provenance.txt" >> "$r/source.before.sha256"
printf '%s\n' 'cc -O2 -std=c11 -IEXTRACTED caller_control.c -o caller-control' > "$r/compile.command"
"$cc" -O2 -std=c11 -I"$r/extracted" -M "$p/caller_control.c" > "$r/includes.make"
perl -0777 -e '$s=<>;$s=~s/\\\n/ /g;$s=~s/^[^:]+://;for(split /\s+/,$s){print "$_\n" if length}' "$r/includes.make" | sort -u > "$r/include.paths"
while IFS= read -r path;do sha256sum "$(realpath "$path")";done < "$r/include.paths" > "$r/headers.before.sha256"
set +e
"$cc" -O2 -std=c11 -I"$r/extracted" "$p/caller_control.c" -o "$r/caller-control" > "$r/compile.stdout" 2> "$r/compile.stderr"
code=$?;set -e;printf '%s\n' "$code" > "$r/compile.status";((code==0)) || exit "$code"
sha256sum "$r/caller-control" > "$r/products.before.sha256"
readelf -h "$r/caller-control" > "$r/product.elf"
ldd "$r/caller-control" > "$r/product.ldd"
! awk '/not found/ {bad=1} END {exit !bad}' "$r/product.ldd"
awk '/=>/ && $3 ~ /^\// {print $3} /^[[:space:]]*\// {print $1}' "$r/product.ldd" |
 while IFS= read -r path;do sha256sum "$(readlink -f "$path")";done > "$r/runtime.before.sha256"
for name in success retry negative peer-retry call101-success exhaustion;do
 expected=0;[[ $name == call101-success || $name == exhaustion ]] && expected=42
 printf 'caller-control %s\n' "$name" > "$r/$name.command"
 set +e
 "$r/caller-control" "$name" > "$r/$name.stdout" 2> "$r/$name.stderr"
 code=$?;set -e;printf '%s\n' "$code" > "$r/$name.status"
 ((code==expected)) || exit 71
done
printf 'cases=6 CALLER_CONTROL_ONLY\n'
