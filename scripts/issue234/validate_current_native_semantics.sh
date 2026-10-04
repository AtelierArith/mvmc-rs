#!/usr/bin/env bash
# Optional post-native audit. Never launches Cargo/MPI or manufactures captures.
set -euo pipefail
main=${1:?expected full main commit};build=${2:?successful build receipt}
native=${3:?fresh native SR root};pins=${4:?actual launch pin manifest}
protocol=${5:?unchanged strict protocol awk};out=${6:?exclusive audit receipt}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $main =~ ^[0-9a-f]{40}$ && ! -e $out && -d ${out%/*} && -f $protocol ]]
[[ $(sha256sum "$protocol" | awk '{print $1}') = 1c90f4a998b53071f7a912cdc7b410b98d5082d77d1c4c0c1806aec4b109b6b8 ]]
grep -Fx "BUILD_RESUME_COMPLETE main=$main nativePendingParentReview" "$build/build-complete.txt"
grep -Eq "^prior=0 post=0 main=$main stage=/" "$build/terminal.txt"
[[ $(cat "$native/aggregate.status") = 0 && $(cat "$native/launcher.status") = 0 ]]
(( $(df -PB1 "${out%/*}" | awk 'NR==2{print $4}') >= (384+4)*1024*1024 ))
mkdir "$out"
finish(){
 prior=$?;trap - EXIT;set +e;post=0
 for name in inputs tools;do
  [[ ! -f $out/$name.before.sha256 ]] || sha256sum -c --quiet "$out/$name.before.sha256" > "$out/$name.post.log" 2>&1 || post=1
 done
 (( $(du -sb "$out" | awk 'NR==1{print $1}')<=4*1024*1024 && $(df -PB1 "$out" | awk 'NR==2{print $4}')>=384*1024*1024 )) || post=1
 printf 'prior=%s post=%s main=%s\n' "$prior" "$post" "$main" > "$out/terminal.txt"
 if ((prior||post));then exit 1;fi;exit 0
}
trap finish EXIT
sha256sum -c --quiet "$build/selected.before.sha256" > "$out/selected.pre.log"
sha256sum -c --quiet "$pins" > "$out/native-pins.pre.log"
selected_row=$(awk '$2~/\/mpi_issue178_sr_failure$/{print}' "$build/selected.before.sha256")
[[ $(printf '%s\n' "$selected_row" | wc -l) = 1 && -n $selected_row ]]
read -r selected_sha selected_path <<< "$selected_row"
awk -v digest="$selected_sha" -v path="$selected_path" '$1==digest&&$2==path{n++}END{exit(n!=1)}' "$pins"
# New launcher must emit these BEFORE running the bound executable. A complete
# old capture with a newly invented main label is not an admissible association.
grep -Fx "main=$main selected_binary=$selected_path selected_sha=$selected_sha" "$native/association.txt"
[[ $(cat "$native/selected.before.sha256") = "$selected_row" ]]
sha256sum -c --quiet "$native/selected.before.sha256" > "$out/native-selected.pre.log"
sha256sum "${BASH_SOURCE[0]}" "$here/validate_sr_semantic.awk" "$protocol" "$pins" > "$out/inputs.before.sha256"
find "$build" "$native" -type f -print0 | sort -z | xargs -0 sha256sum >> "$out/inputs.before.sha256"
for name in bash env awk sha256sum readlink find sort xargs grep cat wc df du mkdir od ldd;do
 actual=$(readlink -f "$(type -P "$name")");sha256sum "$actual" >> "$out/tools.before.sha256"
 magic=$(od -An -tx1 -N4 "$actual" | awk '{print $1$2$3$4}')
 if [[ $magic = 7f454c46 ]];then ldd "$actual" >> "$out/tools.ldd.txt"
 else [[ $(od -An -tx1 -N2 "$actual" | awk '{print $1$2}') = 2321 ]];fi
done
! grep -q 'not found' "$out/tools.ldd.txt"
awk '$2=="=>"&&substr($3,1,1)=="/"{print $3}substr($1,1,1)=="/"{print $1}' "$out/tools.ldd.txt" | sort -u | xargs sha256sum >> "$out/tools.before.sha256"
checked=0;capture_bytes=0
for world in 2 4;do
 [[ $(cat "$native/world-$world/native.status") = 0 ]]
 for ((rank=0;rank<world;rank++));do
  capture="$native/world-$world/rank-$rank.normalized";[[ -s $capture ]]
  capture_bytes=$((capture_bytes+$(wc -c < "$capture")))
  ((capture_bytes<=4*1024*1024))
  awk -v rank="$rank" -v world="$world" -f "$protocol" "$capture" > "$out/world-$world-rank-$rank.protocol.log" 2>&1
  awk -f "$here/validate_sr_semantic.awk" "$capture" > "$out/world-$world-rank-$rank.semantic.log" 2>&1
  checked=$((checked+1))
 done
done
[[ $checked = 6 ]]
printf 'CURRENT_NATIVE_SEMANTIC main=%s rank_captures=6 scratchContentsNotInterpreted=true\n' "$main" > "$out/summary.txt"
