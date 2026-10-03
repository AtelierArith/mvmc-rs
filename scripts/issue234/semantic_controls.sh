#!/usr/bin/env bash
# UNEXECUTED: external owner must wrap with timeout120s/kill-after10s.
# No Cargo, MPI, native models; six historical positives and48 generated negatives.
set -euo pipefail
checker=${1:?reviewed semantic awk};table=${2:?six historical capture table};out=${3:?exclusive result path}
mutation_verifier=${4:?reviewed independent intended-field verifier}
protocol_validator=${5:--} # Optional original protocol checker; - means schema-only.
[[ ! -e $out && -d ${out%/*} && -f $checker && -f $table && -f $mutation_verifier && ( $protocol_validator = - || -f $protocol_validator ) ]]
reserve=$((384*1024*1024));limit=$((16*1024*1024))
(( $(df -PB1 "${out%/*}" | awk 'NR==2{print $4}') >= reserve+limit ))
mkdir "$out"
finish(){
 prior=$?;trap - EXIT;set +e;post=0
 for name in inputs tools;do
  [[ ! -f $out/$name.before.sha256 ]] || sha256sum -c --quiet "$out/$name.before.sha256" > "$out/$name.post.log" 2>&1 || post=1
 done
 free=$(df -PB1 "$out" | awk 'NR==2{print $4}');bytes=$(du -sb "$out" | awk 'NR==1{print $1}')
 printf 'output=%s free=%s limit=%s reserve=%s\n' "$bytes" "$free" "$limit" "$reserve" > "$out/budget.post.txt"
 ((bytes<=limit&&free>=reserve)) || post=1
 printf 'prior=%s post=%s\n' "$prior" "$post" > "$out/terminal.txt"
 if ((prior||post));then exit 1;fi;exit 0
}
trap finish EXIT
sha256sum "${BASH_SOURCE[0]}" "$checker" "$table" "$mutation_verifier" > "$out/inputs.before.sha256"
[[ $protocol_validator = - ]] || sha256sum "$protocol_validator" >> "$out/inputs.before.sha256"
for name in bash awk sha256sum readlink df du stat mkdir timeout grep sort xargs od ldd cmp;do
 actual=$(readlink -f "$(type -P "$name")");sha256sum "$actual" >> "$out/tools.before.sha256"
 magic=$(od -An -tx1 -N4 "$actual" | awk '{print $1$2$3$4}')
 if [[ $magic = 7f454c46 ]];then
  ldd "$actual" >> "$out/tools.ldd.txt";printf 'ELF %s\n' "$actual" >> "$out/tools.classification.txt"
 else
  [[ $(od -An -tx1 -N2 "$actual" | awk '{print $1$2}') = 2321 ]]
  printf 'SCRIPT %s interpreter=bound-bash\n' "$actual" >> "$out/tools.classification.txt"
 fi
done
! grep -q 'not found' "$out/tools.ldd.txt"
awk '$2=="=>"&&substr($3,1,1)=="/"{print $3}substr($1,1,1)=="/"{print $1}' "$out/tools.ldd.txt" | sort -u | xargs sha256sum >> "$out/tools.before.sha256"
declare -A seen=();rows=0;total=0
while read -r rank world capture extra;do
 [[ -z ${extra:-} && $rank =~ ^(0|[1-9][0-9]*)$ && $world =~ ^(2|4)$ && $rank -lt $world && -f $capture ]]
 key="$world-$rank";[[ ! -v 'seen[$key]' ]];seen["$key"]=1
 sha256sum "$capture" >> "$out/inputs.before.sha256"
 total=$((total+$(stat -c '%s' "$capture")));rows=$((rows+1))
done < "$table"
[[ $rows = 6 ]];for key in 2-0 2-1 4-0 4-1 4-2 4-3;do [[ -v 'seen[$key]' ]];done
predicted=$((8*total+1024*1024));((predicted<=limit))
printf 'capture_bytes=%s predicted=%s limit=%s reserve=%s\n' "$total" "$predicted" "$limit" "$reserve" > "$out/budget.pre.txt"
guard(){ (( $(du -sb "$out" | awk 'NR==1{print $1}') <= limit && $(df -PB1 "$out" | awk 'NR==2{print $4}') >= reserve )); }
positive=0;negative=0
while read -r rank world capture;do
 if [[ $protocol_validator != - ]];then
  awk -v rank="$rank" -v world="$world" -f "$protocol_validator" "$capture" > "$out/positive-$world-$rank.protocol.log" 2>&1
 fi
 awk -f "$checker" "$capture" > "$out/positive-$world-$rank.log" 2>&1
 positive=$((positive+1));guard
 for variant in floatoverflow configtruncated cursoroverflow rawoverflow rawnegative rawshort countoverflow configinconsistent;do
  awk -v variant="$variant" '
   /^ISSUE178_SR_CHECK / && !changed {
    changed=1
    if(variant=="floatoverflow")edited=sub(/diagonal=[^ ]+/,"diagonal=1e999")
    if(variant=="configtruncated")edited=sub(/ }$/," ")
    if(variant=="cursoroverflow")edited=sub(/cursor=[0-9]+/,"cursor=625")
    if(variant=="rawoverflow")edited=sub(/raw=\[[0-9]+/,"raw=[4294967296")
    if(variant=="rawnegative")edited=sub(/raw=\[[0-9]+/,"raw=[-1")
    if(variant=="rawshort")edited=sub(/raw=\[[0-9]+, /,"raw=[")
    if(variant=="countoverflow")edited=sub(/count=[0-9]+/,"count=340282366920938463463374607431768211456")
    if(variant=="configinconsistent")edited=sub(/, ele_num: \[[01]/,", ele_num: [2")
   }
   {print}
   END{if(changed!=1||edited!=1)exit 2}
  ' "$capture" > "$out/$world-$rank-$variant.input"
  # A counted substitution alone is insufficient: require changed actual bytes.
  if cmp -s "$capture" "$out/$world-$rank-$variant.input";then exit 2;else comparison=$?;fi
  [[ $comparison = 1 ]] # I/O error2 is not a successful mutation.
  awk -v variant="$variant" -f "$mutation_verifier" "$capture" "$out/$world-$rank-$variant.input" > "$out/$world-$rank-$variant.mutation.log" 2>&1
  if awk -f "$checker" "$out/$world-$rank-$variant.input" > "$out/$world-$rank-$variant.log" 2>&1;then rejection=0;else rejection=$?;fi
  printf 'semantic_status=%s\n' "$rejection" > "$out/$world-$rank-$variant.status"
  [[ $rejection = 1 ]] # Syntax/I/O error2 or crash is not semantic rejection.
  case "$variant" in
   floatoverflow) diagnostic='nonfinite diagonal';;
   configtruncated) diagnostic='config termination';;
   cursoroverflow) diagnostic='cursor';;
   rawoverflow|rawnegative) diagnostic='raw word';;
   rawshort) diagnostic='raw length';;
   countoverflow) diagnostic='u128 count';;
   configinconsistent) diagnostic='occupancy map';;
  esac
  grep -Fx "INVALID_SEMANTIC $diagnostic" "$out/$world-$rank-$variant.log" > "$out/$world-$rank-$variant.rejection.log"
  negative=$((negative+1));guard
 done
done < "$table"
[[ $positive = 6 && $negative = 48 ]]
printf 'SEMANTIC_CONTROLS positives=%s negatives=%s protocolChecked=%s noModels=true\n' "$positive" "$negative" "$([[ $protocol_validator != - ]] && printf true || printf false)" > "$out/summary.txt"
