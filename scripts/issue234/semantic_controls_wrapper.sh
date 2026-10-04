#!/usr/bin/env bash
# Optional developer controls. Never invoked by Cargo/build.rs/Rust tests.
set -euo pipefail
out=${1:?exclusive output root};table_arg=${2:?capture table or --schema-fixtures}
stage=${3:--};protocol=${4:--}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ ! -e $out && -d ${out%/*} ]]
(( $(df -PB1 "${out%/*}" | awk 'NR==2{print $4}') >= (384+16)*1024*1024 ))
mkdir "$out"
finish(){
 prior=$?;trap - EXIT;set +e;post=0
 for name in source tools inputs stage;do
  [[ ! -f $out/$name.before.sha256 ]] || sha256sum -c --quiet "$out/$name.before.sha256" > "$out/$name.post.log" 2>&1 || post=1
 done
 bytes=$(du -sb "$out" | awk 'NR==1{print $1}');free=$(df -PB1 "$out" | awk 'NR==2{print $4}')
 printf 'output=%s free=%s limit=16777216 reserve=402653184\n' "$bytes" "$free" > "$out/budget.post.txt"
 ((bytes<=16*1024*1024&&free>=384*1024*1024)) || post=1
 printf 'prior=%s post=%s\n' "$prior" "$post" > "$out/terminal.txt"
 if ((prior||post));then exit 1;fi;exit 0
}
trap finish EXIT
find "$here" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/source.before.sha256"
for name in bash env timeout df du awk sha256sum readlink find sort xargs grep mkdir od stat cmp;do
 actual=$(readlink -f "$(type -P "$name")");sha256sum "$actual" >> "$out/tools.before.sha256"
 magic=$(od -An -tx1 -N4 "$actual" | awk '{print $1$2$3$4}')
 [[ $magic = 7f454c46 ]] # This wrapper's required command tools must be ELF.
 ldd "$actual" >> "$out/tools.ldd.txt"
done
sha256sum "$(readlink -f "$(type -P ldd)")" >> "$out/tools.before.sha256"
! grep -q 'not found' "$out/tools.ldd.txt"
awk '$2=="=>"&&substr($3,1,1)=="/"{print $3}substr($1,1,1)=="/"{print $1}' "$out/tools.ldd.txt" | sort -u | xargs sha256sum >> "$out/tools.before.sha256"
if [[ $stage != - ]];then
 grep -Eq '^prior=0 post=0 main=[0-9a-f]{40}$' "$stage/terminal.txt"
 find "$stage" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/stage.before.sha256"
fi
if [[ $table_arg = --schema-fixtures ]];then
 [[ $protocol = - ]] # Synthetic schema fixtures are not real protocol receipts.
 mkdir "$out/schema-fixtures";table="$out/schema-fixtures/table.txt"
 for world in 2 4;do for ((rank=0;rank<world;rank++));do
  capture="$out/schema-fixtures/world-$world-rank-$rank.txt"
  awk -v rank="$rank" 'BEGIN {
   raw="0";for(i=1;i<624;i++)raw=raw", 0"
   # Independent literal occupancy: up electron at0, down electron at1.
   cfg="ElectronConfiguration { n_sample: 1, n_size: 2, n_site2: 4, n_proj: 1, ele_idx: [0, 1], ele_cfg: [0, -1, -1, 0], ele_num: [1, 0, 0, 1], ele_proj_cnt: [0], ele_spn: [], tmp_ele_idx: [], tmp_ele_cfg: [], tmp_ele_num: [], tmp_ele_proj_cnt: [], tmp_ele_spn: [], burn_ele_idx: [], burn_ele_cfg: [], burn_ele_num: [], burn_ele_proj_cnt: [], burn_ele_spn: [], counter: [] }"
   for(j=0;j<10;j++)print "ISSUE178_SR_CHECK rank="rank" diagonal=0.25 raw=["raw"] cursor=0 count=0 next624=["raw"] config="cfg
  }' > "$capture"
  printf '%s %s %s\n' "$rank" "$world" "$capture" >> "$table"
 done;done
 printf '%s\n' 'SCHEMA_ONLY literals; zeros are not an SFMT seed/stream oracle; no sampler/native claim.' > "$out/fixture-origin.txt"
else
 table=$table_arg
fi
sha256sum "$table" > "$out/inputs.before.sha256"
while read -r rank world capture extra;do
 [[ -z ${extra:-} && -f $capture ]];sha256sum "$capture" >> "$out/inputs.before.sha256"
done < "$table"
if [[ $protocol != - ]];then
 [[ $(sha256sum "$protocol" | awk '{print $1}') = 1c90f4a998b53071f7a912cdc7b410b98d5082d77d1c4c0c1806aec4b109b6b8 ]]
 sha256sum "$protocol" >> "$out/inputs.before.sha256"
fi
printf '%q ' timeout --signal=TERM --kill-after=10s 120s bash "$here/semantic_controls.sh" "$here/validate_sr_semantic.awk" "$table" "$out/results" "$here/verify_negative_mutation.awk" "$protocol" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
timeout --signal=TERM --kill-after=10s 120s bash "$here/semantic_controls.sh" "$here/validate_sr_semantic.awk" "$table" "$out/results" "$here/verify_negative_mutation.awk" "$protocol" > "$out/run.stdout" 2> "$out/run.stderr"
grep -Fx 'prior=0 post=0' "$out/results/terminal.txt"
grep -Eq '^SEMANTIC_CONTROLS positives=6 negatives=48 protocolChecked=(true|false) noModels=true$' "$out/results/summary.txt"
