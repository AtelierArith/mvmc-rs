#!/usr/bin/env bash
# Developer-only SOURCE. Copies immutable offline input records; no oracle values.
set -euo pipefail
checkout=${1:?pinned checkout};binary=${2:?selected MPI CLI};cell=${3:?new cell path}
world=$4;width=$5;bad_rank=$6;trial=$7;name=$8
[[ $world =~ ^(2|4)$ && $width =~ ^(1|2)$ && $trial =~ ^(0|1)$ ]] || exit 2
[[ $bad_rank =~ ^(0|[1-9][0-9]*)$ ]] && ((bad_rank<world)) || exit 2
model=heisenberg_chain_real;phys=0;opt=0;initial=none
cg=0;nmp=-1;nsp=8;lanczos=0;diagnostic='';family='';status=1;overlay=0
case "$name" in
 parse-opt|parse-phys) family=parse;diagnostic='failed to read namelist.def' ;;
 fixed-missing|fixed-short|fixed-token) phys=1;family=$name
  case "$name" in fixed-missing)diagnostic='fixed parameter file not found';;
   fixed-short)diagnostic='too short';; fixed-token)diagnostic='non-numeric token';; esac ;;
 initial-opt) initial=path;family=initial;diagnostic='explicitly requested path' ;;
 overlay-opt|overlay-phys)family=overlay;overlay=1;diagnostic='UTF-8' ;;
 positive-overlay-opt|positive-overlay-phys)family=positive;overlay=1;status=0 ;;
 positive-opt|positive-phys|positive-phys-cg|positive-phys-ap|positive-phys-initial)
  family=positive;status=0
  case "$name" in positive-phys-cg)cg=1;;positive-phys-ap)nmp=-1;;positive-phys-initial)initial=path;;esac ;;
 output-opt|output-phys)family=output;diagnostic='Is a directory'
  if ((bad_rank!=0));then status=0;fi ;;
 grouped-cg)cg=1;family=grouped;diagnostic='NSplitSize > 1 with SR-CG' ;;
 grouped-phys-fsz)phys=1;model=heisenberg_chain_fsz;nmp=1;nsp=1;family=grouped;diagnostic='FSZ / general-orbital' ;;
 grouped-opt-fsz-spin)model=heisenberg_chain_fsz;nmp=1;nsp=2;family=grouped;diagnostic='FSZ standard-projection NQPFull' ;;
 grouped-opt-fsz-trans)model=heisenberg_chain_fsz;nmp=2;nsp=1;family=grouped;diagnostic='FSZ standard-projection NQPFull' ;;
 grouped-opt-fsz-ap)model=heisenberg_chain_fsz;nmp=-2;nsp=1;family=grouped;diagnostic='FSZ standard-projection NQPFull' ;;
 grouped-opt-opttrans|grouped-phys-opttrans)model=hubbard_chain_dh_opttrans;opt=1;family=grouped;diagnostic='NQPOptTrans > 1 / OptTrans' ;;
 grouped-phys-lanczos1|grouped-phys-lanczos2)phys=1;family=grouped;lanczos=${name: -1};diagnostic='NSplitSize > 1 with NLanczosMode' ;;
 opt-lanczos1|opt-lanczos2)family=global;lanczos=${name: -1};diagnostic='NLanczosMode > 0 is not supported for parameter optimization' ;;
 *)printf 'Unknown declared CLI case %s\n' "$name" >&2;exit 2 ;;
esac
case "$name" in *-phys|positive-phys*|grouped-phys*)phys=1;;esac
if [[ $family == grouped ]] && ((width!=2));then exit 2;fi
if [[ $family == global ]] && ((width!=1));then exit 2;fi
if [[ $name != positive-phys-ap ]];then nmp=${nmp/-1/1};fi
source="$checkout/tests/fixtures/physcal_181/$model"
test -d "$source/inputs" && test -f "$source/zqp_opt.dat"
mkdir "$cell"
printf 'case=%s world=%s width=%s bad_rank=%s trial=%s model=%s phys=%s opt_trans=%s initial=%s\n' "$name" "$world" "$width" "$bad_rank" "$trial" "$model" "$phys" "$opt" "$initial" > "$cell/settings.txt"
printf '%s\n' "$phys" > "$cell/phys";printf '%s\n' "$opt" > "$cell/opt-trans";printf '%s\n' "$initial" > "$cell/initial-kind"
for ((rank=0;rank<world;rank++));do
 dir="$cell/rank-$rank";mkdir "$dir" "$dir/inputs"
 cp "$source/inputs/"* "$dir/inputs/"
 cp "$source/zqp_opt.dat" "$dir/fixed.dat"
 if ((phys==0));then awk '$1!="TwoBodyGEx"' "$dir/inputs/namelist.def" > "$dir/inputs/namelist.tmp";mv "$dir/inputs/namelist.tmp" "$dir/inputs/namelist.def";fi
 awk -v phys="$phys" -v width="$width" -v cg="$cg" -v nmp="$nmp" -v nsp="$nsp" -v lanczos="$lanczos" '
  $1!~/^(NVMCCalMode|NSplitSize|NSRCG|NMPTrans|NSPGaussLeg|NLanczosMode|NSROptItrStep|NSROptItrSmp|NVMCWarmUp|NVMCInterval|NVMCSample|NDataQtySmp|NDataIdxStart|DSROptStaDel)$/ {print}
  END{print "NVMCCalMode "phys;print "NSplitSize "width;print "NSRCG "cg;print "NMPTrans "nmp;print "NSPGaussLeg "nsp;print "NLanczosMode "lanczos;print "NSROptItrStep 1\nNSROptItrSmp 1\nNVMCWarmUp 1\nNVMCInterval 1\nNVMCSample 8\nNDataQtySmp 1\nNDataIdxStart 1\nDSROptStaDel 0.02"}' \
  "$dir/inputs/modpara.def" > "$dir/inputs/modpara.tmp"
 mv "$dir/inputs/modpara.tmp" "$dir/inputs/modpara.def"
 selected="$dir/inputs/namelist.def";fixed="$dir/fixed.dat"
 if [[ $initial == path ]] && ! { [[ $family == initial ]] && ((rank==bad_rank)); };then cp "$source/zqp_opt.dat" "$dir/initial.def";fi
 if ((overlay));then
  printf '\nInGutzwiller overlay.def\n' >> "$dir/inputs/namelist.def"
  # C ReadInputParameters reads all five headers, checks NGutzwillerIdx,
  # then consumes one idx/real/imag record (readdef.c1207..1234).
  printf '=============================================\nNGutzwillerIdx 1\nComplexType 0\n=============================================\n=============================================\n' > "$dir/inputs/overlay.def"
  if [[ $family == overlay ]] && ((rank==bad_rank));then
   printf '0 \377 0.0\n' >> "$dir/inputs/overlay.def"
  else printf '0 0.125 0.0\n' >> "$dir/inputs/overlay.def";fi
 fi
 if ((rank==bad_rank));then
  case "$family" in
   parse) : > "$dir/select-missing-namelist";selected="$dir/missing.def" ;;
   fixed-missing) : > "$dir/select-missing-fixed";fixed="$dir/missing-fixed.dat" ;;
   fixed-short)printf '0 0 0 0 0 0\n' > "$dir/fixed.dat" ;;
   fixed-token)printf '0 0 invalid 0 0 0\n' > "$dir/fixed.dat" ;;
   output)mkdir "$dir/output"
    if ((phys));then mkdir "$dir/output/zvo_out_001.dat";else mkdir "$dir/output/zvo_out.dat";fi ;;
  esac
 fi
 local_diag="$diagnostic"
 if ((rank!=bad_rank)) && [[ $family != grouped && $family != global && $family != positive ]];then
  case "$family" in
   parse)local_diag='CLI parse/validation failed on another MPI rank';;
   overlay)if ((phys));then local_diag='PhysCal parse/load/validation failed on another MPI rank';else local_diag='optimization parameter load failed on another MPI rank';fi ;;
   fixed-missing)local_diag='CLI PhysCal parse/load/validation failed on another MPI rank';;
   fixed-short|fixed-token)local_diag='PhysCal parse/load/validation failed on another MPI rank';;
   initial)local_diag='optimization parameter load failed on another MPI rank';;
   output)if ((phys));then local_diag='output sample 0 failed on another MPI rank';else local_diag='output step 0 failed on another MPI rank';fi ;;
  esac
 fi
 if ((status==0));then local_diag='';fi
 expected_output=false
 if [[ $family == positive || $family == output ]];then
  # High-level OPT creates its requested directory on every rank; only root writes.
  if ((phys==0 || rank==0 || rank==bad_rank && status==0));then expected_output=true;fi
 fi
 printf '%s\n' "$status" > "$dir/expected-status";printf '%s\n' "$expected_output" > "$dir/expected-output"
 printf '%s\n' "$local_diag" > "$dir/expected-diagnostic"
 args=("$selected" --seed 11272 --nsteps 1 --nsmp 1 --initial-def "$initial" --out-dir "$dir/output")
 if [[ $initial == path ]];then args=("$selected" --seed 11272 --nsteps 1 --nsmp 1 --initial-def "$dir/initial.def" --out-dir "$dir/output");fi
 if ((phys));then args+=(--physcal "$fixed");fi
 if ((opt));then args+=(--opt-trans);fi
 printf '%q ' "$binary" "${args[@]}" > "$dir/expected-argv.txt"
done
