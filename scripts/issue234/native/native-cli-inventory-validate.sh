#!/usr/bin/env bash
# Exact independently declared writer/artifact sets, not computed numeric expectations.
set -euo pipefail
file=$1;rank=$2;bad=$3;status=$4;phys=$5;name=$6
if ! test -f "$file" || test -L "$file";then exit 1;fi
actual=$(<"$file")
if ((rank!=0));then
 if [[ $name == output-* ]] && ((rank==bad));then
  if ((phys));then want='zvo_out_001.dat d';else want='zvo_out.dat d';fi
 else want='';fi
elif ((status==0));then
 if ((phys));then
  want=$'zvo_cisajs_001.dat f\nzvo_cisajscktalt_001.dat f\nzvo_cisajscktaltex_001.dat f\nzvo_out_001.dat f\nzvo_var_001.dat f'
 else want=$'zqp_opt.dat f\nzvo_out.dat f\nzvo_var.dat f';fi
elif [[ $name == output-* ]];then
 if ((phys));then want='zvo_out_001.dat d';else want='zvo_out.dat d';fi
else want='';fi
if [[ -z "$want" ]];then
 matches=0;if test ! -s "$file";then matches=1;fi
else
 matches=0;if cmp -s "$file" <(printf '%s\n' "$want");then matches=1;fi
fi
if ((matches==0));then printf 'Wrong independently declared inventory\nactual=%q\nexpected=%q\n' "$actual" "$want" >&2;exit 1;fi
printf 'VALIDATED inventory rank=%s case=%s\n' "$rank" "$name"
