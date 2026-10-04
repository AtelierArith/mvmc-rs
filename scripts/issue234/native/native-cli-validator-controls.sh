#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(mktemp -d "${TMPDIR:?exclusive artifact parent}/mvmc-178-cli-validator.XXXXXX")
trap 'printf "%s\n" "$?" > "$root/status"' EXIT
positive() {
 printf '%s\n' \
  'ISSUE178_CLI_START rank=0 world=2 trial=0 width=1 case=parse' \
  'error: failed to read namelist.def' \
  'ISSUE178_CLI_RETURN rank=0 world=2 trial=0 width=1 case=parse status=1 output_exists=false' \
  'ISSUE178_CLI_DONE rank=0 world=2 trial=0 width=1 case=parse status=1'
}
validate() {
 awk -v rank=0 -v world=2 -v trial=0 -v width=1 -v case_name=parse \
  -v expected_status=1 -v expected_output=false -v diagnostic='failed to read namelist.def' \
  -f "$here/native-cli-validate.awk" "$1"
}
positive > "$root/positive.txt"
validate "$root/positive.txt" > "$root/positive.log" 2>&1
sed '/^error:/d;s/status=1/status=0/g;s/output_exists=false/output_exists=true/' "$root/positive.txt" > "$root/success.txt"
awk -v rank=0 -v world=2 -v trial=0 -v width=1 -v case_name=parse \
 -v expected_status=0 -v expected_output=true -v diagnostic='' \
 -f "$here/native-cli-validate.awk" "$root/success.txt" > "$root/success.log" 2>&1
for variant in missing duplicate-key extra-field rank-leading-zero status-suffix status-zero-suffix wrong-rank duplicate-record interleave unknown-marker wrong-width missing-diagnostic; do
 case "$variant" in
 missing) sed '/ISSUE178_CLI_RETURN/d' "$root/positive.txt" ;;
 duplicate-key) sed '/ISSUE178_CLI_RETURN/s/$/ rank=0/' "$root/positive.txt" ;;
 extra-field) sed '/ISSUE178_CLI_RETURN/s/$/ surprise=true/' "$root/positive.txt" ;;
 rank-leading-zero) sed 's/rank=0/rank=00/g' "$root/positive.txt" ;;
 status-suffix) sed 's/status=1/status=1x/g' "$root/positive.txt" ;;
 status-zero-suffix) sed 's/status=1/status=0x/g' "$root/positive.txt" ;;
 wrong-rank) sed 's/rank=0/rank=1/g' "$root/positive.txt" ;;
 duplicate-record) sed '/ISSUE178_CLI_RETURN/p' "$root/positive.txt" ;;
 interleave) sed '/ISSUE178_CLI_RETURN/s/$/ ISSUE178_CLI_DONE rank=0/' "$root/positive.txt" ;;
 unknown-marker) sed 's/ISSUE178_CLI_DONE/ISSUE178_CLI_OTHER/' "$root/positive.txt" ;;
 wrong-width) sed 's/width=1/width=2/g' "$root/positive.txt" ;;
 missing-diagnostic) sed '/^error:/d' "$root/positive.txt" ;;
 esac > "$root/$variant.txt"
 if validate "$root/$variant.txt" > "$root/$variant.log" 2>&1; then
  printf 'FAIL accepted negative %s\n' "$variant" >&2; exit 1
 fi
done
printf 'CONTROLS positive=2 negative=12 root=%s\n' "$root"
