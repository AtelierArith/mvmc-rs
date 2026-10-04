#!/usr/bin/env bash
# Opt-in Linux/MPICH execution; never invoked by ordinary CI or Cargo.
set -euo pipefail
repo=${1:?checkout};out=${2:?exclusive evidence directory}
cd "$repo"
[[ $(git rev-parse HEAD) = "${EXPECTED_HEAD:?dispatch checkout SHA}" && ! -e $out ]]
[[ -n ${CARGO_TARGET_DIR:?exclusive target} && -d ${TMPDIR:?exclusive native output} ]]
[[ -z ${LD_PRELOAD-} && -z ${LD_AUDIT-} && -z ${BASH_ENV-} ]]
mkdir "$out"
mpi_prefix=${MVMC_ISSUE234_MPI_PREFIX:?job-local matched MPI provider}
mpi_receipt=${MVMC_ISSUE234_MPI_RECEIPT:?startup worlds2/4 receipt}
grep -Fx 'prior=0 post=0' "$mpi_receipt/terminal.txt"
grep -Fx 'MPI_PROVIDER_READY version=4.2.0 pmi=pmi1 worlds=2,4' "$mpi_receipt/ready.txt"
[[ $(readlink -f "$(command -v mpicc)") = "$mpi_prefix/bin/mpicc" && $(readlink -f "$(command -v mpiexec)") = "$mpi_prefix/bin/mpiexec.hydra" ]]
here=$(cd "$(dirname "$0")" && pwd)
source_inventory(){
 find "$repo" -name .git -prune -o -type f -print0 | sort -z | xargs -0 sha256sum
}
symlink_inventory(){
 find "$repo" -name .git -prune -o -type l -printf '%p -> %l\n' | sort
}
finish(){
 prior=$?;trap - EXIT;set +e;post=0
 for manifest in source tools tool-providers mpi-headers mpi-provider runtime selected;do
  [[ ! -f $out/$manifest.sha256 ]] || sha256sum -c --quiet "$out/$manifest.sha256" > "$out/$manifest.post.log" 2>&1 || post=1
 done
 git status --porcelain --untracked-files=all > "$out/status.after.txt"
 cmp "$out/status.before.txt" "$out/status.after.txt" > "$out/status.post.log" 2>&1 || post=1
 if [[ -f $out/full-source.before.sha256 ]];then
  source_inventory > "$out/full-source.after.sha256" || post=1
  cmp "$out/full-source.before.sha256" "$out/full-source.after.sha256" > "$out/full-source.post.log" 2>&1 || post=1
  symlink_inventory > "$out/symlinks.after.txt" || post=1
  cmp "$out/symlinks.before.txt" "$out/symlinks.after.txt" > "$out/symlinks.post.log" 2>&1 || post=1
 fi
 printf 'prior=%s post=%s head=%s\n' "$prior" "$post" "$EXPECTED_HEAD" > "$out/terminal.txt"
 ((prior==0 && post==0))
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
unset MVMC_C_TIMER MVMC_TIMER MVMC_NSTEPS MVMC_CALHAM1_DIAG MVMC_SLATER_DIAG MVMC_MAINCAL_DIAG MVMC_WEIGHTAVG_DIAG
git status --porcelain --untracked-files=all > "$out/status.before.txt"
test ! -s "$out/status.before.txt"
source_inventory > "$out/full-source.before.sha256"
symlink_inventory > "$out/symlinks.before.txt"
git submodule status --recursive > "$out/submodules.txt"
! grep -E '^[+-U]' "$out/submodules.txt"
git ls-files --recurse-submodules -z | xargs -0 sha256sum > "$out/source.sha256"
git ls-files --recurse-submodules -z | while IFS= read -r -d '' path;do
 if [[ -L $path ]];then printf '%s -> %s\n' "$path" "$(readlink "$path")";fi
done > "$out/source-symlinks.txt"
for name in bash cargo rustc cargo-nextest mpicc mpiexec hydra_pmi_proxy mpichversion dpkg-query cc ld ar pkg-config jq awk sed rg timeout sha256sum find sort xargs cmp ldd readlink od cat;do
 path=$(command -v "$name");sha256sum "$(readlink -f "$path")" >> "$out/tools.sha256"
done
for wrapper in "${RUSTC_WRAPPER-}" "${RUSTC_WORKSPACE_WRAPPER-}";do
 [[ -z $wrapper ]] || sha256sum "$(readlink -f "$(command -v "$wrapper")")" >> "$out/tools.sha256"
done
sha256sum "$(rustup which rustc)" "$(rustup which cargo)" >> "$out/tools.sha256"
for config in "$repo/.cargo/config" "$repo/.cargo/config.toml" "${CARGO_HOME:-$HOME/.cargo}/config" "${CARGO_HOME:-$HOME/.cargo}/config.toml";do
 [[ ! -f $config ]] || sha256sum "$config" >> "$out/tools.sha256"
done
# Resolve providers BEFORE compilation. Non-ELF launch scripts are identified
# explicitly; their bytes are still pinned, but ldd is not invoked on them.
while read -r digest executable;do
 magic=$(od -An -tx1 -N4 "$executable" | awk '{print $1$2$3$4}')
 if [[ $magic = 7f454c46 ]];then
  if ldd "$executable" > "$out/ldd-current.txt" 2>&1;then :
  else grep -q 'statically linked\|not a dynamic executable' "$out/ldd-current.txt" || exit 2;fi
  cat "$out/ldd-current.txt" >> "$out/tools.ldd.txt"
 else printf 'NON_ELF %s\n' "$executable" >> "$out/tools.classification.txt";fi
done < "$out/tools.sha256"
! grep -q 'not found' "$out/tools.ldd.txt"
awk '$2=="=>"&&substr($3,1,1)=="/"{print $3}substr($1,1,1)=="/"{print $1}' "$out/tools.ldd.txt" | sort -u | xargs -r sha256sum > "$out/tool-providers.sha256"
test -s "$out/tool-providers.sha256"
{ uname -a;rustc -Vv;cargo -V;cargo nextest --version;mpicc -show;mpiexec -version;pkg-config --modversion openblas; } > "$out/environment.txt" 2>&1
mpichversion > "$out/mpichversion.txt" 2>&1
dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' mpich libmpich12 libmpich-dev libpmix2 > "$out/mpi-packages.txt"
for name in PMI_RANK PMI_SIZE PMI_FD PMI_VERSION PMI_SUBVERSION PMIX_RANK MPIR_CVAR_PMI_VERSION MPICH_PMI_VERSION MPIR_PARAM_PMI_VERSION;do
 if [[ ! -v $name ]];then value=UNSET
 elif [[ ${!name} =~ ^[0-9]+$ || ${!name} = x ]];then value=${!name}
 else value=OTHER_REDACTED;fi
 printf '%s=%s\n' "$name" "$value"
done > "$out/launcher-pmi-env.txt"
for name in PMI_PORT PMIX_NAMESPACE PMIX_SERVER_URI PMIX_SERVER_URI2 PMIX_SERVER_URI21;do
 if [[ -v $name ]];then present=true;else present=false;fi
 printf '%s present=%s\n' "$name" "$present"
done >> "$out/launcher-pmi-env.txt"
{ cc --version;ld --version;rustc --print sysroot;pkg-config --cflags --libs openblas; } > "$out/compiler-backend.txt" 2>&1
printf 'RUSTC_WRAPPER=%s\nRUSTC_WORKSPACE_WRAPPER=%s\nCARGO_TARGET_DIR=%s\n' "${RUSTC_WRAPPER-}" "${RUSTC_WORKSPACE_WRAPPER-}" "$CARGO_TARGET_DIR" > "$out/compiler-settings.txt"
mpiexec -help > "$out/hydra-help.txt" 2>&1
for flag in -disable-auto-cleanup -outfile-pattern -errfile-pattern;do rg -q -- "$flag" "$out/hydra-help.txt";done
sha256sum -c --quiet "$mpi_receipt/provider.sha256"
cp "$mpi_receipt/provider.sha256" "$out/mpi-provider.sha256"
find "$mpi_prefix/include" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/mpi-headers.sha256"
test -s "$out/mpi-headers.sha256"
timeout -k 10s 120s bash scripts/issue234/semantic_controls_wrapper.sh "$out/schema-controls" --schema-fixtures > "$out/schema-controls.log" 2>&1
timeout -k 10s 60s bash "$here/foreground-controls.sh" "$TMPDIR/foreground-controls" > "$out/foreground-controls.log" 2>&1
test "$(grep -c '^FOREGROUND .* PASS$' "$out/foreground-controls.log")" = 16
for model in heisenberg_chain_real heisenberg_chain_fsz hubbard_chain_dh_opttrans;do
 test -f "tests/fixtures/physcal_181/$model/inputs/namelist.def"
 test -f "tests/fixtures/physcal_181/$model/zqp_opt.dat"
done
timeout -k 30s 1200s cargo nextest list --locked --cargo-profile test-fast -p mvmc-core --features mpi \
 --test mpi_issue178_sr_failure --test mpi_issue234_summary --message-format json > "$out/inventory.json" 2> "$out/build.stderr"
serial=finite_negative_regularizer_fails_actual_public_sr_after_healthy_control
timeout -k 10s 120s cargo metadata --locked --no-deps --format-version 1 > "$out/cargo-metadata.json" 2> "$out/cargo-metadata.stderr"
core_id=$(jq -er --arg manifest "$repo/crates/mvmc-core/Cargo.toml" '.packages|map(select(.name=="mvmc-core" and .manifest_path==$manifest))|if length==1 then .[0].id else error("wrong source package") end' "$out/cargo-metadata.json")
jq -e --arg gate "$serial" --arg package "$core_id" '."rust-suites"|to_entries|map(select(.value["binary-name"]=="mpi_issue178_sr_failure" and .value["package-id"]==$package))|if length==1 then .[0].value.testcases[$gate].ignored==false else false end' "$out/inventory.json" > "$out/serial.inventory.log"
jq --arg gate "$serial" '."rust-suites"|to_entries|map(select(.value["binary-name"]=="mpi_issue178_sr_failure"))|map({binary_id:.key,package_id:.value["package-id"],binary_path:.value["binary-path"],gate:$gate,ignored:.value.testcases[$gate].ignored})' "$out/inventory.json" > "$out/serial-association.json"
sha256sum "$repo/crates/mvmc-core/Cargo.toml" "$repo/crates/mvmc-core/tests/mpi_issue178_sr_failure.rs" "$repo/crates/mvmc-core/tests/mpi_issue234_summary.rs" > "$out/gate-sources.sha256"
timeout -k 30s 300s cargo nextest run --locked --cargo-profile test-fast -p mvmc-core --features mpi --test mpi_issue178_sr_failure \
 -E "test(=$serial)" --no-tests fail --no-fail-fast --retries 0 > "$out/serial.log" 2>&1
timeout -k 30s 1200s cargo build --locked --profile test-fast -p mvmc-cli --features mpi > "$out/cli-build.log" 2>&1
mkdir "$out/selected"
cp "$CARGO_TARGET_DIR/test-fast/mvmc" "$out/selected/mvmc"
for suite in mpi_issue178_sr_failure mpi_issue234_summary;do
 if [[ $suite = mpi_issue178_sr_failure ]];then gate=actual_rank_local_nonpd_sr_restores_successful_peers_before_callback;else gate=public_rank_local_summary_and_root_readback_failure;fi
 path=$(jq -er --arg suite "$suite" --arg gate "$gate" --arg package "$core_id" '."rust-suites"|to_entries|map(select(.value["binary-name"]==$suite and .value["package-id"]==$package and .value.testcases[$gate].ignored==true))|if length==1 then .[0].value["binary-path"] else error("missing or ambiguous native gate") end' "$out/inventory.json")
 [[ $(readlink -f "$path") = "$(readlink -f "$CARGO_TARGET_DIR")/"* ]]
 jq --arg suite "$suite" --arg gate "$gate" '."rust-suites"|to_entries|map(select(.value["binary-name"]==$suite))|map({binary_id:.key,package_id:.value["package-id"],binary_path:.value["binary-path"],gate:$gate,ignored:.value.testcases[$gate].ignored})' "$out/inventory.json" > "$out/$suite.association.json"
 cp "$path" "$out/selected/$suite"
done
sha256sum "$out"/selected/* > "$out/selected.sha256"
for binary in "$out"/selected/* "$(command -v mpiexec)" "$(command -v hydra_pmi_proxy)";do
 ldd "$binary" >> "$out/ldd.txt"
done
! grep -q 'not found' "$out/ldd.txt"
awk -v prefix="$mpi_prefix/lib/" '$1~/^libmpi[.]so/{if(index($3,prefix)!=1)bad=1;n++}END{exit(bad||n!=3)}' "$out/ldd.txt"
awk '/=> \/|^\s*\//{for(i=1;i<=NF;i++)if($i~/^\//)print $i}' "$out/ldd.txt" | sort -u | xargs -r sha256sum > "$out/runtime.sha256"
test -s "$out/runtime.sha256"
for suite in mpi_issue178_sr_failure mpi_issue234_summary;do
 if [[ $suite = mpi_issue178_sr_failure ]];then gate=actual_rank_local_nonpd_sr_restores_successful_peers_before_callback;else gate=public_rank_local_summary_and_root_readback_failure;fi
 for world in 2 4;do
  cell="$out/$suite-world-$world";mkdir "$cell"
  for ((rank=0;rank<world;rank++));do (set -o noclobber; : > "$cell/rank-$rank.stderr");done
  export MPI_ISSUE178_SR_OUTPUT="$cell/output" MPI_ISSUE234_SUMMARY_OUTPUT="$cell/output"
  set +e
  timeout -k 10s 120s mpiexec -disable-auto-cleanup -outfile-pattern "$cell/rank-%r.stdout" -errfile-pattern "$cell/rank-%r.stderr" -n "$world" \
   "$out/selected/$suite" --ignored --exact "$gate" --nocapture > "$cell/launcher.stdout" 2> "$cell/launcher.stderr"
  status=$?;set -e;printf '%s\n' "$status" > "$cell/native.status";test "$status" = 0
  for ((rank=0;rank<world;rank++));do
   test -s "$cell/rank-$rank.stdout"
   if [[ $suite = mpi_issue178_sr_failure ]];then
    sed 's/^test actual_rank_local_nonpd_sr_restores_successful_peers_before_callback \.\.\. //' "$cell/rank-$rank.stdout" > "$cell/rank-$rank.normalized"
    awk -v rank="$rank" -v world="$world" -f "$here/validate-sr.awk" "$cell/rank-$rank.normalized" > "$cell/rank-$rank.protocol.log" 2>&1
    awk -v rank="$rank" -v world="$world" -f "$repo/scripts/issue234/validate_sr_semantic.awk" "$cell/rank-$rank.normalized" > "$cell/rank-$rank.semantic.log" 2>&1
   else
    awk -v rank="$rank" -v world="$world" -f "$here/validate-summary.awk" "$cell/rank-$rank.stdout" "$cell/rank-$rank.stderr" > "$cell/rank-$rank.protocol.log" 2>&1
   fi
  done
 done
done
sha256sum "$out/selected/mvmc" "$here"/* > "$out/cli-pins.sha256"
bash "$here/full-matrix-SOURCE.sh" "$repo" "$out/selected/mvmc" "$out/cli-pins.sha256" > "$out/cli260.log" 2>&1
test "$(grep -c '^CELL_VALIDATED ' "$out/cli260.log")" = 260
printf 'CURRENT_CHECKOUT_NATIVE_COMPLETE head=%s srRanks=6 summaryRanks=6 cliCells=260\n' "$EXPECTED_HEAD" > "$out/complete.txt"
