#!/usr/bin/env bash
# Job-local MPI dependency, matching the repository Dev Container's PMI recipe.
# Never invoked by Cargo or normal Rust tests; no numerical C oracle.
set -euo pipefail
repo=${1:?checkout};prefix=${2:?exclusive job-local MPI prefix};proof=${3:?exclusive receipt}
[[ ! -e $prefix && ! -e $proof && -d ${prefix%/*} && -d ${proof%/*} ]]
mkdir "$proof"
version=4.2.0
digest=a64a66781b9e5312ad052d32689e23252f745b27ee8818ac2ac0c8209bc0b90e
here=$(cd "$(dirname "$0")" && pwd)
finish(){
 prior=$?;trap - EXIT;set +e;post=0
 for manifest in source tools compiler-providers provider startup startup-providers;do
  [[ ! -f $proof/$manifest.sha256 ]] || sha256sum -c --quiet "$proof/$manifest.sha256" > "$proof/$manifest.post.log" 2>&1 || post=1
 done
 printf 'prior=%s post=%s\n' "$prior" "$post" > "$proof/terminal.txt"
 ((prior==0 && post==0))
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sha256sum "$here/install-pmi1-mpich.sh" "$here/mpi-startup.c" "$repo/.devcontainer/Dockerfile" > "$proof/source.sha256"
grep -Fx "ARG MPICH_VERSION=$version" "$repo/.devcontainer/Dockerfile"
grep -Fx "ARG MPICH_SHA256=$digest" "$repo/.devcontainer/Dockerfile"
for name in gcc g++ gfortran make curl tar sha256sum bash timeout awk ldd readlink grep sort xargs cp find;do
 sha256sum "$(readlink -f "$(command -v "$name")")" >> "$proof/tools.sha256"
done
for name in gcc g++ gfortran make;do ldd "$(readlink -f "$(command -v "$name")")" >> "$proof/compiler.ldd.txt";done
! grep -q 'not found' "$proof/compiler.ldd.txt"
awk '$2=="=>"&&substr($3,1,1)=="/"{print $3}substr($1,1,1)=="/"{print $1}' "$proof/compiler.ldd.txt" | sort -u | xargs -r sha256sum > "$proof/compiler-providers.sha256"
printf 'version=%s sha256=%s prefix=%s\n' "$version" "$digest" "$prefix" > "$proof/settings.txt"
printf 'UCX_TLS=%s\n' "${UCX_TLS-NotSet}" >> "$proof/settings.txt"
source_root=$(mktemp -d "${RUNNER_TEMP:?}/issue234-mpich-source.XXXXXX")
printf '%s\n' "$source_root" > "$proof/source-root.txt"
timeout -k 10s 120s curl -fsSL --max-time 110 \
 "https://www.mpich.org/static/downloads/$version/mpich-$version.tar.gz" -o "$source_root/mpich.tar.gz"
printf '%s  %s\n' "$digest" "$source_root/mpich.tar.gz" | sha256sum -c - > "$proof/release.sha.log"
tar xzf "$source_root/mpich.tar.gz" -C "$source_root"
cd "$source_root/mpich-$version"
printf '%q ' ./configure "--prefix=$prefix" --with-device=ch4:ucx --with-ucx=/usr --with-pm=hydra --with-pmi=pmi1 --without-pmix > "$proof/configure.argv"
CC=gcc CXX=g++ FC=gfortran timeout -k 30s 600s ./configure "--prefix=$prefix" \
 --with-device=ch4:ucx --with-ucx=/usr --with-pm=hydra --with-pmi=pmi1 --without-pmix > "$proof/configure.log" 2>&1
timeout -k 30s 1800s make -j4 > "$proof/make.log" 2>&1
timeout -k 30s 300s make install > "$proof/install.log" 2>&1
cp config.log "$proof/config.log"
"$prefix/bin/mpichversion" > "$proof/mpichversion.txt"
grep -q -- '--with-pmi=pmi1' "$proof/mpichversion.txt"
grep -q -- '--without-pmix' "$proof/mpichversion.txt"
find "$prefix" -type f -print0 | sort -z | xargs -0 sha256sum > "$proof/provider.sha256"
timeout -k 10s 60s "$prefix/bin/mpicc" -std=c11 -O0 -Wall -Wextra -Werror \
 "$here/mpi-startup.c" -o "$proof/mpi-startup"
LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$proof/mpi-startup" > "$proof/startup.ldd.txt"
! grep -q 'not found\|libpmix' "$proof/startup.ldd.txt"
awk -v prefix="$prefix/lib/" '$1~/^libmpi[.]so/ && index($3,prefix)==1{n++}END{exit(n!=1)}' "$proof/startup.ldd.txt"
sha256sum "$proof/mpi-startup" > "$proof/startup.sha256"
awk '$2=="=>"&&substr($3,1,1)=="/"{print $3}substr($1,1,1)=="/"{print $1}' "$proof/startup.ldd.txt" | sort -u | xargs -r sha256sum > "$proof/startup-providers.sha256"
for world in 2 4;do
 set +e
 LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" timeout -k 10s 60s \
  "$prefix/bin/mpiexec" -n "$world" "$proof/mpi-startup" "$world" > "$proof/world-$world.stdout" 2> "$proof/world-$world.stderr"
 status=$?;set -e;printf '%s\n' "$status" > "$proof/world-$world.native.status"
 test "$status" = 0
 awk -v world="$world" '
  BEGIN{expected_sum=world*(world-1)/2}
  NF!=7 || $1!="MPI_STARTUP"{bad=1;next}
  {split($2,r,"=");split($3,s,"=");split($4,q,"=");split($5,p,"=");split($6,t,"=");
   if($2!~/^rank=[0-3]$/||r[2]>=world||seen[r[2]]++||$3!="size="world||$4!~/^required=[0-3]$/||$5!~/^provided=[0-3]$/||p[2]<q[2]||$6!="rank_sum="expected_sum||$7!="ok=1")bad=1;n++}
  END{exit(bad||n!=world)}' "$proof/world-$world.stdout"
done
printf 'MPI_PROVIDER_READY version=4.2.0 pmi=pmi1 worlds=2,4\n' > "$proof/ready.txt"
printf '%s\n' "$prefix/bin" >> "${GITHUB_PATH:?Actions job environment}"
printf 'MPICC=%s/bin/mpicc\nMVMC_ISSUE234_MPI_PREFIX=%s\nMVMC_ISSUE234_MPI_RECEIPT=%s\n' "$prefix" "$prefix" "$proof" >> "${GITHUB_ENV:?}"
printf 'LD_LIBRARY_PATH=%s/lib%s\nPKG_CONFIG_PATH=%s/lib/pkgconfig%s\n' "$prefix" "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$prefix" "${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" >> "$GITHUB_ENV"
