# Optional developer-only C allocation/shadow-copy probe; never used by Cargo.
# This is not a full sampler, matrix, differential or OO/HO oracle.
using SHA

length(ARGS) == 1 || error("usage: julia check_threaded_182_real_shadow.jl NEW-external-artifact-dir")
repo = realpath(dirname(@__DIR__))
requested_dest = abspath(ARGS[1])
# Resolve the existing parent so a symlink cannot bypass repository exclusion.
# mkdir below is exclusive: never reuse an existing artifact directory.
dest = joinpath(realpath(dirname(requested_dest)), basename(requested_dest))
startswith(dest * "/", repo * "/") && error("artifacts must be outside repository")
(ispath(dest) || islink(dest)) && error("artifact directory must not exist")
sources = (
    ("setmemory.c", "573d1fb995de33386e7452f5e2fa34aeb05efe14b5bf4a904eb54d2ac7cf1b9e"),
    ("vmccal_fsz.c", "9ac529ef2a18d80f61c5aca92aacce10391e2794ecc56aa8ca947691dbdd974e"),
)
texts = Dict{String,String}()
for (name, expected) in sources
    text = read(joinpath(repo, "extern/mVMC-1.3.0/src/mVMC", name), String)
    bytes2hex(sha256(text)) == expected || error("upstream source changed: $name")
    texts[name] = text
end
function unique_line(text, needle)
    matches = filter(line -> occursin(needle, line), split(text, '\n'))
    length(matches) == 1 || error("extraction boundary changed: $needle")
    only(matches)
end
allocation = join([unique_line(texts["setmemory.c"], needle) for needle in
    ("  InvM = ", "  PfM = ", "  InvM_real      = ", "  PfM_real       = ")], "\n")
copy_loop = unique_line(texts["vmccal_fsz.c"], "for(tmp_i=0;tmp_i<NQPFull*(Nsize*Nsize+1);tmp_i++)")
function upstream_notice(name)
    text = texts[name]
    startswith(text, "/*\n") || error("upstream license header changed: $name")
    boundary = findfirst("*/", text)
    boundary === nothing && error("upstream license header missing: $name")
    "/* Extracted origin: mVMC-1.3.0/src/mVMC/$name; upstream author Satoshi Morita. */\n" *
        text[1:last(boundary)] * "\n"
end
notices = join(upstream_notice.(first.(sources)), "\n")
driver = notices * """
/* Standalone shadow-layout harness; source hashes and extraction boundaries
 * are recorded in provenance.txt. Extracted upstream statements retain
 * their GPL-3.0-or-later license; this is not a full numerical runner. */
#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  for (int NQPFull=1; NQPFull<=4; ++NQPFull) {
    for (int Nsize=2; Nsize<=8; Nsize+=2) {
      double complex *InvM, *PfM;
      double *InvM_real, *PfM_real;
      int tmp_i;
$allocation
      assert(InvM && InvM_real);
      int total=NQPFull*(Nsize*Nsize+1);
      for (int k=0;k<total;++k) {
        InvM[k]=-777.0+999.0*I;
        InvM_real[k]=(k+1)*0.125;
      }
$copy_loop
      for (int k=0;k<NQPFull*Nsize*Nsize;++k)
        assert(creal(InvM[k])==InvM_real[k] && cimag(InvM[k])==0.0);
      for (int q=0;q<NQPFull;++q)
        assert(creal(PfM[q])==PfM_real[q] && cimag(PfM[q])==0.0);
      printf("QP=%d side=%d inverse=%d PfM=%d shadow_verified\\n",
             NQPFull,Nsize,NQPFull*Nsize*Nsize,NQPFull);
      free(InvM); free(InvM_real);
    }
  }
  return 0;
}
"""
mkdir(dest)
path = joinpath(dest, "probe.c")
binary = joinpath(dest, "probe")
write(path, driver)
compiler = get(ENV, "CC", "cc")
command = `$compiler -std=c11 -O0 -ffp-contract=off -Wall -Wextra -Werror $path -o $binary`
run(command)
open(joinpath(dest, "stdout.txt"), "w") do io
    run(pipeline(`$binary`, stdout=io, stderr=io))
end
open(joinpath(dest, "provenance.txt"), "w") do io
    println(io, "scope=extracted-C-allocation-and-shadow-copy; patterned-inputs; not-full-runner-or-numerical-stage-proof")
    println(io, "origin=mVMC-1.3.0/src/mVMC/{setmemory.c,vmccal_fsz.c}; author=Satoshi Morita; extracted-code-license=GPL-3.0-or-later")
    println(io, "Original upstream copyright/license notices are retained verbatim in probe.c:")
    println(io, notices)
    println(io, "Superseded investigation: missing PfM synchronization was suspected, NOT a C bug. PfM aliases the inverse allocation tail; the full copy includes it. Independently confirmed by Pauli.")
    println(io, "boundaries=setmemory.c allocation/alias lines; vmccal_fsz.c:82 complete copy loop; serial execution of disjoint copy")
    println(io, "platform=$(Sys.MACHINE) Julia=$VERSION compiler=", readchomp(`$compiler --version`))
    println(io, "command=", command)
    for (name, hash) in sources
        println(io, name, " sha256=", hash)
    end
    for file in (path, binary, @__FILE__)
        println(io, basename(file), " sha256=", bytes2hex(sha256(read(file))))
    end
end
println("PASS: 16 extracted C shadow-layout cases; artifacts=$dest")
