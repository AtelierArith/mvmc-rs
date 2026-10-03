# Optional reader-only oracle; Cargo tests consume the resulting fixture only.
using SHA
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
repo = dirname(@__DIR__)
source_path = joinpath(repo, "extern/mVMC-1.3.0/src/mVMC/readdef.c")
source = read(source_path, String)
function body(first_marker, next_marker)
    first_pos = first(findlast(first_marker, source))
    last_pos = first(findnext(next_marker, source, first_pos)) - 1
    source[first_pos:last_pos]
end
driver = """
#include <stdio.h>
#include <stdlib.h>
static int CheckPairSite(int a,int b,int n) { return a<0 || b<0 || a>=n || b>=n; }
static int ReadDefFileError(char *name) { (void)name; return 1; }
""" * body("int ReadPairHopValue(FILE *fp", "int ReadPairDValue(FILE *fp") *
    body("int ReadPairDValue(FILE *fp", "int GetInfoOptOrbitalParalell(FILE *fp") * """
int main(void) {
  const char *keys[]={"CoulombInter","Hund","Exchange","PairHop"};
  const char *records[]={"0 1 0.375\\n","0 1 -0.125\\n","0 1 0.25\\n","0 1 -0.0625\\n"};
  for(int k=0;k<4;k++) {
    int storage[2][2],*rows[2]={storage[0],storage[1]}; double coefficients[2];
    FILE *f=tmpfile(); if(!f) return 2;
    fputs(records[k],f); rewind(f);
    int n=k==3?2:1;
    int status=k==3?ReadPairHopValue(f,rows,coefficients,6,n,"pairhop"):
      ReadPairDValue(f,rows,coefficients,6,n,"pair");
    fclose(f); if(status) return 3;
    for(int i=0;i<n;i++) printf("%s %d %d %.17g\\n",keys[k],rows[i][0],rows[i][1],coefficients[i]);
  }
  return 0;
}
"""
fixture = joinpath(repo, "tests/fixtures/physcal_181/all-terms-reader")
mkpath(fixture)
mktempdir() do work
    path = joinpath(work, "probe.c"); write(path, driver)
    binary = joinpath(work, "probe")
    run(`cc -std=c11 -O0 $path -o $binary`)
    write(joinpath(fixture, "ordered-pairs.txt"), read(`$binary`, String))
    write(joinpath(fixture, "provenance.txt"),
        "source=extern/mVMC-1.3.0/src/mVMC/readdef.c\nsource_sha256=$(bytes2hex(sha256(read(source_path))))\n" *
        "generator_sha256=$(bytes2hex(sha256(read(@__FILE__))))\ndriver_sha256=$(bytes2hex(sha256(driver)))\n" *
        "architecture=$(Sys.MACHINE)\njulia=$VERSION\ncompiler=$(strip(read(`cc --version`, String)))\noptions=-std=c11 -O0\n" *
        "boundaries=ReadPairHopValue through before ReadPairDValue; ReadPairDValue through before GetInfoOptOrbitalParalell\n" *
        "scope=actual C pair readers, supplied literal records; PairHop expands both directions; no full definition reader, MPI, BLAS, RNG or Lanczos measurement\n" *
        "count_authority=readdef.c:697 NPairHopping=2*declared count\n")
end
