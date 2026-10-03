# Explicit C count/reader probe. Allocation is padded so C's normal reader
# can write the extra duplicate before reporting its count error safely.
using SHA
repo = dirname(@__DIR__)
source_path = joinpath(repo,"extern/mVMC-1.3.0/src/mVMC/readdef.c")
source = read(source_path,String)
function extract(start_marker, stop_marker)
    # The same signature prefix occurs in upstream forward declarations.
    start = first(findlast(start_marker,source))
    stop = first(findnext(stop_marker,source,start))-1
    source[start:stop]
end
reader = extract("int\nGetInfoOneBodyG(FILE *fp, int **ArrayIdx, int **ArrayToIdx, int IndirectGFOn, int Nsite, int NArray, char *defname) {", "// Formerly CisAjsCktAlt")
factored_reader = extract("int GetInfoTwoBodyGEx(FILE *fp, int **ArrayIdx, int **ArrayToIdx, int **ArrayIdxOneBodyG,", "int GetInfoTwoBodyG(FILE")
counter = extract("int CountOneBodyGForLanczos(char *xNameListFile, int Nca, int Ncacadc, int Ns, int **iFlgOneBodyG) {", "int ReadDefFileNInt(")
driver = """
#include <stdio.h>
#include <stdlib.h>
static int CheckPairSite(int a,int b,int n) { return a<0 || b<0 || a>=n || b>=n; }
static int ReadDefFileError(char *name) { (void)name; return 1; }
static int CheckQuadSite(int a,int b,int c,int d,int n) { return CheckPairSite(a,b,n) || CheckPairSite(c,d,n); }
""" * reader * factored_reader * """
/* Adapter supplies the two original declared records. No fake LS execution. */
static int ReadGreen(char *path,int n,int **rows,int nex,int **ex,int ns) {
  (void)path;
  if(nex) {
    int term[8]={0,0,1,0,1,0,0,0};
    for(int i=0;i<8;i++) ex[0][i]=term[i];
  }
  FILE *file=tmpfile(); fputs("0 0 1 0\\n0 0 1 0\\n",file); rewind(file);
  int status=GetInfoOneBodyG(file,rows,NULL,0,ns,n,"two duplicates");
  fclose(file); return status;
}
""" * counter * """
int main(int argc,char **argv) {
  if(argc!=2) return 2;
  int storage[4][4],*map[4]; for(int i=0;i<4;i++) map[i]=storage[i];
  int count=CountOneBodyGForLanczos("adapter",2,0,2,map);
  int row_storage[2][4],*rows[2]={row_storage[0],row_storage[1]};
  FILE *file=tmpfile(); fputs("0 0 1 0\\n0 0 1 0\\n",file); rewind(file);
  int status=GetInfoOneBodyG(file,rows,map,0,2,count,"two duplicates");
  fclose(file);
  printf("declared_records=2\\ncount_after_mode2_dedup=%d\\nnormal_reader_status=%d\\n",count,status);
  int indirect_count=CountOneBodyGForLanczos("adapter",2,1,2,map);
  file=tmpfile(); fputs("0 0 1 0\\n0 0 1 0\\n",file); rewind(file);
  int indirect_status=GetInfoOneBodyG(file,rows,map,1,2,indirect_count,"two duplicates with GEx");
  fclose(file);
  printf("with_gex_canonical_count=%d\\nindirect_reader_status=%d\\n",indirect_count,indirect_status);
  int pair_storage[1][2],*pairs[1]={pair_storage[0]};
  file=tmpfile(); fputs("0 0 1 0 1 0 0 0\\n",file); rewind(file);
  int factored_status=GetInfoTwoBodyGEx(file,pairs,map,rows,2,1,"GEx");
  fclose(file);
  char path[4096]; snprintf(path,sizeof(path),"%s/canonical-layout.txt",argv[1]);
  file=fopen(path,"w"); if(!file) return 3;
  fprintf(file,"%d 1\\n",indirect_count);
  for(int i=0;i<indirect_count;i++) fprintf(file,"%d %d %d %d\\n",rows[i][0],rows[i][1],rows[i][2],rows[i][3]);
  fprintf(file,"%d %d\\n",pairs[0][0],pairs[0][1]); fclose(file);
  return count!=1 || status!=1 || indirect_count!=1 || indirect_status!=0 || factored_status!=0;
}
"""
fixture = joinpath(repo,"tests/fixtures/physcal_181/duplicate-reader"); mkpath(fixture)
mktempdir() do work
    path=joinpath(work,"probe.c"); write(path,driver)
    binary=joinpath(work,"probe"); run(`cc -std=c11 -O0 $path -o $binary`)
    write(joinpath(fixture,"result.txt"),read(`$binary $fixture`,String))
    write(joinpath(fixture,"provenance.txt"),
        "source_sha256=$(bytes2hex(sha256(read(source_path))))\n" *
        "driver_sha256=$(bytes2hex(sha256(driver)))\narchitecture=$(Sys.MACHINE)\ncompiler=$(strip(read(`cc --version`,String)))\noptions=-std=c11 -O0\n" *
        "scope=actual CountOneBodyGForLanczos, GetInfoOneBodyG and GetInfoTwoBodyGEx; ReadGreen adapter supplies declared records; padded storage; no MPI/BLAS/RNG/LS execution\n")
end
