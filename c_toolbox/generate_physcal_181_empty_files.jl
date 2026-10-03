# Explicit optional C file-opening oracle; no sampling/MPI or numerical oracle.
using SHA
repo = dirname(@__DIR__)
source_path = joinpath(repo, "extern/mVMC-1.3.0/src/mVMC/initfile.c")
source = read(source_path, String)
start = first(findfirst("void InitFilePhysCal(", source))
stop = first(findnext("void CloseFile(", source, start)) - 1
body = source[start:stop]
fixture = joinpath(repo, "tests/fixtures/physcal_181/energy-only")
mkpath(fixture)
mktempdir() do work
    driver = """
    #include <stdio.h>
    #include <stdlib.h>
    #define D_FileNameMax 4096
    int NDataIdxStart=7, FlagBinary=0, NPara=14;
    int NCisAjs=0, NCisAjsCktAlt=0, NCisAjsCktAltDC=0, NLanczosMode;
    char *CDataFileHead="zvo";
    FILE *FileOut,*FileVar,*FileLS,*FileLSQQQQ,*FileCisAjs,*FileCisAjsCktAlt,*FileCisAjsCktAltDC;
    FILE *FileLSCisAjs,*FileLSCisAjsCktAlt,*FileLSCisAjsCktAltDC;
    """ * body * """
    int main(int argc, char **argv) {
      if(argc!=2) return 2;
      NLanczosMode=atoi(argv[1]); InitFilePhysCal(0,0);
      FILE *files[]={FileOut,FileVar,FileLS,FileLSQQQQ,FileCisAjs,FileCisAjsCktAlt,FileCisAjsCktAltDC,FileLSCisAjs,FileLSCisAjsCktAlt,FileLSCisAjsCktAltDC};
      for(int i=0;i<10;i++) if(files[i]) fclose(files[i]);
      return 0;
    }
    """
    path = joinpath(work,"probe.c"); write(path,driver)
    binary = joinpath(work,"probe")
    run(`cc -std=c11 -O0 $path -o $binary`)
    for mode in 0:2
        stage = joinpath(work,"mode-$mode"); mkpath(stage)
        cd(stage) do; run(`$binary $mode`); end
        write(joinpath(fixture,"mode-$mode-files.txt"),join(sort(readdir(stage)),"\n") * "\n")
    end
    write(joinpath(fixture,"provenance.txt"),
        "scope=actual C InitFilePhysCal, zero normal Green counts, serial rank0, FlagBinary0; not full executable/input-parser/sampling parity\n" *
        "source_sha256=$(bytes2hex(sha256(read(source_path))))\n" *
        "extraction=void InitFilePhysCal through before void CloseFile\n" *
        "driver_sha256=$(bytes2hex(sha256(driver)))\n" *
        "architecture=$(Sys.MACHINE)\ncompiler=$(strip(read(`cc --version`,String)))\noptions=-std=c11 -O0\nBLAS=none\nRNG=none\n")
end
