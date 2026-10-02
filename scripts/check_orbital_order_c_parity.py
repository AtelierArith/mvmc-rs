#!/usr/bin/env python3
"""Compile C's filename registry and AP/P readers, independent of MPI."""
import argparse
import hashlib
import itertools
import re
from pathlib import Path
import subprocess
import tempfile

from check_projection_count_c_parity import function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader = (src / "readdef.c").read_text()
    # Retain the upstream keyword iteration and empty-slot handling verbatim.
    start = reader.index("    for (iKWidx = 0; iKWidx < KWIdxInt_end; iKWidx++) {", reader.index("GetInfoFromModPara(bufInt, bufDouble)"))
    end = reader.index("      fprintf(stdout,", start)
    iteration = reader[start:end]
    header_start = reader.index("          case KWOrbital:")
    header_end = reader.index("          case KWOrbitalGeneral:", header_start)
    header_cases = reader[header_start:header_end]
    code = '''
#include <assert.h>
#include <ctype.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int MPI_Comm;
#define D_FileNameMax 256
#include "readdef.h"
'''
    for signature in ("int ReadDefFileError(", "int CheckWords(\n", "int CheckKW(\n",
                      "int GetFileName(\n", "int CheckSite(\n", "int CheckPairSite(\n",
                      "int GetInfoOpt(FILE", "int GetInfoOptOrbitalParalell(FILE",
                      "char *ReadBuffIntCmpFlg(FILE", "int GetInfoOrbitalAntiParallel(FILE",
                      "int GetInfoOrbitalParallel(FILE"):
        # Skip forward declarations without altering the function body.
        definition = next(match.start() for match in re.finditer(re.escape(signature), reader)
                          if ";" not in reader[match.start():reader.index("{", match.start())])
        code += function(reader[definition:], signature) + "\n"
    code += '''
int main(int argc, char **argv) {
  if (argc == 2) {
    FILE *fp=tmpfile(); assert(fp); fputs("0 0\\n",fp); rewind(fp);
    int flags[4]={0}, count=0;
    assert(GetInfoOptOrbitalParalell(fp,flags,1,&count,0)==1);
    assert(count==2 && flags[0]==0 && flags[1]==1 && flags[2]==0 && flags[3]==1);
    fclose(fp); puts("complex P inactive real flags: 0 1 0 1"); return 0;
  }
  assert(argc == 3);
  char names[KWIdxInt_end][D_CharTmpReadDef], defname[D_CharTmpReadDef];
  int iKWidx;
  assert(GetFileName(argv[1], names) == 0);
  cFileNameListFile = names;
'''
    code += iteration
    code += '''
      printf("%s ", cKWListOfFileNameList[iKWidx]);
  }
  puts("");
  int bufInt[ParamIdxInt_End]={0}, iNOrbitalAntiParallel=0, iNOrbitalParallel=0;
  int iFlgOrbitalAntiParallel=0, iFlgOrbitalParallel=0, iOrbitalComplex=0;
  char *cerr; FILE *fp;
'''
    code += iteration
    code += '''
      if (iKWidx!=KWOrbital && iKWidx!=KWOrbitalAntiParallel && iKWidx!=KWOrbitalParallel) continue;
      fp=fopen(defname,"r"); assert(fp);
      switch(iKWidx) {
'''
    code += header_cases
    code += '''
      }
      assert(cerr); fclose(fp);
  }
  int width=bufInt[IdxNOrbit], ap=iNOrbitalAntiParallel, p=iNOrbitalParallel;
  printf("%d %d\\n",ap,width);
  int idx_data[4][4]={{0}}, sign_data[4][4]={{0}}, *idx[4], *sign[4], flags[26]={0};
  for(int i=0;i<4;i++) {idx[i]=idx_data[i];sign[i]=sign_data[i];}
  int count=0, info=0, boundary=atoi(argv[2]); char buf[256];
  fp=fopen(names[KWOrbital][0]?names[KWOrbital]:names[KWOrbitalAntiParallel], "r");
  for(int i=0;i<5;i++) assert(fgets(buf,sizeof(buf),fp));
  info=GetInfoOrbitalAntiParallel(fp,idx,flags,sign,&count,0,0,1,boundary,2,ap,"AP"); fclose(fp);
  fp=fopen(names[KWOrbitalParallel],"r");
  for(int i=0;i<5;i++) assert(fgets(buf,sizeof(buf),fp));
  info+=GetInfoOrbitalParallel(fp,idx,flags,sign,&count,ap,0,1,boundary,2,p,ap,"P"); fclose(fp);
  assert(info==0 && count==width);
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) printf("%d ",idx[i][j]); puts("");
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) printf("%d ",sign[i][j]); puts("");
  for(int i=0;i<2*width;i++) printf("%d ",flags[i]); puts("");
}
'''
    cases = {}
    with tempfile.TemporaryDirectory(prefix="mvmc-c-orbital-order-") as directory:
        tmp = Path(directory)
        (tmp / "probe.c").write_text(code)
        subprocess.run(["cc", "-O0", "-I", str(src / "include"), str(tmp / "probe.c"),
                        "-o", str(tmp / "probe")], check=True)
        subprocess.run([str(tmp / "probe"), "complex"], check=True)
        def definition(header, width, rows):
            return f"===\n{header} {width}\nComplexType 0\n===\n===\n{rows}"
        (tmp / "ap.def").write_text(definition("NOrbitalIdx", 7,
            "0 0 0 -1\n0 1 1 1\n1 0 1 -1\n1 1 0 1\n" +
            "".join(f"{i} {int(i % 3 != 0)}\n" for i in range(7))))
        (tmp / "p.def").write_text(definition("NOrbitalParallel", 3,
            "0 1 0 -1\n0 1\n1 0\n2 1\n"))
        kinds = ["ModPara", "LocSpin", "OrbitalAntiParallel", "OrbitalParallel", "TransSym"]
        paths = ["modpara.def", "locspin.def", "ap.def", "p.def", "qp.def"]
        for alias in ("Orbital", "OrbitalAntiParallel"):
            for order in itertools.permutations(range(5)):
                (tmp / "namelist.def").write_text("".join(
                    f"{alias if i == 2 else kinds[i]} {tmp / paths[i]}\n" for i in order))
                for boundary in (0, 1):
                    actual = subprocess.check_output([str(tmp / "probe"),
                        str(tmp / "namelist.def"), str(boundary)], text=True)
                    actual = "\n".join(line.rstrip() for line in actual.splitlines()) + "\n"
                    key = (alias, boundary)
                    if key in cases:
                        assert actual == cases[key], (key, order)
                    cases[key] = actual
    checksums = "; ".join(f"{name} sha256={hashlib.sha256((src / name).read_bytes()).hexdigest()}"
                          for name in ("readdef.c", "include/readdef.h"))
    actual = f"# mVMC-1.3.0 actual C filename registry, keyword iteration and AP/P readers; {checksums}\n"
    for (alias, boundary), result in cases.items():
        actual += f"{alias} {boundary}\n{result}"
    target = root / "tests/fixtures/orbital_general/c_order.txt"
    if args.write:
        target.write_text(actual)
    else:
        assert actual == target.read_text(), "C orbital order contract changed"
    print("480 C namelist permutations/aliases/boundaries passed")


if __name__ == "__main__":
    main()
