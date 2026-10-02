#!/usr/bin/env python3
"""Compile the actual C count conversion and QP kernels for #45 contracts."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--write", action="store_true")
    args.add_argument("--source", type=Path, help="Path to the mVMC-1.3.0 source tree")
    options = args.parse_args()
    write = options.write
    root = Path(__file__).resolve().parent.parent
    src = (options.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader = (src / "readdef.c").read_text()
    qp = (src / "qp.c").read_text()
    read_count = next(line for line in reader.splitlines() if "NMPTrans = bufInt[IdxMPTrans];" in line)
    start = reader.index("  if (NMPTrans < 0) {")
    end = reader.index("  if (DSROptStepDt < 0)", start)
    boundary = reader[start:end]
    sizes = "\n".join(line for line in reader.splitlines() if line.strip().startswith(("NQPFix =", "NQPFull =")))
    code = r'''
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int NMPTrans, APFlag, NQPFix, NQPFull, NSPGaussLeg=1, NSPStot=0;
int NQPOptTrans, NOptTrans, FlagOptTrans;
double complex ParaQPTrans[2], OptTrans[2];
double complex QPFixWeight[4], QPFullWeight[8];
double complex SPGLCos[1], SPGLSin[1], SPGLCosSin[1], SPGLCosCos[1], SPGLSinSin[1];
double scratch[2];
int scratch_offset;
void RequestWorkSpaceDouble(int n) { assert(n == 2); scratch_offset=0; }
double *GetWorkSpaceDouble(int n) { double *p=scratch+scratch_offset; scratch_offset+=n; return p; }
void ReleaseWorkSpaceDouble(void) {}
void GaussLeg(double, double, double *, double *, int);
double LegendrePoly(double, int);
void UpdateQPWeight(void);
'''
    code += function(qp, "void InitQPWeight()") + "\n"
    code += function(qp, "void UpdateQPWeight()") + "\n"
    code += r'''
void print_bits(double complex *v, int n) {
  for(int j=0;j<n;j++) {
    double parts[2]={creal(v[j]),cimag(v[j])};
    for(int k=0;k<2;k++) { uint64_t bits; memcpy(&bits, &parts[k], 8); printf("%016llx ",(unsigned long long)bits); }
  }
  puts("");
}
int main(void) {
  int counts[]={0,1,-1,2,-2};
  for(int c=0;c<5;c++) for(int opt=0;opt<=2;opt+=2) {
    int IdxMPTrans=0, bufInt[]={counts[c]};
    NQPOptTrans=opt?opt:1; NOptTrans=opt; FlagOptTrans=opt>0;
    ParaQPTrans[0]=1.0+0.25*I; ParaQPTrans[1]=-0.5-0.125*I;
    OptTrans[0]=0.75+0.5*I; OptTrans[1]=-0.25+0.75*I;
'''
    code += read_count + "\n" + boundary + sizes + "\n"
    code += r'''
    InitQPWeight();
    printf("%d %d %d %d %d %d\n",counts[c],NMPTrans,APFlag,NQPFix,NQPFull,opt);
    print_bits(QPFixWeight,NQPFix); print_bits(QPFullWeight,NQPFull);
    print_bits(SPGLCos,1); print_bits(SPGLSin,1);
  }
  return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="mvmc-c-projection-") as tmp:
        tmp = Path(tmp)
        (tmp / "probe.c").write_text(code)
        subprocess.run(["cc", "-O0", "-ffp-contract=off", "-include", "math.h", "-I", str(src / "include"),
                        str(tmp / "probe.c"), str(src / "gauleg.c"),
                        str(src / "legendrepoly.c"), "-lm", "-o", str(tmp / "probe")], check=True)
        actual = subprocess.check_output([str(tmp / "probe")], text=True)
    checksums = "; ".join(f"{name} sha256={hashlib.sha256((src / name).read_bytes()).hexdigest()}"
                          for name in ("readdef.c", "qp.c"))
    actual = f"# mVMC-1.3.0 snapshot; actual C readdef conversion and QP kernels; NSPGaussLeg=1; {checksums}\n" + "\n".join(line.rstrip() for line in actual.splitlines()) + "\n"
    target = root / "tests/fixtures/projection_count/c_contracts.txt"
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(actual)
    else:
        assert actual == target.read_text(), "C projection contract changed"
    print("10 C projection/boundary/OptTrans cases passed")


if __name__ == "__main__":
    main()
