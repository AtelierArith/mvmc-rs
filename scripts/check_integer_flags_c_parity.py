#!/usr/bin/env python3
"""Optional actual-C normalized orbital flags, raw values, RNG and SR selection."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function
from check_orbital_contracts_c_parity import definition


def cases():
    raw = [-2, 0, 1, 2, 3]
    for ap_complex,p_complex in [(0,None),(1,None),(2,None),(0,0),(1,0),(0,1),(1,1)]:
        for shift in range(5):
            flags = "".join(f"{99 if k%2 else 4-k} {raw[(k+shift)%5]}\n" for k in range(5))
            ap = definition(5, "0 0 0 1\n0 1 1 1\n1 0 1 1\n1 1 0 1\n",flags,complex_flag=ap_complex)
            p = definition(5,"0 1 0 1\n",flags,complex_flag=p_complex) if p_complex is not None else ""
            for seed in (1,11272):
                yield f"ap{ap_complex}_p{p_complex}_rotate{shift}_seed{seed}",seed,ap,p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--source",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    src=(args.source or root/"extern/mVMC-1.3.0")/"src"
    reader=(src/"mVMC/readdef.c").read_text()
    parameter=(src/"mVMC/parameter.c").read_text()
    sr=(src/"mVMC/stcopt.c").read_text()
    # Verify all reused header/site/AP/P/flag bodies as well as new excerpts.
    common = "\n".join(function(reader, signature) for signature in (
        "int ReadDefFileError(", "int CheckSite(\n", "int CheckPairSite(\n",
        "int GetInfoOpt(FILE", "int GetInfoOptOrbitalParalell(FILE",
        "char *ReadBuffIntCmpFlg(FILE", "int GetInfoOrbitalAntiParallel(FILE",
        "int GetInfoOrbitalParallel(FILE"))
    materialize(root, "orbital_contracts_upstream.inc", common, [src / "mVMC/readdef.c"], args.write)
    start=reader.index("  if (rank == 0) {\n    AllComplexFlag =")
    end=reader.index("\n\n  if (info != 0)",start)
    materialize(root,"integer_flag_complex_upstream.inc",reader[start:end],[src/"mVMC/readdef.c"],args.write)
    materialize(root,"integer_flag_init_upstream.inc",function(parameter,"void InitParameter() {"),[src/"mVMC/parameter.c"],args.write)
    start=sr.index("  si = 0;\n  for(pi=0;pi<2*nPara;pi++)")
    end=sr.index("\n\n#ifdef _DEBUG_STCOPT",start)
    materialize(root,"integer_flag_sr_upstream.inc",sr[start:end],[src/"mVMC/stcopt.c"],args.write)
    hashes="; ".join(f"{name} sha256={hashlib.sha256((src/name).read_bytes()).hexdigest()}" for name in ("mVMC/readdef.c","mVMC/parameter.c","mVMC/stcopt.c","sfmt/SFMT.c"))
    output=f"# C AP/P header normalization, integer flags, actual InitParameter/SFMT and SR filter; Apple clang 17 -O0 -ffp-contract=off -DMEXP=19937; {hashes}\n"
    count=0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-integer-flags-") as directory:
        tmp=Path(directory)
        subprocess.run(["cc","-O0","-ffp-contract=off","-DMEXP=19937","-I",str(src/"sfmt"),str(root/"c_toolbox/integer_flags.c"),str(src/"sfmt/SFMT.c"),"-lm","-o",str(tmp/"probe")],check=True)
        for name,seed,ap,p in cases():
            (tmp/"ap.def").write_text(ap)
            if p: (tmp/"p.def").write_text(p)
            result=subprocess.run([str(tmp/"probe"),str(tmp/"ap.def"),str(tmp/"p.def") if p else "-",str(seed)],capture_output=True,text=True,check=True,timeout=5)
            lines=result.stdout.splitlines()
            assert len(lines)==5 and lines[0].split()[-1]=="0"
            output+=f"{name} {seed} {' '.join(lines[0].split())}\n{ap.replace(chr(10),'|')}\n{p.replace(chr(10),'|')}\n"
            output+="\n".join(" ".join(line.split()) for line in lines[1:])+"\n"
            count+=1
    target=root/"tests/fixtures/optimization_flags/c_integer_flags.txt"
    if args.write:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or target.read_text()!=output: target.write_text(output)
    else:
        assert target.read_text()==output,"C integer flag fixture changed"
    print(f"{count} C integer flag/read/init/RNG/SR-filter cases passed")


if __name__=="__main__":
    main()
