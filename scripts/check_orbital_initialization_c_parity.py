#!/usr/bin/env python3
"""C InitParameter/SyncModifiedParameter and native SFMT declared-width oracle."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from check_projection_count_c_parity import function
from numerical_comparison import compare_text
import math
import re
from c_toolbox import materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src"
    parameter = (src / "mVMC/parameter.c").read_text()
    slater = (src / "mVMC/slater.c").read_text()
    bodies = [function(parameter, "void InitParameter()"),
              function(parameter, "void SyncModifiedParameter(MPI_Comm comm)"),
              function(parameter, "int ReadInitParameter(char *initFile)"),
              function(slater, "void UpdateSlaterElm_fcmp()")]
    materialize(root, "orbital_parameter_upstream.inc", "\n".join(bodies),
                [src / "mVMC/parameter.c", src / "mVMC/slater.c"], args.write)
    probe_source = root / "c_toolbox/orbital_initialization.c"
    with tempfile.TemporaryDirectory(prefix="mvmc-c-orbital-init-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", "-ffp-contract=off", "-DMEXP=19937", "-I", str(src / "sfmt"),
                        str(probe_source), str(src / "sfmt/SFMT.c"), "-lm", "-o", str(tmp / "probe")], check=True)
        actual = subprocess.check_output([str(tmp / "probe")], text=True)
        flag_cases = "".join(subprocess.check_output([str(tmp / "probe"), "flags", str(mode), str(width)], text=True)
                             for width in (4, 15) for mode in (0, 1))
        flag_cases += "".join(subprocess.check_output([str(tmp / "probe"), "combined", str(active), "rbm", "dh"], text=True)
                              for active in (1, 0))
        flag_cases += subprocess.check_output([str(tmp / "probe"), "sync", "four", "declared", "slots", "after", "sr"], text=True)
        prefix_cases = "".join(subprocess.check_output([str(tmp / "probe"), "prefix", str(mode), str(active), str(mask), "rng"], text=True)
                                for mode in (0, 1) for active in (0, 8, 16, 18, 27) for mask in (0, 10, 15))
        shared_case = subprocess.check_output([str(tmp / "probe"), "shared"], text=True)
        flag_cases += subprocess.check_output([str(tmp / "probe"), "rbm-loaded",
            str(root / "tests/fixtures/rbm/production/initial.def"), "full", "slater", "sync", "all", "slots"], text=True)
        loaded_cases = ""
        for mode in (0, 1):
            values = [(8.0 if i == 12 else (i+1)/16, -(i+1)/32 if mode else 0.0)
                      for i in range(13)]
            payload = "0 0 0 0 0 0 " + " ".join(f"{real} {imag} 99" for real, imag in values) + "\n"
            (tmp / "initial.def").write_text(payload)
            loaded_cases += f"{mode}\n" + subprocess.check_output([str(tmp / "probe"), str(tmp / "initial.def")], text=True)
    checksums = "; ".join(f"{name} sha256={hashlib.sha256((src / name).read_bytes()).hexdigest()}"
                          for name in ("mVMC/parameter.c", "mVMC/slater.c", "sfmt/SFMT.c", "sfmt/SFMT.h", "sfmt/SFMT-params19937.h"))
    actual = f"# mVMC-1.3.0 actual InitParameter/SyncModifiedParameter/SFMT; NSlater=13; no correlation/RBM/OptTrans factors; {checksums}\n" + "\n".join(line.rstrip() for line in actual.splitlines()) + "\n"
    target = root / "tests/fixtures/orbital_general/c_initialization.txt"
    if args.write:
        if not target.exists() or target.read_text() != actual:
            target.write_text(actual)
    else:
        compare_text(actual,target.read_text(),lambda r,c,t: (32*math.ulp(1.0),32*math.ulp(1.0)) if r%4 in (1,2) else None)
    for name, body in (("c_declared_flags.txt", flag_cases), ("c_loaded.txt", loaded_cases),
                       ("c_rbm_prefix.txt", prefix_cases), ("c_shared_matrix.txt", shared_case)):
        result = f"# mVMC-1.3.0 actual InitParameter/ReadInitParameter/SyncModifiedParameter/SFMT; {checksums}\n" + "\n".join(line.rstrip() for line in body.splitlines()) + "\n"
        path = target.parent / name
        if args.write:
            if not path.exists() or path.read_text() != result:
                path.write_text(result)
        else:
            def computed(row,column,fields):
                if name == "c_shared_matrix.txt":
                    return (1e-13,1e-13)
                offset=row%4
                if offset not in (1,2): return None
                # Last two declared-flag blocks read/copy supplied values
                # before SyncModifiedParameter; loaded initial values are inputs.
                literal=(name=="c_loaded.txt" or (name=="c_declared_flags.txt" and row//4>=6)) and offset==1
                return None if literal else (32*math.ulp(1.0),32*math.ulp(1.0))
            compare_text(result,path.read_text(),computed)
    print("8 C initialization + 8 declared-flag/sync + 2 full-loading + 30 RBM prefix + 1 shared matrix cases passed")


if __name__ == "__main__":
    main()
