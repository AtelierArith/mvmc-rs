#!/usr/bin/env python3
"""Optional native DH integer flag readers and C gauge eligibility checks."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader, parameter = (src / "readdef.c").read_text(), (src / "parameter.c").read_text()
    # Verify all reused header/site/AP/P/flag bodies as well as new excerpts.
    common = "\n".join(function(reader, signature) for signature in (
        "int ReadDefFileError(", "int CheckSite(\n", "int CheckPairSite(\n",
        "int GetInfoOpt(FILE", "int GetInfoOptOrbitalParalell(FILE",
        "char *ReadBuffIntCmpFlg(FILE", "int GetInfoOrbitalAntiParallel(FILE",
        "int GetInfoOrbitalParallel(FILE"))
    materialize(root, "orbital_contracts_upstream.inc", common, [src / "readdef.c"], args.write)
    body = "\n".join(function(reader, signature) for signature in (
        "int CheckQuadSite(\n", "int\nGetInfoDH2(FILE", "int\nGetInfoDH4(FILE"))
    materialize(root, "dh_flag_reader_upstream.inc", body, [src / "readdef.c"], args.write)
    materialize(root, "flag_shift_upstream.inc", function(parameter, "void SetFlagShift() {"),
                [src / "parameter.c"], args.write)
    provenance = "; ".join(f"{name} sha256={hashlib.sha256((src/name).read_bytes()).hexdigest()}"
                           for name in ("readdef.c", "parameter.c"))
    dh = f"# actual C DH readers; zero untouched imaginary storage; Apple clang 17 -O0; {provenance}\n"
    shifts = f"# actual C SetFlagShift eligibility only; Apple clang 17 -O0; {provenance}\n"
    dh_count = shift_count = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-projection-flags-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", str(root / "c_toolbox/projection_flags.c"),
                        "-o", str(tmp / "probe")], check=True)
        for family in (2, 4):
            width = 6 if family == 2 else 10
            for complex_flag in (0, 1):
                for rotation in range(6):
                    raw = [-2, -1, 0, 1, 2, 3]
                    maps = "".join(" ".join(map(str, [i]+[(i+k+1)%3 for k in range(family)]+[0]))+"\n"
                                   for i in range(3))
                    flags = "".join(f"{99-k} {raw[(k+rotation)%6]}\n" for k in range(width))
                    payload = maps + flags
                    (tmp / "dh.def").write_text(payload)
                    result = subprocess.run([str(tmp / "probe"),str(family),str(complex_flag),
                                             str(tmp / "dh.def")], capture_output=True,
                                            text=True, check=True, timeout=5)
                    dh += f"dh{family}_complex{complex_flag}_rotate{rotation} {family} {complex_flag}\n"
                    dh += payload.replace("\n", "|")+"\n"
                    dh += "\n".join(" ".join(line.split()) for line in result.stdout.splitlines())+"\n"
                    dh_count += 1
        for counts in ((2,2,1,1),(0,2,1,1),(2,0,1,1),(2,2,0,1),(2,2,1,0)):
            n = counts[0]+counts[1]+6*counts[2]+10*counts[3]
            variants = [("all_one", [1]*(2*n))]
            if counts == (2,2,1,1):
                for index in range(n):
                    for value in (-2,0,2,3):
                        flags = [1]*(2*n)
                        flags[2*index] = value
                        variants.append((f"real{index}_flag{value}", flags))
                # Imaginary values must not influence real gauge eligibility.
                variants.append(("imaginary_nonbinary", [v for _ in range(n) for v in (1,3)]))
            for name, flags in variants:
                header = " ".join(map(str, counts))
                values = " ".join(map(str, flags))
                result = subprocess.run([str(tmp / "probe")],input=header+"\n"+values+"\n",
                                        capture_output=True,text=True,check=True,timeout=5)
                shifts += f"{header.replace(' ', '_')}_{name} {header}\n{values}\n{result.stdout.strip()}\n"
                shift_count += 1
    for name, contents in (("c_dh_flags.txt",dh),("c_shift_flags.txt",shifts)):
        target = root / "tests/fixtures/optimization_flags" / name
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_text()!=contents: target.write_text(contents)
        else:
            assert target.read_text()==contents, f"C fixture changed: {name}"
    print(f"{dh_count} C DH raw-flag readers and {shift_count} C gauge eligibility cases passed")


if __name__ == "__main__":
    main()
