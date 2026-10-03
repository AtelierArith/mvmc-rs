#!/usr/bin/env -S uv run --no-project
"""Optional actual-C FSZ Green kernels; historical Julia arrays are inputs only.

No RBM, RNG, full executable, sampling or SR parity is claimed by this probe.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function
from check_interall_real_c_parity import bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--real", action="store_true", help="Probe the separate scalar FSZ family")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    selected = {
        "projection.c": ["inline double ProjRatio(", "void UpdateProjCnt(", "void UpdateProjCnt_fsz("],
        "pfupdate_fsz.c": ["void CalculateNewPfM_fsz("],
        "pfupdate_two_fsz.c": ["void CalculateNewPfMTwo_fsz(", "void calculateNewPfMTwo_child_fsz("],
        "qp.c": ["double complex CalculateIP_fcmp("],
        "locgrn_fsz.c": ["double complex GreenFunc1_fsz(", "double complex GreenFunc1_fsz2(",
                         "double complex GreenFunc2_fsz(", "double complex GreenFunc2_fsz2("],
    }
    if args.real:
        selected = {
            "projection.c": selected["projection.c"],
            "pfupdate_fsz_real.c": ["void CalculateNewPfM_fsz_real("],
            "pfupdate_two_fsz_real.c": ["void CalculateNewPfMTwo_fsz_real(",
                                      "void calculateNewPfMTwo_child_fsz_real("],
            "qp_real.c": ["double  CalculateIP_real("],
            "locgrn_fsz_real.c": ["double GreenFunc1_fsz_real(", "double GreenFunc1_fsz2_real(",
                                 "double GreenFunc2_fsz_real(", "double GreenFunc2_fsz2_real("],
        }
    files = [src / name for name in selected]
    body = "\n".join(function(path.read_text(), sig)
                     for path in files for sig in selected[path.name])
    snippet = "fsz_green_real_upstream.inc" if args.real else "fsz_green_upstream.inc"
    materialize(root, snippet, body, files, args.write)
    accumulator = src / ("calham_fsz_real.c" if args.real else "calham_fsz.c")
    source = accumulator.read_text()
    start = source.index("for(idx=0;idx<NInterAll;idx++)")
    brace = source.index("{", start)
    snippet = "fsz_interall_real_accumulator_upstream.inc" if args.real else "fsz_interall_accumulator_upstream.inc"
    materialize(root, snippet,
                source[start:brace] + function(source[brace:], "{"),
                [src / ("locgrn_fsz_real.c" if args.real else "locgrn_fsz.c"), accumulator], args.write)
    files.append(accumulator)
    provenance = "; ".join(f"{p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files)
    family = ("actual C scalar FSZ GreenFunc1/2/1_fsz2/2_fsz2_real; real input arrays; "
              "real accumulator discards coefficient imaginary part per C compound assignment; "
              if args.real else "actual C FSZ GreenFunc1/2/1_fsz2/2_fsz2; real and complex input arrays; ")
    output = ("# " + family +
              "zero/nonzero real projection; no RBM; MPI_COMM_SELF only; "
              "Apple clang 17 -O0 -ffp-contract=off; " + provenance + "\n")
    original = [line for line in (root / "tests/fixtures/interall/green_fsz.txt").read_text().splitlines()
                if not line.startswith("#")]
    checked_one = checked_two = models = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-fsz-green-") as directory:
        exe = Path(directory) / "probe"
        flags = ["-DFSZ_REAL=1"] if args.real else []
        subprocess.run(["cc", "-O0", "-ffp-contract=off"] + flags + [str(root / "c_toolbox/fsz_green.c"),
                        "-lm", "-o", str(exe)], check=True)
        for case in range(6):
            rows = original[case*13:(case+1)*13]
            if args.real and rows[0] != "0":
                continue
            for projection in ((0.0, 0.0), (0.125, -0.2)):
                parameters = " ".join(map(bits, (projection[0], 0.0, projection[1], 0.0)))
                header = rows[:6] + [parameters] + rows[6:10]
                payload = "\n".join(header[1:]) + "\n"
                result = subprocess.run([str(exe)], input=payload, text=True, capture_output=True,
                                        check=True, timeout=5)
                expected = result.stdout.splitlines()
                assert len(expected) == 64 + 4096 + 1
                output += "\n".join(header) + "\n"
                output += " ".join(expected[:64]) + "\n"
                output += " ".join(expected[64:64+4096]) + "\n"
                output += expected[-1] + "\n"
                checked_one += 64
                checked_two += 4096
                models += 1
    target = root / "tests/fixtures/interall" / ("c_fsz_real_green.txt" if args.real else "c_fsz_green.txt")
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, "Native C FSZ Green fixture changed"
    print(f"{checked_one} native C FSZ one-body and {checked_two} two-body operators, "
          f"{models} ordered InterAll sums with duplicates passed")


if __name__ == "__main__":
    main()
