#!/usr/bin/env -S uv run --no-project
"""Optional actual-C complex Green kernels and ordered InterAll accumulator.

Historical Julia arrays are inputs; native C supplies independent expectations.
One-process MPI_COMM_SELF plumbing, no RBM or full sampling/SR claim.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize, add_native_platform_argument, native_platform, native_compiler, native_provenance, native_target
from check_general_orbital_c_parity import function
from check_interall_real_c_parity import bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    add_native_platform_argument(parser)
    args = parser.parse_args()
    kind = native_platform(args.platform)
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    selected = {
        "projection.c": ["inline double ProjRatio(", "void UpdateProjCnt("],
        "pfupdate.c": ["void CalculateNewPfM("],
        "pfupdate_two_fcmp.c": ["void CalculateNewPfMTwo_fcmp(", "void calculateNewPfMTwo_child_fcmp("],
        "qp.c": ["double complex CalculateIP_fcmp("],
        "locgrn.c": ["double complex GreenFunc1(", "double complex GreenFunc2("],
    }
    files = [src / name for name in selected]
    body = "\n".join(function(path.read_text(), sig)
                     for path in files for sig in selected[path.name])
    materialize(root, "interall_complex_upstream.inc", body, files, args.write)
    accumulator = src / "calham.c"
    source = accumulator.read_text()
    start = source.index("for(idx=0;idx<NInterAll;idx++)")
    brace = source.index("{", start)
    materialize(root, "interall_complex_accumulator_upstream.inc",
                source[start:brace] + function(source[brace:], "{"),
                [src / "locgrn.c", accumulator], args.write)
    start = source.index("for(idx=0;idx<NPairHopping;idx++)")
    brace = source.index("{", start)
    materialize(root, "pairhop_complex_accumulator_upstream.inc",
                source[start:brace] + function(source[brace:], "{"),
                [src / "locgrn.c", accumulator], args.write)
    files.append(accumulator)
    provenance = "; ".join(f"{p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files)
    output = ("# actual C complex GreenFunc1/2, projection, Pfaffian update, overlap and serial InterAll loop; "
              "historical Julia wavefunction inputs; no RBM; MPI_COMM_SELF only; "
              "Apple clang 17 -O0 -ffp-contract=off; " + provenance + "\n")
    if kind == "linux-gnu":
        output = output.replace("Apple clang 17 -O0 -ffp-contract=off", native_provenance(kind))
    original = [line for line in (root / "tests/fixtures/interall/green_normal.txt").read_text().splitlines()
                if not line.startswith("#")]
    checked = models = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-interall-complex-") as directory:
        exe = Path(directory) / "probe"
        subprocess.run(native_compiler() + ["-O0", "-ffp-contract=off", str(root / "c_toolbox/interall_complex.c"),
                        "-lm", "-o", str(exe)], check=True)
        for case in range(4):
            rows = original[case*1034:(case+1)*1034]
            if rows[0] != "1":
                continue
            for projection in ((0.0, 0.0), (0.125, -0.2)):
                parameters = " ".join(map(bits, (projection[0], 0.0, projection[1], 0.0)))
                header = rows[:5] + [parameters] + rows[5:9]
                operators = []
                coefficients = [1e16, 0.3, -1e16, -0.7, 1/3, 1e-17, -0.125]
                for k, row in enumerate(rows[10:]):
                    operators.append(" ".join(row.split()[:6] + [bits(coefficients[k%len(coefficients)]),
                                                             bits((k%11-5)/16)]))
                payload = "\n".join(rows[1:5] + [parameters] + rows[5:9] + operators) + "\n"
                result = subprocess.run([str(exe)], input=payload, text=True, capture_output=True,
                                        check=True, timeout=5)
                expected = result.stdout.splitlines()
                assert len(expected) == 1026
                output += "\n".join(header) + "\n"
                output += "\n".join(row + " " + value for row, value in zip(operators, expected)) + "\n"
                output += "\n".join(expected[-2:]) + "\n"
                checked += 1024
                models += 1
    target = native_target(root, "c_complex_green", kind)
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, "Native C complex Green/InterAll fixture changed"
    print(f"{checked} native C complex Green operators, {models} ordered InterAll sums and {models} PairHop sums passed")


if __name__ == "__main__":
    main()
