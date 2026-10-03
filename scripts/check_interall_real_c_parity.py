#!/usr/bin/env -S uv run --no-project
"""Optional actual-C fixed-Sz real Green kernels and InterAll accumulator.

Input wavefunction arrays come from historical Julia fixtures; all expected
Greens and energies are independently computed by extracted native C bodies.
MPI_COMM_SELF is supplied as single-process plumbing. No RBM, full C runner,
sampling, initialization or SR parity is claimed by this standalone probe.
"""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile

from c_toolbox import materialize, add_native_platform_argument, native_platform, native_compiler, native_provenance, native_target
from check_general_orbital_c_parity import function


def bits(value):
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


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
        "pfupdate_real.c": ["void CalculateNewPfM_real("],
        "pfupdate_two_real.c": ["void CalculateNewPfMTwo_real(", "void calculateNewPfMTwo_child_real("],
        "qp_real.c": ["double  CalculateIP_real("],
        "locgrn_real.c": ["double  GreenFunc1_real(", "double GreenFunc2_real("],
    }
    files = [src / name for name in selected]
    body = "\n".join(function(path.read_text(), sig)
                     for path in files for sig in selected[path.name])
    materialize(root, "interall_real_upstream.inc", body, files, args.write)
    accumulator = src / "calham_real.c"
    source = accumulator.read_text()
    start = source.index("for(idx=0;idx<NInterAll;idx++)")
    brace = source.index("{", start)
    materialize(root, "interall_real_accumulator_upstream.inc",
                source[start:brace] + function(source[brace:], "{"),
                [src / "locgrn_real.c", accumulator], args.write)
    files.append(accumulator)
    provenance = "; ".join(f"{p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files)
    output = ("# actual C real GreenFunc1/2, projection, Pfaffian update, overlap and serial InterAll loop; "
              "historical Julia wavefunction inputs; no RBM; MPI_COMM_SELF only; "
              "Apple clang 17 -O0 -ffp-contract=off; " + provenance + "\n")
    if kind == "linux-gnu":
        output = output.replace("Apple clang 17 -O0 -ffp-contract=off", native_provenance(kind))
    original = [line for line in (root / "tests/fixtures/interall/green_normal.txt").read_text().splitlines()
                if not line.startswith("#")]
    checked = models = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-interall-real-") as directory:
        exe = Path(directory) / "probe"
        subprocess.run(native_compiler() + ["-O0", "-ffp-contract=off", str(root / "c_toolbox/interall_real.c"),
                        "-lm", "-o", str(exe)], check=True)
        for case in range(4):
            rows = original[case*1034:(case+1)*1034]
            if rows[0] != "0":
                continue
            # Fixtures already serialize the Rust/C row-major kernel layout.
            slater = rows[5].split()[::2]
            inverse = rows[7].split()[::2]
            for projection in ((0.0, 0.0), (0.125, -0.2)):
                header = rows[:5] + [" ".join(map(bits, projection))] + rows[5:9]
                operators = []
                coefficients = [1e16, 0.3, -1e16, -0.7, 1/3, 1e-17, -0.125]
                for k, row in enumerate(rows[10:]):
                    ops = row.split()[:6]
                    operators.append(" ".join(ops + [bits(coefficients[k%len(coefficients)]), bits((k%11-5)/16)]))
                payload = "\n".join(rows[1:5] + [" ".join(map(bits, projection)), " ".join(slater),
                                               " ".join(rows[6].split()[::2]), " ".join(inverse),
                                               rows[8].split()[0]] + operators) + "\n"
                result = subprocess.run([str(exe)], input=payload, text=True, capture_output=True,
                                        check=True, timeout=5)
                expected = result.stdout.splitlines()
                assert len(expected) == 1025
                output += "\n".join(header) + "\n"
                output += "\n".join(row + " " + value for row, value in zip(operators, expected)) + "\n"
                output += expected[-1] + "\n"
                checked += 1024
                models += 1
    target = native_target(root, "c_real_green", kind)
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, "Native C real Green/InterAll fixture changed"
    print(f"{checked} native C real Green operators and {models} ordered InterAll sums passed")


if __name__ == "__main__":
    main()
