#!/usr/bin/env python3
"""Optional native C complete-record and scalar-conversion oracle for #28."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_projection_count_c_parity import function


def record(count, number, complex_mode=False, ignored_nonfinite=False):
    header = "NaN Inf -Inf 0 -0 0.5" if ignored_nonfinite else "0 1 2 3 4 5"
    triples = []
    for index in range(count):
        value = number * (index+1)/16
        gradient = "Inf" if ignored_nonfinite else "99"
        triples.append(f"{value} {-value/2 if complex_mode else 0.0} {gradient}")
    return header + " " + " ".join(triples)


def cases(root):
    yield "single_real", (0, 0, 13, 0), record(13, 1)
    yield "two_real", (0, 0, 13, 0), record(13, 1) + " " + record(13, 2)
    yield "three_complex", (0, 0, 13, 0), " ".join(record(13, i, True) for i in (1, 2, 3))
    yield "all_factors", (37, 27, 13, 2), " ".join(record(79, i, True) for i in (1, 2, 3))
    yield "all_ignored_nonfinite", (37, 27, 13, 2), record(79, 1, True, True)
    yield "empty", (0, 0, 13, 0), ""
    yield "no_parameters", (0, 0, 0, 0), "0 1 2 3 4 5 6 7 8 9 10 11"
    yield "ignored_nonfinite", (0, 0, 13, 0), record(13, 1, True, True)
    zeros = [("-0", "0"), ("0", "-0"), ("-0", "-0"), ("-1", "-0"), ("0", "0")]
    yield "signed_zero", (0, 0, 13, 0), "0 1 2 3 4 5 " + " ".join(
        f"{zeros[i % len(zeros)][0]} {zeros[i % len(zeros)][1]} 99" for i in range(13))
    numbers = [("1e-999", "-1e-999"), ("0x1.8p+1", "-0x1p-1"), ("5e-324", "0"),
               ("-1e999", "0"), ("0", "Inf"), ("NaN", "0"), ("1", "-Inf")]
    special = "0 1 2 3 4 5 " + " ".join(
        f"{numbers[i % len(numbers)][0]} {numbers[i % len(numbers)][1]} 99" for i in range(13))
    yield "scalar_conversion", (0, 0, 13, 0), special
    yield "overwritten_nonfinite", (0, 0, 13, 0), special + " " + record(13, 2, True)

    # Legacy Julia definitions supply these explicit C kernel dimensions. This
    # checks ReadInitParameter, not C acceptance of their mapping/flag files.
    for mode, dims in (("valid", (0, 0, 0, 2)), ("layout", (37, 27, 4, 2)),
                       ("qp_only", (37, 27, 4, 0)), ("empty_opt", (37, 27, 4, 0)),
                       ("short_opt", (37, 27, 4, 1)), ("long_opt", (37, 27, 4, 3))):
        for field in ("nan_gradient", "nan_parameter", "overflow", "underflow"):
            payload = (root / f"tests/fixtures/opttrans/load_{mode}_{field}.def").read_text()
            yield f"legacy_opt_{mode}_{field}", dims, " ".join(payload.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src"
    materialize(root, "initial_records_upstream.inc",
                function((src / "mVMC/parameter.c").read_text(), "int ReadInitParameter(char *initFile)"),
                [src / "mVMC/parameter.c"], args.write)
    rows = []
    with tempfile.TemporaryDirectory(prefix="mvmc-c-initial-records-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", "-ffp-contract=off", "-DMEXP=19937", "-I", str(src / "sfmt"),
                        str(root / "c_toolbox/initial_records.c"), str(src / "sfmt/SFMT.c"),
                        "-lm", "-o", str(tmp / "probe")], check=True)
        for name, dims, payload in cases(root):
            (tmp / "initial.def").write_text(payload)
            output = subprocess.check_output([str(tmp / "probe"), str(tmp / "initial.def"),
                                               *map(str, dims)], text=True, timeout=5)
            rows.append(f"{name} {' '.join(map(str, dims))}\n{payload}\n" + output)
    checksums = "; ".join(f"{name} sha256={hashlib.sha256((src / name).read_bytes()).hexdigest()}"
                          for name in ("mVMC/parameter.c", "sfmt/SFMT.c", "sfmt/SFMT.h", "sfmt/SFMT-params19937.h"))
    result = f"# actual mVMC-1.3.0 ReadInitParameter and native SFMT; Apple clang 17; -O0 -ffp-contract=off -DMEXP=19937; {checksums}\n"
    result += "\n".join(line.rstrip() for line in "".join(rows).splitlines()) + "\n"
    target = root / "tests/fixtures/initial_records/c_records.txt"
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != result:
            target.write_text(result)
    else:
        assert target.read_text() == result, "C complete-record contract changed"
    print("35 C complete-record/scalar-conversion cases passed")


if __name__ == "__main__":
    main()
