#!/usr/bin/env python3
"""Compile the actual C count conversion and QP kernels for #45 contracts."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize


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
    materialize(root, "projection_qp_upstream.inc",
                function(qp, "void InitQPWeight()") + "\n" + function(qp, "void UpdateQPWeight()"),
                [src / "qp.c"], write)
    materialize(root, "projection_count_conversion.inc", read_count + "\n" + boundary + sizes,
                [src / "readdef.c"], write)
    probe_source = root / "c_toolbox/projection_count.c"
    with tempfile.TemporaryDirectory(prefix="mvmc-c-projection-") as tmp:
        tmp = Path(tmp)
        subprocess.run(["cc", "-O0", "-ffp-contract=off", "-include", "math.h", "-I", str(src / "include"),
                        str(probe_source), str(src / "gauleg.c"),
                        str(src / "legendrepoly.c"), "-lm", "-o", str(tmp / "probe")], check=True)
        actual = subprocess.check_output([str(tmp / "probe")], text=True)
    checksums = "; ".join(f"{name} sha256={hashlib.sha256((src / name).read_bytes()).hexdigest()}"
                          for name in ("readdef.c", "qp.c"))
    actual = f"# mVMC-1.3.0 snapshot; actual C readdef conversion and QP kernels; NSPGaussLeg=1; {checksums}\n" + "\n".join(line.rstrip() for line in actual.splitlines()) + "\n"
    target = root / "tests/fixtures/projection_count/c_contracts.txt"
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != actual:
            target.write_text(actual)
    else:
        assert actual == target.read_text(), "C projection contract changed"
    print("10 C projection/boundary/OptTrans cases passed")


if __name__ == "__main__":
    main()
