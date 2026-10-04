#!/usr/bin/env python3
"""Optional C RBM/OptTrans/flag audits; no dependency from Rust tests."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from c_toolbox import materialize
from check_projection_count_c_parity import function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC/readdef.c"
    reader = src.read_text()
    bodies = []
    for signature in ("int ReadDefFileError(", "int CheckSite(\n", "int GetInfoOpt(FILE",
                      "char *ReadBuffIntCmpFlg(FILE", "int GetInfoRBM_Layer(FILE",
                      "int GetInfoOptTrans(FILE", "int GetInfoOptOrbitalParalell(FILE"):
        start = next(match.start() for match in re.finditer(re.escape(signature), reader)
                     if ";" not in reader[match.start():reader.index("{", match.start())])
        bodies.append(function(reader[start:], signature))
    materialize(root, "reader_audit_upstream.inc", "\n".join(bodies), [src], args.write)
    expected = {
        "rbm_header": "header_ok=1 declared=97 complex=0\n"
                      "reader_status=0 opt_count=97 mapped=0,2,0\n",
        "opttrans_activation": "enabled=0 status=0 opt_count=0 weights=-7,-7 defined_flag_writes=\n"
                               "enabled=1 status=0 opt_count=2 weights=0.5,0.75 defined_flag_writes=3:1,4:1,\n",
        "orbital_flags": "AP row-order flags: 0 0 1 1\n"
                         "P aggregate-complex=2 flags: 1 2 1 2\n",
    }
    with tempfile.TemporaryDirectory(prefix="mvmc-c-reader-audits-") as directory:
        for name, result in expected.items():
            exe = Path(directory) / name
            subprocess.run(["cc", "-O0", str(root / "c_toolbox" / f"{name}.c"),
                            "-o", str(exe)], check=True)
            actual = subprocess.check_output([str(exe)], text=True)
            assert actual == result, f"C reader audit changed: {name}"
            print(actual, end="")
    print("3 C reader audits passed")


if __name__ == "__main__":
    main()
