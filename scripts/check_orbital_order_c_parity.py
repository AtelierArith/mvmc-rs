#!/usr/bin/env python3
"""Compile C's filename registry and AP/P readers, independent of MPI."""
import argparse
import hashlib
import itertools
import re
from pathlib import Path
import subprocess
import tempfile

from check_projection_count_c_parity import function
from c_toolbox import materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader = (src / "readdef.c").read_text()
    # Retain the upstream keyword iteration and empty-slot handling verbatim.
    start = reader.index("    for (iKWidx = 0; iKWidx < KWIdxInt_end; iKWidx++) {", reader.index("GetInfoFromModPara(bufInt, bufDouble)"))
    end = reader.index("      fprintf(stdout,", start)
    iteration = reader[start:end]
    header_start = reader.index("          case KWOrbital:")
    header_end = reader.index("          case KWOrbitalGeneral:", header_start)
    header_cases = reader[header_start:header_end]
    bodies = []
    for signature in ("int ReadDefFileError(", "int CheckWords(\n", "int CheckKW(\n",
                      "int GetFileName(\n", "int CheckSite(\n", "int CheckPairSite(\n",
                      "int GetInfoOpt(FILE", "int GetInfoOptOrbitalParalell(FILE",
                      "char *ReadBuffIntCmpFlg(FILE", "int GetInfoOrbitalAntiParallel(FILE",
                      "int GetInfoOrbitalParallel(FILE"):
        # Skip forward declarations without altering the function body.
        definition = next(match.start() for match in re.finditer(re.escape(signature), reader)
                          if ";" not in reader[match.start():reader.index("{", match.start())])
        bodies.append(function(reader[definition:], signature))
    materialize(root, "orbital_readdef_upstream.inc", "\n".join(bodies), [src / "readdef.c"], args.write)
    materialize(root, "orbital_keyword_loop.inc", iteration, [src / "readdef.c"], args.write)
    materialize(root, "orbital_ap_headers.inc", header_cases, [src / "readdef.c"], args.write)
    probe_source = root / "c_toolbox/orbital_order.c"
    cases = {}
    with tempfile.TemporaryDirectory(prefix="mvmc-c-orbital-order-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", "-I", str(src / "include"), str(probe_source),
                        "-o", str(tmp / "probe")], check=True)
        subprocess.run([str(tmp / "probe"), "complex"], check=True)
        def definition(header, width, rows):
            return f"===\n{header} {width}\nComplexType 0\n===\n===\n{rows}"
        (tmp / "ap.def").write_text(definition("NOrbitalIdx", 7,
            "0 0 0 -1\n0 1 1 1\n1 0 1 -1\n1 1 0 1\n" +
            "".join(f"{i} {int(i % 3 != 0)}\n" for i in range(7))))
        (tmp / "p.def").write_text(definition("NOrbitalParallel", 3,
            "0 1 0 -1\n0 1\n1 0\n2 1\n"))
        kinds = ["ModPara", "LocSpin", "OrbitalAntiParallel", "OrbitalParallel", "TransSym"]
        paths = ["modpara.def", "locspin.def", "ap.def", "p.def", "qp.def"]
        for alias in ("Orbital", "OrbitalAntiParallel"):
            for order in itertools.permutations(range(5)):
                (tmp / "namelist.def").write_text("".join(
                    f"{alias if i == 2 else kinds[i]} {tmp / paths[i]}\n" for i in order))
                for boundary in (0, 1):
                    actual = subprocess.check_output([str(tmp / "probe"),
                        str(tmp / "namelist.def"), str(boundary)], text=True)
                    actual = "\n".join(line.rstrip() for line in actual.splitlines()) + "\n"
                    key = (alias, boundary)
                    if key in cases:
                        assert actual == cases[key], (key, order)
                    cases[key] = actual
    checksums = "; ".join(f"{name} sha256={hashlib.sha256((src / name).read_bytes()).hexdigest()}"
                          for name in ("readdef.c", "include/readdef.h"))
    actual = f"# mVMC-1.3.0 actual C filename registry, keyword iteration and AP/P readers; {checksums}\n"
    for (alias, boundary), result in cases.items():
        actual += f"{alias} {boundary}\n{result}"
    target = root / "tests/fixtures/orbital_general/c_order.txt"
    if args.write:
        if not target.exists() or target.read_text() != actual:
            target.write_text(actual)
    else:
        assert actual == target.read_text(), "C orbital order contract changed"
    print("480 C namelist permutations/aliases/boundaries passed")


if __name__ == "__main__":
    main()
