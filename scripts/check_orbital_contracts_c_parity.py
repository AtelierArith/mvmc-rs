#!/usr/bin/env python3
"""Optional actual-C AP/P header and complete-row input contract oracle."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

from c_toolbox import materialize
from check_projection_count_c_parity import function


def definition(width, mappings, flags, label="NOrbitalIdx", complex_flag=0):
    return f"===\n{label} {width}\nComplexType {complex_flag}\n===\n===\n" + mappings + flags


def cases(root):
    for mode in ("AP", "P"):
        for nsite in (2, 3):
            width = 7 if mode == "AP" else 3
            pairs = [(i, j) for i in range(nsite) for j in range(nsite)
                     if mode == "AP" or i < j]
            rows = [f"{i} {j} {k % 2} 1\n" for k, (i, j) in enumerate(pairs)]
            flags = [f"{k} {k % 2}\n" for k in range(width)]
            prefix = f"{mode.lower()}_{nsite}"
            yield prefix + "_complete", mode, nsite, definition(width, "".join(rows), "".join(flags))
            yield prefix + "_complex", mode, nsite, definition(width, "".join(rows), "".join(flags), complex_flag=1)
            yield prefix + "_short_mapping", mode, nsite, definition(width, "".join(rows[:-1]), "")
            yield prefix + "_empty_mapping", mode, nsite, definition(width, "", "")
            yield prefix + "_short_flags", mode, nsite, definition(width, "".join(rows), "".join(flags[:-1]))
            yield prefix + "_empty_flags", mode, nsite, definition(width, "".join(rows), "")
            yield prefix + "_extra_flags", mode, nsite, definition(width, "".join(rows), "".join(flags) + f"{width} 1\n")
            yield prefix + "_extra_mapping", mode, nsite, definition(width, "".join(rows) + rows[0], "".join(flags))
            yield prefix + "_zero_header", mode, nsite, definition(0, "".join(rows), "".join(flags))
            yield prefix + "_nonstandard_label", mode, nsite, definition(width, "".join(rows), "".join(flags), label="DeclaredWidth")
            yield prefix + "_folded_flags", mode, nsite, definition(width, "".join(rows), " ".join(s.strip() for s in flags) + "\n")
            yield prefix + "_split_flags", mode, nsite, definition(width, "".join(rows), "\n".join(" ".join(flags).split()) + "\n")
            yield prefix + "_flag_whitespace", mode, nsite, definition(width, "".join(rows), "\n\t" + "\n\t".join(flags))
            yield prefix + "_three_column", mode, nsite, definition(width, "".join(" ".join(s.split()[:3]) + "\n" for s in rows), "".join(flags))
            yield prefix + "_reordered_mapping", mode, nsite, definition(width, "".join(reversed(rows)), "".join(flags))
            yield prefix + "_duplicate_flag_labels", mode, nsite, definition(width, "".join(rows), "".join(f"99 {k % 2}\n" for k in range(width)))
            yield prefix + "_reversed_flag_labels", mode, nsite, definition(width, "".join(rows), "".join(f"{width - 1 - k} {k % 2}\n" for k in range(width)))
            yield prefix + "_bad_site", mode, nsite, definition(width, f"{nsite} 0 0 1\n" + "".join(rows[1:]), "".join(flags))
            if mode == "P":
                yield prefix + "_reversed_pair", mode, nsite, definition(width, "1 0 0 1\n" + "".join(rows[1:]), "".join(flags))
                yield prefix + "_diagonal", mode, nsite, definition(width, "0 0 0 1\n" + "".join(rows[1:]), "".join(flags))
    yield "ap_headerless", "AP", 2, "0 0 0 1\n1 0 1 1\n0 1 1 1\n1 1 0 1\n"
    yield "ap_illegal_header", "AP", 2, definition("illegal", "0 0 0 1\n", "")
    for name, mode, nsite in (("ap_three", "AP", 3), ("p_three", "P", 3),
                              ("ap_four", "AP", 4), ("ap_general_three", "AP", 3),
                              ("p_general_three", "P", 3)):
        yield f"fixture_{name}", mode, nsite, (root / f"tests/fixtures/c_orbital_inputs/{name}.def").read_text()
    for name, mode, nsite in (("dh2/orbital", "AP", 3), ("dh4/orbital", "AP", 3),
                              ("rbm/orbital", "AP", 3), ("dh4/parallel_orbital", "P", 3),
                              ("interall/orbital", "AP", 4), ("orbital_general/ap", "AP", 3),
                              ("orbital_general/parallel", "P", 3)):
        yield "historical_" + name.replace("/", "_"), mode, nsite, (root / f"tests/fixtures/{name}.def").read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader = (src / "readdef.c").read_text()
    bodies = []
    for signature in ("int ReadDefFileError(", "int CheckSite(\n", "int CheckPairSite(\n",
                      "int GetInfoOpt(FILE", "int GetInfoOptOrbitalParalell(FILE",
                      "char *ReadBuffIntCmpFlg(FILE", "int GetInfoOrbitalAntiParallel(FILE",
                      "int GetInfoOrbitalParallel(FILE"):
        start = next(m.start() for m in re.finditer(re.escape(signature), reader)
                     if ";" not in reader[m.start():reader.index("{", m.start())])
        bodies.append(function(reader[start:], signature))
    materialize(root, "orbital_contracts_upstream.inc", "\n".join(bodies), [src / "readdef.c"], args.write)
    output = f"# actual C AP/P header/row readers; Apple clang 17 -O0; readdef.c sha256={hashlib.sha256((src / 'readdef.c').read_bytes()).hexdigest()}\n"
    count = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-orbital-contracts-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", str(root / "c_toolbox/orbital_contracts.c"), "-o", str(tmp / "probe")], check=True)
        for name, mode, nsite, payload in cases(root):
            (tmp / "orbital.def").write_text(payload)
            result = subprocess.run([str(tmp / "probe"), mode, str(nsite), str(tmp / "orbital.def")],
                                    capture_output=True, text=True, check=True, timeout=5)
            output += f"{name} {mode} {nsite} {' '.join(result.stdout.split())}\n{payload.replace(chr(10), '|')}\n"
            count += 1
    target = root / "tests/fixtures/orbital_general/c_reader_contracts.txt"
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, "C orbital input contracts changed"
    print(f"{count} C AP/P input contract cases passed")


if __name__ == "__main__":
    main()
