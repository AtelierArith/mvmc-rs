#!/usr/bin/env -S uv run --no-project
"""Optional actual-C nine-section RBM reader oracle; Cargo uses fixtures only."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function

NAMES = [f"{kind}RBM_{layer}" for layer in ("PhysLayer", "HiddenLayer", "PhysHidden")
         for kind in ("Charge", "Spin", "General")]


def geometry(section, sites, hidden):
    if section == 2:
        return [(i, spin) for spin in range(2) for i in range(sites)]
    if section < 6:
        return [(i,) for i in range(sites if section < 3 else hidden)]
    if section == 8:
        return [(i, spin, h) for spin in range(2) for i in range(sites) for h in range(hidden)]
    return [(i, h) for i in range(sites) for h in range(hidden)]


def definition(width, complex_flag, maps, flags, label="NParameter"):
    return f"===\n{label} {width}\nComplexType {complex_flag}\n===\n===\n{maps}{flags}"


def cases(root):
    raw = [-2, -1, 0, 1, 2, 3]
    for section, name in enumerate(NAMES):
        for sites, hidden in ((1, 1), (3, 2), (6, 4)):
            coords = geometry(section, sites, hidden)
            width = len(coords)+4
            rows = [" ".join(map(str, coord+(k % 3,)))+"\n" for k, coord in enumerate(coords)]
            for complex_flag in (-1, 0, 1, 2):
                flags = [f"{99-k} {raw[k % 6]}\n" for k in range(width)]
                prefix = f"{name}_sites{sites}_hidden{hidden}_complex{complex_flag}"
                for variant, maps, opts in (
                    ("complete", "".join(rows), "".join(flags)),
                    ("reordered", "".join(reversed(rows)), "".join(flags)),
                    ("duplicate_coordinates", "".join(" ".join(map(str, coords[0]+(k%width,)))+"\n" for k in range(len(rows))), "".join(flags)),
                    ("folded", " ".join(rows), " ".join(flags)),
                    ("all_whitespace", "\v".join(" ".join(rows).split())+"\n", "\f\t".join(" ".join(flags).split())+"\n"),
                    ("short_mapping", "".join(rows[:-1]), ""),
                    ("short_flags", "".join(rows), "".join(flags[:-1])),
                    ("extra_flags", "".join(rows), "".join(flags)+"0 1\n"),
                    ("missing_flags", "".join(rows), ""),
                    ("duplicate_flag_labels", "".join(rows), "".join(f"-7 {raw[k%6]}\n" for k in range(width))),
                ):
                    yield prefix+"_"+variant, section, sites, hidden, definition(width, complex_flag, maps, opts)
                yield prefix+"_ignored_header_label", section, sites, hidden, definition(width, complex_flag, "".join(rows), "".join(flags), "Width")
                yield prefix+"_zero_header", section, sites, hidden, definition(0, complex_flag, "".join(rows), "".join(flags))
    # Authoritative declaration 97 with only three mapped slots.
    yield "ChargeRBM_PhysLayer_declared97", 0, 3, 2, definition(97, 0, "0 0\n1 2\n2 0\n", "".join(f"{k} {raw[k%6]}\n" for k in range(97)))
    yield "historical_ChargeRBM_PhysLayer_incomplete97", 0, 3, 2, (root / "tests/fixtures/rbm/ChargeRBM_PhysLayer.def").read_text()
    for section, name in enumerate(NAMES):
        yield "complete_six_site_control_"+name, section, 6, 2, (root / "tests/fixtures/c_orbital_inputs/historical_binary_rbm" / f"{name}.def").read_text()


def materialize_readers(root, source, write):
    reader = source.read_text()
    bodies = "\n".join(function(reader, sig) for sig in (
        "int ReadDefFileError(", "int CheckSite(\n", "int GetInfoOpt(FILE",
        "char *ReadBuffIntCmpFlg(FILE", "int GetInfoRBM_Layer(FILE",
        "int GetInfoGeneralRBM_Layer(FILE", "int GetInfoRBM_PhysHidden(FILE",
        "int GetInfoGeneralRBM_PhysHidden(FILE"))
    materialize(root, "rbm_contracts_upstream.inc", bodies, [source], write)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC/readdef.c"
    materialize_readers(root, source, args.write)
    output = f"# actual C nine-section RBM readers; untouched index=-1 and imaginary=0 sentinels; Apple clang 17 -O0; readdef.c sha256={hashlib.sha256(source.read_bytes()).hexdigest()}\n"
    accepted = rejected = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-rbm-contracts-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", str(root / "c_toolbox/rbm_contracts.c"), "-o", str(tmp / "probe")], check=True)
        for name, section, sites, hidden, payload in cases(root):
            (tmp / "rbm.def").write_text(payload)
            result = subprocess.run([str(tmp / "probe"), str(section), str(sites), str(hidden), str(tmp / "rbm.def")],
                                    capture_output=True, text=True, check=True, timeout=5)
            lines = result.stdout.splitlines()
            ok = lines[0].split()[-1] == "0"
            accepted += ok
            rejected += not ok
            output += f"{name} {section} {sites} {hidden} {' '.join(lines[0].split())}\n{payload.replace(chr(10), '|')}\n"
            output += (" ".join(lines[1].split()) if ok else "-")+"\n"
            output += (" ".join(lines[2].split()) if ok else "-")+"\n"
    target = root / "tests/fixtures/rbm/c_reader_contracts.txt"
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, "C RBM contracts changed"
    print(f"{accepted+rejected} C RBM cases passed ({accepted} accepted / {rejected} rejected)")


if __name__ == "__main__":
    main()
