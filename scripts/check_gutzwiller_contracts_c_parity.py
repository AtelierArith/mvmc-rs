#!/usr/bin/env python3
"""Optional actual-C Gutzwiller complete-row/header/raw-flag oracle."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function


def definition(width, complex_flag, maps, flags, label="NGutzwillerIdx"):
    return f"===\n{label} {width}\nComplexType {complex_flag}\n===\n===\n{maps}{flags}"


def cases():
    for nsite in (1, 2, 3, 6):
        width = nsite + 2
        rows = [f"{k} {k % 2}\n" for k in range(nsite)]
        raw = [-2, -1, 0, 1, 2, 3]
        flags = [f"{99-k} {raw[k % 6]}\n" for k in range(width)]
        for complex_flag in (0, 1, 2):
            prefix = f"sites{nsite}_complex{complex_flag}"
            for name, maps, opts in (
                ("complete", "".join(rows), "".join(flags)),
                ("reordered", "".join(reversed(rows)), "".join(flags)),
                ("duplicate_sites", rows[0]*nsite, "".join(flags)),
                ("all_whitespace", "\n\t".join(" ".join(rows).split())+"\n", "\v".join(" ".join(flags).split())+"\n"),
                ("folded", " ".join(rows), " ".join(flags)),
                ("short_mapping", "".join(rows[:-1]), ""),
                ("extra_mapping", "".join(rows)+rows[0], "".join(flags)),
                ("short_flags", "".join(rows), "".join(flags[:-1])),
                ("extra_flags", "".join(rows), "".join(flags)+"0 1\n"),
                ("no_flags", "".join(rows), ""),
            ):
                yield prefix+"_"+name, nsite, definition(width, complex_flag, maps, opts)
            yield prefix+"_ignored_header_label", nsite, definition(width, complex_flag, "".join(rows), "".join(flags), "DeclaredWidth")
            yield prefix+"_zero_header", nsite, definition(0, complex_flag, "".join(rows), "".join(flags))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC/readdef.c"
    reader = source.read_text()
    body = "\n".join(function(reader, signature) for signature in (
        "int ReadDefFileError(", "int CheckSite(\n", "int GetInfoOpt(FILE",
        "char *ReadBuffIntCmpFlg(FILE", "int GetInfoGutzwiller(FILE"))
    materialize(root, "gutzwiller_contracts_upstream.inc", body, [source], args.write)
    output = f"# actual C Gutzwiller reader; untouched cells zeroed as sentinels; Apple clang 17 -O0; readdef.c sha256={hashlib.sha256(source.read_bytes()).hexdigest()}\n"
    accepted = rejected = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-gutzwiller-contracts-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", str(root / "c_toolbox/gutzwiller_contracts.c"), "-o", str(tmp / "probe")], check=True)
        for name, nsite, payload in cases():
            (tmp / "gutz.def").write_text(payload)
            result = subprocess.run([str(tmp / "probe"), str(nsite), str(tmp / "gutz.def")],
                                    capture_output=True, text=True, check=True, timeout=5)
            lines = result.stdout.splitlines()
            header = " ".join(lines[0].split())
            ok = header.split()[-1] == "0"
            accepted += ok
            rejected += not ok
            output += f"{name} {nsite} {header}\n{payload.replace(chr(10), '|')}\n"
            # Fixed four-line records make rejected cases explicit too.
            output += (" ".join(lines[1].split()) if ok else "-")+"\n"
            output += (" ".join(lines[2].split()) if ok else "-")+"\n"
    target = root / "tests/fixtures/gutzwiller/c_reader_contracts.txt"
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != output: target.write_text(output)
    else:
        assert target.read_text() == output, "C Gutzwiller contracts changed"
    print(f"{accepted+rejected} C Gutzwiller cases passed ({accepted} accepted / {rejected} rejected)")


if __name__ == "__main__":
    main()
