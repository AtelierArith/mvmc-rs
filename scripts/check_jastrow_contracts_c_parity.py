#!/usr/bin/env python3
"""Optional actual-C ordered Jastrow mapping/header/raw-flag oracle."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function


def definition(width, complex_flag, maps, flags, label="NJastrowIdx"):
    return f"===\n{label} {width}\nComplexType {complex_flag}\n===\n===\n{maps}{flags}"


def cases(root):
    for nsite in (2, 3, 6):
        width = nsite + 3
        rows = [f"{i} {j} {(2*i+j)%3}\n" for i in range(nsite)
                for j in range(nsite) if i != j]
        raw = [-2, -1, 0, 1, 2, 3]
        flags = [f"{99-k} {raw[k % 6]}\n" for k in range(width)]
        for complex_flag in (0, 1, 2):
            prefix = f"sites{nsite}_complex{complex_flag}"
            for name, maps, opts in (
                ("complete", "".join(rows), "".join(flags)),
                ("reordered", "".join(reversed(rows)), "".join(flags)),
                ("duplicate_pairs", "".join(f"0 1 {k%width}\n" for k in range(len(rows))), "".join(flags)),
                ("all_whitespace", "\n\t".join(" ".join(rows).split())+"\n", "\v".join(" ".join(flags).split())+"\n"),
                ("folded", " ".join(rows), " ".join(flags)),
                ("short_mapping", "".join(rows[:-1]), ""),
                ("extra_mapping", "".join(rows)+rows[0], "".join(flags)),
                ("short_flags", "".join(rows), "".join(flags[:-1])),
                ("extra_flags", "".join(rows), "".join(flags)+"0 1\n"),
                ("no_flags", "".join(rows), ""),
                ("diagonal", "0 0 0\n"+"".join(rows[1:]), "".join(flags)),
                ("bad_site", f"{nsite} 0 0\n"+"".join(rows[1:]), "".join(flags)),
                ("duplicate_flag_labels", "".join(rows), "".join(f"-7 {raw[k%6]}\n" for k in range(width))),
            ):
                yield prefix+"_"+name, nsite, definition(width, complex_flag, maps, opts)
            yield prefix+"_ignored_header_label", nsite, definition(width, complex_flag, "".join(rows), "".join(flags), "DeclaredWidth")
            yield prefix+"_zero_header", nsite, definition(0, complex_flag, "".join(rows), "".join(flags))

    yield "fixture_three_complete", 3, (root / "tests/fixtures/jastrow/c_three.def").read_text()
    for family in ("dh2", "dh4", "rbm"):
        yield f"historical_{family}_upper_only", 3, (root / f"tests/fixtures/{family}/jast.def").read_text()
    for case in ("heisenberg_chain_real", "heisenberg_chain_cmp", "heisenberg_chain_fsz", "hubbard_chain_real"):
        yield "historical_"+case, 6, (root / f"extern/Julia-mVMC/examples/inputs/{case}/jastrowidx.def").read_text()
    yield "historical_pairhop_fsz", 6, (root / "extern/Julia-mVMC/test/integration/reference/hubbard_chain_pairhop_fsz/inputs/jastrowidx.def").read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    originals = [root / f"tests/fixtures/{family}/jast.def" for family in ("dh2", "dh4", "rbm")]
    assert len({p.read_text() for p in originals}) == 1
    lines = originals[0].read_text().splitlines()
    assert len(lines) == 11 and lines[1].split()[1] == "3"
    rows = lines[5:8]
    reverse = [f"{row.split()[1]} {row.split()[0]} {row.split()[2]}" for row in rows]
    complete = "\n".join(lines[:5]+rows+reverse+lines[8:])+"\n"
    target = root / "tests/fixtures/jastrow/c_three.def"
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != complete: target.write_text(complete)
    else:
        assert target.read_text() == complete, "historical Jastrow replacement changed"
    source = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC/readdef.c"
    reader = source.read_text()
    body = "\n".join(function(reader, signature) for signature in (
        "int ReadDefFileError(", "int CheckSite(\n", "int CheckPairSite(\n",
        "int GetInfoOpt(FILE", "char *ReadBuffIntCmpFlg(FILE", "int GetInfoJastrow(FILE"))
    materialize(root, "jastrow_contracts_upstream.inc", body, [source], args.write)
    output = f"# actual C directional Jastrow reader; untouched index=-1 and imaginary=0 sentinels; Apple clang 17 -O0; readdef.c sha256={hashlib.sha256(source.read_bytes()).hexdigest()}\n"
    projection = source.with_name("projection.c")
    projection_body = "\n".join(function(projection.read_text(), signature) for signature in (
        "void MakeProjCnt(int", "void UpdateProjCnt(const int"))
    materialize(root, "jastrow_projection_upstream.inc", projection_body, [projection], args.write)
    kernels = f"# actual C MakeProjCnt/UpdateProjCnt; fully assigned tables only; Gutzwiller indices site%2; Apple clang 17 -O0; projection.c sha256={hashlib.sha256(projection.read_bytes()).hexdigest()}; readdef.c sha256={hashlib.sha256(source.read_bytes()).hexdigest()}\n"
    kernel_cases = kernel_records = 0
    accepted = rejected = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-jastrow-contracts-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", str(root / "c_toolbox/jastrow_contracts.c"), "-o", str(tmp / "probe")], check=True)
        subprocess.run(["cc", "-O0", str(root / "c_toolbox/jastrow_projection.c"), "-o", str(tmp / "kernels")], check=True)
        for name, nsite, payload in cases(root):
            (tmp / "jast.def").write_text(payload)
            result = subprocess.run([str(tmp / "probe"), str(nsite), str(tmp / "jast.def")],
                                    capture_output=True, text=True, check=True, timeout=5)
            lines = result.stdout.splitlines()
            header = " ".join(lines[0].split())
            ok = header.split()[-1] == "0"
            accepted += ok
            rejected += not ok
            output += f"{name} {nsite} {header}\n{payload.replace(chr(10), '|')}\n"
            output += (" ".join(lines[1].split()) if ok else "-")+"\n"
            output += (" ".join(lines[2].split()) if ok else "-")+"\n"
            if ok:
                matrix = list(map(int, lines[1].split()))
                width = int(header.split()[1])
                complete = all(0 <= matrix[i*nsite+j] < width
                               for i in range(nsite) for j in range(nsite) if i != j)
                if complete:
                    result = subprocess.run([str(tmp / "kernels"), str(nsite), str(tmp / "jast.def")],
                                            capture_output=True, text=True, check=True, timeout=5)
                    values = [" ".join(line.split()) for line in result.stdout.splitlines()]
                    kernels += f"{name} {nsite} {width} {len(values)}\n{payload.replace(chr(10), '|')}\n"+"|".join(values)+"\n"
                    kernel_cases += 1
                    kernel_records += len(values)
    target = root / "tests/fixtures/jastrow/c_reader_contracts.txt"
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != output: target.write_text(output)
    else:
        assert target.read_text() == output, "C Jastrow contracts changed"
    target = root / "tests/fixtures/jastrow/c_projection_counts.txt"
    if args.write:
        if not target.exists() or target.read_text() != kernels: target.write_text(kernels)
    else:
        assert target.read_text() == kernels, "C Jastrow projection counts changed"
    print(f"{accepted+rejected} C Jastrow cases passed ({accepted} accepted / {rejected} rejected); {kernel_records} native projection cases across {kernel_cases} complete tables")


if __name__ == "__main__":
    main()
