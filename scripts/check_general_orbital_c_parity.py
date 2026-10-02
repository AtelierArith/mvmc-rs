#!/usr/bin/env python3
"""Optional actual-C six-column General reader/FSZ oracle (no Cargo dependency)."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

from c_toolbox import materialize
from check_orbital_contracts_c_parity import definition


def function(source, signature):
    """Extract verbatim bodies while ignoring braces in C comments/strings."""
    start = next(match.start() for match in re.finditer(re.escape(signature), source)
                 if ";" not in source[match.start():source.index("{", match.start())])
    brace = source.index("{", start)
    depth = 0
    tokens = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    for token in tokens.finditer(source, brace):
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                return source[start:token.end()]
    raise ValueError(f"unterminated function: {signature}")


def cases(root):
    for nsite in (2, 3):
        for complex_flag in (0, 1):
            width = 9 if nsite == 2 else 19
            pairs = [(i, j) for i in range(2*nsite) for j in range(i+1, 2*nsite)]
            rows = [f"{i%nsite} {i//nsite} {j%nsite} {j//nsite} {k%4} {(-1 if k%2 else 1)}\n"
                    for k, (i, j) in enumerate(pairs)]
            flags = [f"{k} {k%2}\n" for k in range(width)]
            variants = {
                "complete": (rows, flags, width),
                "reordered": (list(reversed(rows)), flags, width),
                "duplicate_mapping": (rows[:1]+rows[:-1], flags, width),
                "four_column": ([" ".join(row.split()[:4])+"\n" for row in rows], flags, width),
                "five_column": ([" ".join(row.split()[:5])+"\n" for row in rows], flags, width),
                "carried_index_sign": ([row if k%3==0 else " ".join(row.split()[:4 if k%3==1 else 5])+"\n" for k,row in enumerate(rows)], flags, width),
                "short_mapping": (rows[:-1], [], width),
                "empty_mapping": ([], [], width),
                "extra_mapping": (rows + rows[:1], flags, width),
                "short_flags": (rows, flags[:-1], width),
                "empty_flags": (rows, [], width),
                "extra_flags": (rows, flags + ["999 1\n"], width),
                "zero_width": (rows, flags, 0),
                "folded_flags": (rows, [" ".join(flags)], width),
                "split_flags": (rows, ["\n".join(" ".join(flags).split())+"\n"], width),
                "duplicate_labels": (rows, [f"99 {k%2}\n" for k in range(width)], width),
                "reversed_labels": (rows, [f"{width-k-1} {k%2}\n" for k in range(width)], width),
                "bad_site": ([f"{nsite} 0 0 1 0 1\n"] + rows[1:], flags, width),
                "lower_triangle": (["1 0 0 0 0 1\n"] + rows[1:], flags, width),
                "diagonal": (["0 0 0 0 0 1\n"] + rows[1:], flags, width),
                "integer_sign": ([row.rsplit(" ", 1)[0]+" 3\n" for row in rows], flags, width),
                "zero_sign": ([row.rsplit(" ", 1)[0]+" 0\n" for row in rows], flags, width),
            }
            for anti in (0, 1):
                for name, (maps, opts, count) in variants.items():
                    yield f"g{nsite}_complex{complex_flag}_anti{anti}_{name}", nsite, anti, definition(
                        count, "".join(maps), "".join(opts), complex_flag=complex_flag)
    for name, nsite in (("general_three", 3), ("general_three_real", 3),
                        ("general_sparse_three", 3), ("general_heisenberg_six", 6)):
        for anti in (0, 1):
            yield name+f"_anti{anti}", nsite, anti, (root / f"tests/fixtures/c_orbital_inputs/{name}.def").read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader, fsz = (src / "readdef.c").read_text(), (src / "slater_fsz.c").read_text()
    common = [function(reader, signature) for signature in (
        "int ReadDefFileError(", "int CheckSite(\n", "int CheckPairSite(\n",
        "int GetInfoOpt(FILE", "int GetInfoOptOrbitalParalell(FILE",
        "char *ReadBuffIntCmpFlg(FILE", "int GetInfoOrbitalAntiParallel(FILE",
        "int GetInfoOrbitalParallel(FILE")]
    materialize(root, "orbital_contracts_upstream.inc", "\n".join(common), [src / "readdef.c"], args.write)
    materialize(root, "general_reader_upstream.inc", function(reader, "int GetInfoOrbitalGeneral(FILE"), [src / "readdef.c"], args.write)
    # Skip forward declarations when extracting the two complete function bodies.
    functions = []
    for signature in ("void UpdateSlaterElm_fsz() {", "void SlaterElmDiff_fsz(double complex *srOptO, const double complex ip, int *eleIdx,int *eleSpn) {"):
        functions.append(function(fsz, signature))
    materialize(root, "general_fsz_upstream.inc", "\n".join(functions), [src / "slater_fsz.c"], args.write)
    provenance = "; ".join(f"{name} sha256={hashlib.sha256((src/name).read_bytes()).hexdigest()}" for name in ("readdef.c", "slater_fsz.c"))
    readers = f"# actual C General reader; Apple clang 17 -O0 -ffp-contract=off; {provenance}\n"
    kernels = f"# actual C General FSZ kernels; explicit zero diagonals; two QP translations, one identity OptTrans; {provenance}\n"
    count = accepted = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-general-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", "-ffp-contract=off", str(root / "c_toolbox/general_orbital.c"), "-o", str(tmp / "probe")], check=True)
        for name, nsite, anti, payload in cases(root):
            (tmp / "general.def").write_text(payload)
            result = subprocess.run([str(tmp / "probe"), str(nsite), str(anti), str(tmp / "general.def")], capture_output=True, text=True, check=True, timeout=5)
            rows = result.stdout.splitlines()
            header = rows[0].split()
            valid = header[0] == "1" and header[3] == "0"
            count += 1
            accepted += int(valid)
            readers += f"{name} {nsite} {anti} {' '.join(header)}\n{payload.replace(chr(10), '|')}\n"
            readers += "\n".join(" ".join(row.split()) for row in (rows[1:4] if valid else ["", "", ""])) + "\n"
            if valid:
                assert len(rows) == 10
                kernels += f"{name} {nsite} {anti} {header[1]} {header[2]}\n{payload.replace(chr(10), '|')}\n"
                kernels += "\n".join(" ".join(row.split()) for row in rows[4:]) + "\n"
    for name, content in (("c_general_reader.txt", readers), ("c_general_kernels.txt", kernels)):
        path = root / "tests/fixtures/orbital_general" / name
        if args.write:
            if not path.exists() or path.read_text() != content:
                path.write_text(content)
        else:
            assert path.read_text() == content, f"C General fixture changed: {name}"
    print(f"{count} C General reader cases ({accepted} accepted), {accepted} FSZ kernel cases passed")


if __name__ == "__main__":
    main()
