#!/usr/bin/env -S uv run --no-project
"""Optional C complete RBM declarations/initialization/loading/SFMT oracle."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from numerical_comparison import compare_text
import math
from c_toolbox import materialize
from check_general_orbital_c_parity import function
from check_rbm_contracts_c_parity import geometry, definition, materialize_readers


def cases():
    raw = [-2, -1, 0, 1, 2, 3]
    for selected in range(12):
        widths = [0]*9
        if selected < 9:
            widths[selected] = selected+5
        elif selected == 9:
            widths = list(range(5, 14))
        elif selected == 11:
            widths[0] = 97
        definitions = []
        for section, width in enumerate(widths):
            if not width:
                definitions.append("-")
                continue
            maps = "".join(" ".join(map(str, coord+(k%3,)))+"\n"
                           for k, coord in enumerate(geometry(section, 3, 2)))
            flags = "".join(f"{-7} {raw[k%6]}\n" for k in range(width))
            definitions.append(definition(width, 0, maps, flags))
        for neurons in (0, -7, 10):
            for seed in (1, 11272):
                yield f"section{selected}_neurons{neurons}_seed{seed}", widths, definitions, neurons, seed, "init", "-"
        if selected in (3, 9, 10, 11):
            count = 6+sum(widths)
            payload = "0 0 0 0 0 0 "+" ".join(f"{(k+1)/8} {-((k+1)/16)} 99" for k in range(count))+"\n"
            later = "0 0 0 0 0 0 "+" ".join(f"{(k+17)/8} {-((k+17)/16)} 123" for k in range(count))+"\n"
            for mode, content in (("load", payload), ("last_record", payload+later), ("empty", "")):
                yield f"section{selected}_{mode}", widths, definitions, 10, 11272, mode, content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src"
    materialize_readers(root, src / "mVMC/readdef.c", args.write)
    parameter = src / "mVMC/parameter.c"
    materialize(root, "rbm_parameters_upstream.inc", "\n".join(function(parameter.read_text(), sig) for sig in (
        "void InitParameter()", "int ReadInitParameter(char *initFile)")), [parameter], args.write)
    hashes = "; ".join(f"{name} sha256={hashlib.sha256((src/name).read_bytes()).hexdigest()}"
                       for name in ("mVMC/readdef.c", "mVMC/parameter.c", "sfmt/SFMT.c", "sfmt/SFMT.h", "sfmt/SFMT-params19937.h"))
    output = f"# actual C RBM readers/InitParameter/ReadInitParameter/SFMT; real initializer; NProj=2 NSlater=4; Apple clang 17 -O0 -ffp-contract=off -DMEXP=19937; {hashes}\n"
    checked = 0
    initialized_rows = set()
    with tempfile.TemporaryDirectory(prefix="mvmc-c-rbm-parameters-") as directory:
        tmp = Path(directory)
        exe = tmp / "probe"
        subprocess.run(["cc", "-O0", "-ffp-contract=off", "-DMEXP=19937", "-I", str(src / "sfmt"),
                        str(root / "c_toolbox/rbm_parameters.c"), str(src / "sfmt/SFMT.c"), "-lm", "-o", str(exe)], check=True)
        for name, widths, definitions, neurons, seed, mode, payload in cases():
            files = []
            for section, content in enumerate(definitions):
                filename = tmp / f"section{section}.def"
                if content != "-": filename.write_text(content)
                files.append(str(filename) if content != "-" else "-")
            argv = [str(exe), "3", "2", str(neurons), str(seed)] + files
            if mode != "init":
                filename = tmp / "initial.def"
                filename.write_text(payload)
                argv.append(str(filename))
            result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=5)
            lines = [" ".join(line.split()) for line in result.stdout.splitlines()]
            assert len(lines) == 3
            output += f"{name} 3 2 {neurons} {seed} {mode} {' '.join(map(str,widths))}\n"
            output += "~".join(content.replace("\n", "|") for content in definitions)+"\n"
            output += (payload.replace("\n", "|") if mode != "init" else "-")+"\n"
            output += "\n".join(lines)+"\n"
            if mode == "init": initialized_rows.add(sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in output.splitlines())-2)
            checked += 1
    target = root / "tests/fixtures/rbm/c_parameters.txt"
    if args.write:
        if not target.exists() or target.read_text() != output: target.write_text(output)
    else:
        compare_text(output,target.read_text(),lambda r,c,t: (32*math.ulp(1.0),32*math.ulp(1.0)) if r in initialized_rows else None)
    print(f"{checked} native C RBM declaration/parameter/RNG cases passed")


if __name__ == "__main__":
    main()
