#!/usr/bin/env python3
"""Optional #367 oracle: C handling of a negative DSROptStepDt.

Extracts the authoritative normalization block (readdef.c), the SRinfo header
selection (initfile.c) and `stcOptInit` (stcopt_dposv.c) verbatim into
c_toolbox/negative_stepdt_upstream.inc, compiles c_toolbox/negative_stepdt.c
against system LAPACK, and (with --write) stores the results under
tests/fixtures/negative_stepdt/. Rust tests only read the fixture.
"""
import argparse
import hashlib
from pathlib import Path
import platform
import subprocess
import tempfile

from c_toolbox import materialize
from check_projection_count_c_parity import function

# (name, DSROptStepDt text, DSROptStaDel text)
CASES = [
    ("positive", "0.0125", "0.0"),
    ("negative", "-0.0125", "0.0"),
    ("negative_shifted", "-0.125", "0.02"),
    ("positive_shifted", "0.125", "0.02"),
    ("negative_zero", "-0.0", "0.0"),
]


def extract(reader, initfile, dposv):
    marker = "if (DSROptStepDt < 0) {"
    start = reader.index(marker)
    end = reader.index("Nsize = 2 * Ne;", start)
    block = reader[start:end].rstrip()
    normalize = ("/* Wrapper added by the toolbox: the block is verbatim from ReadDefFileNInt. */\n"
                 "static void NormalizeStepDt(int rank) {\n  " + block + "\n}")
    header_start = initfile.index("if(SRFlag == 0){")
    header_end = initfile.index('sprintf(fileName, "%s_out_%03d.dat"', header_start)
    header = ("/* Verbatim initfile.c SRinfo header selection (printed by the toolbox probe). */\n"
              "#if 0\n" + initfile[header_start:header_end].rstrip() + "\n#endif")
    return "\n".join([normalize, header, function(dposv, "void stcOptInit(")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    sources = [src / "readdef.c", src / "initfile.c", src / "stcopt_dposv.c", src / "include/global.h"]
    materialize(root, "negative_stepdt_upstream.inc",
                extract(sources[0].read_text(), sources[1].read_text(), sources[2].read_text()),
                sources, args.write)
    cc_version = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    lines = [
        "# C mVMC 1.3.0 negative DSROptStepDt oracle (#367): readdef.c:749-755 normalization,",
        "# initfile.c:47-54 SRinfo header and stcOptInit + dposv on fixed operands (3 parameters).",
        *(f"# {p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}" for p in sources),
        f"# compiler: {cc_version}; options: cc -O0 -ffp-contract=off -I extern/mVMC-1.3.0/src/mVMC/include -I c_toolbox ... -llapack",
        f"# platform: {platform.system()} {platform.machine()} {platform.libc_ver()[0]} {platform.libc_ver()[1]}",
        "# reproduce: uv run --no-project python scripts/check_negative_stepdt_c_parity.py --write",
        "# format: 'case <name> raw_dt <text> sta_del <text>' then 'key value' lines",
    ]
    with tempfile.TemporaryDirectory(prefix="mvmc-c-stepdt-") as directory:
        probe = Path(directory) / "probe"
        subprocess.run(["cc", "-O0", "-ffp-contract=off", "-I", str(src / "include"), "-I",
                        str(root / "c_toolbox"), str(root / "c_toolbox/negative_stepdt.c"), "-o",
                        str(probe), "-llapack"], check=True)
        for name, dt, delta in CASES:
            out = subprocess.run([str(probe), dt, delta], capture_output=True, text=True,
                                 timeout=10, check=True).stdout
            lines.append(f"case {name} raw_dt {dt} sta_del {delta}")
            lines.extend(line[len("RESULT "):] for line in out.splitlines()
                         if line.startswith("RESULT "))
    output = "\n".join(lines) + "\n"
    target = root / "tests/fixtures/negative_stepdt/c_sr.txt"
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output)
    else:
        assert target.read_text() == output, "C negative DSROptStepDt results changed"
    print(f"{len(CASES)} C negative DSROptStepDt cases checked")


if __name__ == "__main__":
    main()
