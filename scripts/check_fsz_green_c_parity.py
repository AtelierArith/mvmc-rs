#!/usr/bin/env -S uv run --no-project
"""Optional actual-C FSZ Green kernels; historical Julia arrays are inputs only.

No RBM, RNG, full executable, sampling or SR parity is claimed by this probe.
"""
import argparse
import hashlib
import platform
from pathlib import Path
import subprocess
import tempfile

from numerical_comparison import compare_text, GREEN, ENERGY, MEASUREMENT
from c_toolbox import materialize, add_native_platform_argument, native_platform, native_compiler, native_provenance, native_target
from check_general_orbital_c_parity import function
from check_interall_real_c_parity import bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--real", action="store_true", help="Probe the separate scalar FSZ family")
    parser.add_argument("--hamiltonian", action="store_true", help="Probe the complete serial Hamiltonian")
    parser.add_argument("--measurements", action="store_true", help="Probe serial weighted FSZ measurements")
    add_native_platform_argument(parser)
    args = parser.parse_args()
    kind = native_platform(args.platform)
    if args.measurements and (args.real or args.hamiltonian):
        parser.error("--measurements uses the complex FSZ family and is separate from --real/--hamiltonian")
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    selected = {
        "projection.c": ["inline double ProjRatio(", "void UpdateProjCnt(", "void UpdateProjCnt_fsz("],
        "pfupdate_fsz.c": ["void CalculateNewPfM_fsz("],
        "pfupdate_two_fsz.c": ["void CalculateNewPfMTwo_fsz(", "void calculateNewPfMTwo_child_fsz("],
        "qp.c": ["double complex CalculateIP_fcmp("],
        "locgrn_fsz.c": ["double complex GreenFunc1_fsz(", "double complex GreenFunc1_fsz2(",
                         "double complex GreenFunc2_fsz(", "double complex GreenFunc2_fsz2("],
    }
    if args.real:
        selected = {
            "projection.c": selected["projection.c"],
            "pfupdate_fsz_real.c": ["void CalculateNewPfM_fsz_real("],
            "pfupdate_two_fsz_real.c": ["void CalculateNewPfMTwo_fsz_real(",
                                      "void calculateNewPfMTwo_child_fsz_real("],
            "qp_real.c": ["double  CalculateIP_real("],
            "locgrn_fsz_real.c": ["double GreenFunc1_fsz_real(", "double GreenFunc1_fsz2_real(",
                                 "double GreenFunc2_fsz_real(", "double GreenFunc2_fsz2_real("],
        }
    files = [src / name for name in selected]
    body = "\n".join(function(path.read_text(), sig)
                     for path in files for sig in selected[path.name])
    snippet = "fsz_green_real_upstream.inc" if args.real else "fsz_green_upstream.inc"
    materialize(root, snippet, body, files, args.write)
    accumulator = src / ("calham_fsz_real.c" if args.real else "calham_fsz.c")
    source = accumulator.read_text()
    start = source.index("for(idx=0;idx<NInterAll;idx++)")
    brace = source.index("{", start)
    snippet = "fsz_interall_real_accumulator_upstream.inc" if args.real else "fsz_interall_accumulator_upstream.inc"
    materialize(root, snippet,
                source[start:brace] + function(source[brace:], "{"),
                [src / ("locgrn_fsz_real.c" if args.real else "locgrn_fsz.c"), accumulator], args.write)
    files.append(accumulator)
    if args.hamiltonian:
        signature = "double CalculateHamiltonian_fsz_real(" if args.real else "double complex CalculateHamiltonian_fsz("
        snippet = "fsz_hamiltonian_real_upstream.inc" if args.real else "fsz_hamiltonian_upstream.inc"
        materialize(root, snippet, function(source, signature), [accumulator], args.write)
    if args.measurements:
        measurements = src / "calgrn_fsz.c"
        materialize(root, "fsz_measurements_upstream.inc",
                    function(measurements.read_text(), "void CalculateGreenFunc_fsz("),
                    [measurements], args.write)
        files.append(measurements)
        average = src / "average.c"
        materialize(root, "fsz_measurements_average_upstream.inc",
                    function(average.read_text(), "void weightAverageReduce_fcmp(int n, double  complex"),
                    [average], args.write)
        files.append(average)
        files += [src / "setmemory.c", src / "vmccal_fsz.c", src / "include/global.h"]
    provenance = "; ".join(f"{p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files)
    family = ("actual C scalar FSZ GreenFunc1/2/1_fsz2/2_fsz2_real; real input arrays; "
              "real accumulator discards coefficient imaginary part per C compound assignment; "
              if args.real else "actual C FSZ GreenFunc1/2/1_fsz2/2_fsz2; real and complex input arrays; ")
    if args.hamiltonian:
        family += "complete serial Hamiltonian: diagonal, Transfer, PairHop, Exchange, InterAll, all combined; "
    if args.measurements:
        family += ("CalculateGreenFunc_fsz verbatim body, serial workspace/timer stubs; "
                   "canonical one=64 direct=4098 (4096+duplicate73,3072) factored=8; "
                   "factored indices=(0,0),(19,35),(35,19),(18,63),(63,0),(19,19),(27,35),(0,0); "
                   "weights=.375,1.25,.125, successive cumulative frames; "
                   "rawOne/rawDirect then weightedOne/weightedDirect/factored per frame; "
                   "weightAverageReduce_fcmp verbatim serial branch, complex Wc=1.75+0i per global.h, "
                   "real weights summed in double then assigned to Wc, normalized final triple; "
                   "real wavefunctions use complex shadow family: setmemory.c PfM follows InvM, "
                   "vmccal_fsz.c real-to-complex NQPFull*(Nsize*Nsize+1) copy includes PfM tail; ")
    output = ("# " + family +
              "zero/nonzero real projection; no RBM; MPI_COMM_SELF only; "
              "Apple clang 17 -O0 -ffp-contract=off; " + provenance + "\n")
    if kind == "linux-gnu":
        output = output.replace("Apple clang 17 -O0 -ffp-contract=off", native_provenance(kind))
    elif args.hamiltonian or args.measurements:
        # Preserve historical Green headers; new production references record
        # the actual compiler even when CC is overridden.
        version = subprocess.check_output(native_compiler() + ["--version"], text=True).splitlines()[0]
        measured = (f"macOS {platform.mac_ver()[0]} {platform.machine()}; compiler={version}; "
                    "-O0 -ffp-contract=off")
        output = output.replace("Apple clang 17 -O0 -ffp-contract=off", measured)
    if args.measurements:
        inputs = root / "tests/fixtures/interall/green_fsz.txt"
        output += ("# 26 lines/model: original 11 input lines, 3 weight hex values, rawOne, rawDirect, "
                   "3 frames of weightedOne/weightedDirect/factored, then normalizedOne/Direct/factored; "
                   "complex values are component IEEE hex. "
                   "Historical Julia arrays are inputs only; "
                   f"green_fsz.txt sha256={hashlib.sha256(inputs.read_bytes()).hexdigest()}; "
                   f"fsz_green.c sha256={hashlib.sha256((root / 'c_toolbox/fsz_green.c').read_bytes()).hexdigest()}\n")
        output += ("# Extraction: complete calgrn_fsz.c CalculateGreenFunc_fsz and average.c weightAverageReduce_fcmp "
                   "functions, signatures through "
                   "balanced closing brace; upstream OpenMP pragmas retained and ignored for serial execution. "
                   "Flags: -DFSZ_MEASUREMENTS=1 -Wno-unknown-pragmas. Reproduce with "
                   "uv run --no-project python scripts/check_fsz_green_c_parity.py --measurements "
                   "(--write explicitly regenerates).\n")
    original = [line for line in (root / "tests/fixtures/interall/green_fsz.txt").read_text().splitlines()
                if not line.startswith("#")]
    checked_one = checked_two = models = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-fsz-green-") as directory:
        exe = Path(directory) / "probe"
        flags = ["-DFSZ_REAL=1"] if args.real else []
        if args.hamiltonian:
            flags += ["-DFSZ_HAMILTONIAN=1", "-Wno-unknown-pragmas"]
        if args.measurements:
            flags += ["-DFSZ_MEASUREMENTS=1", "-Wno-unknown-pragmas"]
        subprocess.run(native_compiler() + ["-O0", "-ffp-contract=off"] + flags + [str(root / "c_toolbox/fsz_green.c"),
                        "-lm", "-o", str(exe)], check=True)
        for case in range(6):
            rows = original[case*13:(case+1)*13]
            if args.real and rows[0] != "0":
                continue
            for projection in ((0.0, 0.0), (0.125, -0.2)):
                parameters = " ".join(map(bits, (projection[0], 0.0, projection[1], 0.0)))
                header = rows[:6] + [parameters] + rows[6:10]
                payload = "\n".join(header[1:]) + "\n"
                result = subprocess.run([str(exe)], input=payload, text=True, capture_output=True,
                                        check=True, timeout=5)
                expected = result.stdout.splitlines()
                assert len(expected) == (64 + 4096 + 1 + (6 if args.hamiltonian else 0)
                                         + (2 + 4 * (64 + 4098 + 8) if args.measurements else 0))
                output += "\n".join(header) + "\n"
                if args.measurements:
                    output += " ".join(map(bits, (.375, 1.25, .125))) + "\n"
                    output += " ".join(expected[:64]) + "\n"
                    output += " ".join(expected[64:64+4098]) + "\n"
                    # The existing ordered InterAll scalar is outside measurement output.
                    cursor = 64 + 4098 + 1
                    for _ in range(4):
                        for size in (64, 4098, 8):
                            output += " ".join(expected[cursor:cursor+size]) + "\n"
                            cursor += size
                    assert cursor == len(expected)
                elif args.hamiltonian:
                    output += " ".join(expected[-6:]) + "\n"
                else:
                    output += " ".join(expected[:64]) + "\n"
                    output += " ".join(expected[64:64+4096]) + "\n"
                    output += expected[-1] + "\n"
                checked_one += 64
                checked_two += 4096
                models += 1
    name = "c_fsz_real_green.txt" if args.real else "c_fsz_green.txt"
    if args.hamiltonian:
        name = "c_fsz_real_hamiltonian.txt" if args.real else "c_fsz_hamiltonian.txt"
    if args.measurements:
        name = "c_fsz_measurements.txt"
    target = native_target(root, name.removesuffix(".txt"), kind)
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        stride = 26 if args.measurements else 12 if args.hamiltonian else 14
        def computed(row, column, fields):
            offset = row % stride
            if args.measurements:
                return (GREEN if offset in (12, 13) else MEASUREMENT) if offset >= 12 else None
            if args.hamiltonian:
                return ENERGY if offset == 11 else None
            return GREEN if offset in (11, 12) else ENERGY if offset == 13 else None
        assert compare_text(output, target.read_text(), computed) == models * stride
    if args.measurements:
        print(f"{models} native C FSZ measurement models / {models * 3} successive weighted frames "
              f"and {models * 3} native normalized vectors passed")
    elif args.hamiltonian:
        print(f"{models * 6} complete native C FSZ Hamiltonians passed")
    else:
        print(f"{checked_one} native C FSZ one-body and {checked_two} two-body operators, "
              f"{models} ordered InterAll sums with duplicates passed")


if __name__ == "__main__":
    main()
