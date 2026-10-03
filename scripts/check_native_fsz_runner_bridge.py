#!/usr/bin/env -S uv run --no-project
"""Build an optional native C energy bridge and validate its 108 frozen inputs.

Platform-specific optional oracle. Cargo never imports, compiles or invokes it.
"""
import argparse
import ctypes
import hashlib
import itertools
from pathlib import Path
import struct
import subprocess
import tempfile

from c_toolbox import materialize, native_compiler, native_platform, native_provenance
from check_general_orbital_c_parity import function


def verify_sources(root, real):
    src = root / "extern/mVMC-1.3.0/src/mVMC"
    selected = {
        "projection.c": ["inline double ProjRatio(", "void UpdateProjCnt(", "void UpdateProjCnt_fsz("],
        "pfupdate_fsz.c": ["void CalculateNewPfM_fsz("],
        "pfupdate_two_fsz.c": ["void CalculateNewPfMTwo_fsz(", "void calculateNewPfMTwo_child_fsz("],
        "qp.c": ["double complex CalculateIP_fcmp("],
        "locgrn_fsz.c": ["double complex GreenFunc1_fsz(", "double complex GreenFunc1_fsz2(",
                         "double complex GreenFunc2_fsz(", "double complex GreenFunc2_fsz2("],
    }
    if real:
        selected = {
            "projection.c": selected["projection.c"],
            "pfupdate_fsz_real.c": ["void CalculateNewPfM_fsz_real("],
            "pfupdate_two_fsz_real.c": ["void CalculateNewPfMTwo_fsz_real(", "void calculateNewPfMTwo_child_fsz_real("],
            "qp_real.c": ["double  CalculateIP_real("],
            "locgrn_fsz_real.c": ["double GreenFunc1_fsz_real(", "double GreenFunc1_fsz2_real(",
                                "double GreenFunc2_fsz_real(", "double GreenFunc2_fsz2_real("],
        }
    files = [src / name for name in selected]
    body = "\n".join(function(p.read_text(), sig) for p in files for sig in selected[p.name])
    materialize(root, "fsz_green_real_upstream.inc" if real else "fsz_green_upstream.inc", body, files, False)
    ham = src / ("calham_fsz_real.c" if real else "calham_fsz.c")
    signature = "double CalculateHamiltonian_fsz_real(" if real else "double complex CalculateHamiltonian_fsz("
    materialize(root, "fsz_hamiltonian_real_upstream.inc" if real else "fsz_hamiltonian_upstream.inc",
                function(ham.read_text(), signature), [ham], False)


def interactions():
    intra = [((i,), complex((i - 1) / 8)) for i in range(4)]
    inter, hund, pair, exchange = [], [], [], []
    for i, j in itertools.product(range(4), repeat=2):
        k = 4*i + j
        inter.append(((i, j), complex((k % 5 - 2) / 16)))
        hund.append(((i, j), complex((k % 7 - 3) / 32)))
        pair.append(((i, j), complex((k % 11 - 5) / 16)))
        exchange.append(((i, j), complex((k % 13 - 6) / 32)))
    pair += [pair[0], pair[7]]
    exchange += [exchange[0], exchange[11]]
    transfer, interall = [], []
    for s, t, i, j in itertools.product(range(2), range(2), range(4), range(4)):
        k = len(transfer)
        transfer.append(((i, s, j, t), complex((k % 7 - 3) / 16, (k % 11 - 5) / 32)))
    transfer.append(transfer[19])
    for s, t, u, v in itertools.product(range(2), repeat=4):
        for i, j, k, l in itertools.product(range(4), repeat=4):
            q = len(interall)
            interall.append(((i, s, j, t, k, u, l, v), complex((q % 7 - 3) / 16, (q % 11 - 5) / 32)))
    interall += [interall[73], interall[3072]]
    return (intra, inter, hund, transfer, pair, exchange, interall)


def doubles(row):
    return [struct.unpack("!d", bytes.fromhex(word))[0] for word in row.split()]


def complex_components(values):
    return [part for z in values for part in (z.real, z.imag)]


def array(kind, values):
    return (kind * len(values))(*values)


def validate(root, library, real, kind):
    fn = ctypes.CDLL(str(library)).mvmc_reference_fsz_energy
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2 + [ctypes.POINTER(ctypes.c_double)] * 4 \
        + [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    fn.restype = None
    stem = "c_fsz_real_hamiltonian" if real else "c_fsz_hamiltonian"
    name = stem + ("_linux_gnu" if kind == "linux-gnu" else "") + ".txt"
    rows = [r for r in (root / "tests/fixtures/interall" / name).read_text().splitlines() if not r.startswith("#")]
    assert len(rows) % 12 == 0
    sections = interactions()
    checked = 0
    for start in range(0, len(rows), 12):
        r = rows[start:start+12]
        for group in range(6):
            active = (group in (0, 5), group in (0, 5), group in (0, 5),
                      group in (1, 5), group in (2, 5), group in (3, 5), group in (4, 5))
            groups = [s if keep else [] for s, keep in zip(sections, active)]
            dims = array(ctypes.c_int, [4, 4, 2, 1, 1, 0, 0] + list(map(len, groups)))
            mappings = [0]*4 + [(-1 if i == j else 0) for i in range(4) for j in range(4)]
            couplings = []
            for terms in groups:
                mappings += [x for coords, _ in terms for x in coords]
                couplings += [z for _, z in terms]
            tables = array(ctypes.c_int, mappings)
            values = array(ctypes.c_double, doubles(r[6]) + [1, 0, -0.375, 0] + complex_components(couplings))
            slater, pf, inv = [array(ctypes.c_double, doubles(r[i])) for i in (7, 8, 9)]
            config = array(ctypes.c_int, [int(x) for row in r[1:6] for x in row.split()])
            ip = array(ctypes.c_double, doubles(r[10]))
            out = (ctypes.c_double * 2)()
            operands = (dims, tables, values, slater, pf, inv, config, ip)
            before = [bytes(v) for v in operands]
            fn(*operands, out)
            assert [bytes(v) for v in operands] == before, "Native energy bridge mutated a borrowed operand"
            actual = [struct.unpack("=Q", struct.pack("=d", part))[0] for part in out]
            expected = [int(word, 16) for word in r[11].split()[2*group:2*group+2]]
            assert actual == expected, (real, start // 12, group, actual, expected)
            checked += 1
    return checked


def build_and_validate(root, destination, write=False):
    kind = native_platform("auto")
    projection = root / "extern/mVMC-1.3.0/src/mVMC/projection.c"
    materialize(root, "fsz_runner_projection_upstream.inc",
                function(projection.read_text(), "void MakeProjCnt("), [projection], write)
    destination.mkdir(parents=True, exist_ok=True)
    counts = []
    for real in (False, True):
        verify_sources(root, real)
        suffix = ".so" if kind == "linux-gnu" else ".dylib"
        library = destination / (("libmvmc_fsz_reference_real" if real else "libmvmc_fsz_reference") + suffix)
        flags = ["-O0", "-ffp-contract=off", "-fPIC", "-shared" if kind == "linux-gnu" else "-dynamiclib",
                 "-fvisibility=hidden", "-Wno-unknown-pragmas"]
        if real:
            flags.append("-DFSZ_REAL=1")
        subprocess.run(native_compiler() + flags + [str(root / "c_toolbox/fsz_runner_reference.c"),
                       "-lm", "-o", str(library)], check=True)
        counts.append(validate(root, library, real, kind))
    text = native_provenance(kind)
    if text is None:
        text = subprocess.run(native_compiler() + ["--version"], check=True, text=True,
                              capture_output=True).stdout.splitlines()[0] + "; macOS -O0 -ffp-contract=off"
    text += "; shared bridge -fPIC -fvisibility=hidden; "
    text += "native exact-input gates=" + str(sum(counts)) + "; "
    for path in [root / "c_toolbox" / name for name in ("fsz_runner_reference.c", "fsz_green_upstream.inc",
                 "fsz_green_real_upstream.inc", "fsz_hamiltonian_upstream.inc", "fsz_hamiltonian_real_upstream.inc",
                 "fsz_runner_projection_upstream.inc")]:
        text += f"{path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}; "
    (destination / "provenance.txt").write_text(text + "\n")
    print(f"Native bridge passed {counts[0]} complex + {counts[1]} scalar FSZ energy gates; borrowed operands unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--write", action="store_true", help="Explicitly materialize native MakeProjCnt source")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.build_dir:
        build_and_validate(root, args.build_dir, args.write)
    else:
        with tempfile.TemporaryDirectory(prefix="mvmc-native-fsz-runner-") as directory:
            build_and_validate(root, Path(directory), args.write)


if __name__ == "__main__":
    main()
