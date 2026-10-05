#!/usr/bin/env python3
"""Input definitions for the ComplexUHF C-parity cases (#350).

`build_cases(root)` returns {case_name: {file_name: text}}. Copied cases are
taken verbatim from the vendored upstream trees (their `ref/` directories are
not used: the expectations are regenerated from the C executable). Generated
cases use only the file formats read by `src/ComplexUHF/readdef.c`.
"""
import cmath
import math
from pathlib import Path

HEADER = "=" * 24


def header(title, count, extra=None):
    lines = [HEADER, f"{title} {count}"]
    lines.append(extra if extra else HEADER)
    lines += [HEADER] * (5 - len(lines))
    return "\n".join(lines) + "\n"


def modpara(nsite, ncond, two_sz=0, extra=()):
    rows = [
        "--------------------", "Model_Parameters   0", "--------------------",
        "VMC_Cal_Parameters", "--------------------", "CDataFileHead  zvo",
        "CParaFileHead  zqp", "--------------------", "NVMCCalMode    0",
        "--------------------", "NDataIdxStart  1", "NDataQtySmp    1",
        "--------------------", f"Nsite          {nsite}", f"Ncond          {ncond}",
        f"2Sz            {two_sz}", "NMPTrans       1",
    ]
    rows += list(extra)
    return "\n".join(rows) + "\n"


def namelist(entries):
    return "".join(f"{key:>16s}  {name}\n" for key, name in entries)


def locspn(nsite):
    text = header("NlocalSpin", 0)
    return text + "".join(f"{i:5d} 0\n" for i in range(nsite))


def trans(nsite, hops, flux=0.0):
    """Ring hopping `hops` list of (i, j, t); value t*exp(i*flux) for i->j."""
    lines = []
    for i, j, t in hops:
        phase = cmath.exp(1j * flux) * t
        for spin in range(2):
            lines.append(f"{i} {spin} {j} {spin} {phase.real:.15f} {phase.imag:.15f}")
            lines.append(f"{j} {spin} {i} {spin} {phase.real:.15f} {-phase.imag:.15f}")
    return header("NTransfer", len(lines)) + "\n".join(lines) + "\n"


def two_site(title, rows, fmt):
    return header(title, len(rows)) + "".join(fmt(row) for row in rows)


def ring(nsite):
    return [(i, (i + 1) % nsite) for i in range(nsite)]


def coulomb_intra(nsite, value):
    rows = [(i, value) for i in range(nsite)]
    return two_site("NCoulombIntra", rows, lambda r: f"{r[0]} {r[1]:.15f}\n")


def pair_rows(title, pairs, value):
    rows = [(i, j, value) for i, j in pairs]
    return two_site(title, rows, lambda r: f"{r[0]} {r[1]} {r[2]:.15f}\n")


def orbital_text(nsite, index, title="NOrbitalIdx"):
    rows = [(i, j, index(i, j)) for i in range(nsite) for j in range(nsite)]
    nidx = max(r[2] for r in rows) + 1
    text = header(title, nidx, "ComplexType          0")
    return text + "".join(f"{i} {j} {k}\n" for i, j, k in rows)


def copy_tree(path, skip=("ref",)):
    out = {}
    for entry in sorted(Path(path).iterdir()):
        if entry.is_file() and entry.name not in skip and entry.suffix in (".def", ".in"):
            out[entry.name] = entry.read_text()
    return out


def read_modpara_extra(text, extra):
    return text.rstrip("\n") + "\n" + "\n".join(extra) + "\n"


def build_cases(root):
    root = Path(root)
    data = root / "extern/mVMC-1.3.0/test/python/data"
    cases = {}

    # --- verbatim upstream inputs ---------------------------------------
    cases["hubbard_chain_real"] = copy_tree(
        root / "extern/Julia-mVMC/examples/inputs/hubbard_chain_real")
    cases["uhf_hubbard_square"] = copy_tree(data / "UHF_HubbardSquare")
    cases["uhf_hubbard_triangular"] = copy_tree(data / "UHF_HubbardTriangular")
    cases["uhf_interall_n2"] = copy_tree(data / "UHF_InterAll_N2")
    # Same as the square case but stopped after five iterations (not converged).
    square = dict(cases["uhf_hubbard_square"])
    square["modpara.def"] = read_modpara_extra(square["modpara.def"], ["IterationMax    5"])
    cases["uhf_hubbard_square_unconverged"] = square

    # --- InterAll form of Coulomb + Exchange (test_UHF_InterAll.py recipe) ---
    ex = copy_tree(data / "UHF_InterAll_Exchange")
    rows = []
    for line in ex["coulombintra.def"].splitlines()[5:]:
        site, value = line.split()[:2]
        rows.append((int(site), 0, int(site), 0, int(site), 1, int(site), 1, float(value), 0.0))
    for line in ex["exchange.def"].splitlines()[5:]:
        a, b, value = line.split()[:3]
        a, b, value = int(a), int(b), float(value)
        rows.append((a, 0, b, 0, b, 1, a, 1, value, 0.0))
        rows.append((a, 1, b, 1, b, 0, a, 0, value, 0.0))
    text = header("TotalNumber", len(rows), "Comment: interall")
    text += "".join(" ".join(str(v) for v in row) + " \n" for row in rows)
    ex["interall.def"] = text
    ex["namelist.def"] = ex.pop("namelist_all.def")
    cases["uhf_interall_exchange"] = ex

    # --- generated: antiparallel chain with a partial Neel Initial seed ------
    n = 8
    case = {
        "namelist.def": namelist([
            ("ModPara", "modpara.def"), ("LocSpin", "locspn.def"), ("Trans", "trans.def"),
            ("CoulombIntra", "coulombintra.def"), ("Orbital", "orbitalidx.def"),
            ("Initial", "initial.def")]),
        "modpara.def": modpara(n, n, extra=["Mix            0.7", "EPS            8",
                                            "IterationMax   500", "EpsSlater      3",
                                            "RndSeed        4242", "Print          1"]),
        "locspn.def": locspn(n),
        "trans.def": trans(n, [(i, (i + 1) % n, 1.0) for i in range(n)]),
        "coulombintra.def": coulomb_intra(n, 4.0),
        "orbitalidx.def": orbital_text(n, lambda i, j: (j - i) % n, "NOrbitalIdx"),
    }
    seed = []
    for i in range(n):
        up = 0.9 if i % 2 == 0 else 0.1
        seed.append(f"{i} 0 {i} 0 {up:.15f} 0.0")
        seed.append(f"{i} 1 {i} 1 {1.0 - up:.15f} 0.0")
    case["initial.def"] = header("NInitial", len(seed)) + "\n".join(seed) + "\n"
    cases["chain_ap_neel"] = case

    # --- generated: General orbital, flux + all two-body families -------------
    n = 6
    bonds = ring(n)
    pairs = [(a, b) for a in range(2 * n) for b in range(a + 1, 2 * n)]
    general_rows = []
    for k, (a, b) in enumerate(pairs):
        general_rows.append((a % n, a // n, b % n, b // n, k // 2, 1 if k % 2 == 0 else -1))
    general = header("NOrbitalIdx", len(pairs) // 2 + len(pairs) % 2, "ComplexType          1")
    general += "".join(" ".join(str(v) for v in row) + "\n" for row in general_rows)
    cases["ring_general_mixed"] = {
        "namelist.def": namelist([
            ("ModPara", "modpara.def"), ("LocSpin", "locspn.def"), ("Trans", "trans.def"),
            ("CoulombIntra", "coulombintra.def"), ("CoulombInter", "coulombinter.def"),
            ("Hund", "hund.def"), ("Exchange", "exchange.def"), ("PairHop", "pairhop.def"),
            ("OrbitalGeneral", "orbitalgeneral.def")]),
        "modpara.def": modpara(n, n, extra=["Mix            0.5", "EPS            9",
                                            "IterationMax   1500", "RndSeed        777",
                                            "EpsSlater      3"]),
        "locspn.def": locspn(n),
        "trans.def": trans(n, [(i, j, 1.0) for i, j in bonds], flux=0.35),
        "coulombintra.def": coulomb_intra(n, 3.0),
        "coulombinter.def": pair_rows("NCoulombInter", bonds, 1.0),
        "hund.def": pair_rows("NHund", bonds, 0.4),
        "exchange.def": pair_rows("NExchange", bonds, 0.3),
        "pairhop.def": pair_rows("NPairhop", bonds, 0.2),
        "orbitalgeneral.def": general,
    }

    # --- generated: AP + P (mode 2) -----------------------------------------
    n = 6
    cases["ring_ap_parallel"] = {
        "namelist.def": namelist([
            ("ModPara", "modpara.def"), ("LocSpin", "locspn.def"), ("Trans", "trans.def"),
            ("CoulombIntra", "coulombintra.def"), ("Orbital", "orbitalap.def"),
            ("OrbitalParallel", "orbitalp.def")]),
        "modpara.def": modpara(n, n, extra=["Mix            0.5", "EPS            9",
                                            "RndSeed        31415", "EpsSlater      3"]),
        "locspn.def": locspn(n),
        "trans.def": trans(n, [(i, j, 1.0) for i, j in ring(n)]),
        "coulombintra.def": coulomb_intra(n, 3.0),
        "orbitalap.def": orbital_text(n, lambda i, j: (j - i) % n, "NOrbitalIdx"),
        "orbitalp.def": header("NOrbitalIdx", 5, "ComplexType          0") + "".join(
            f"{i} {j} {abs(i - j) - 1} {1 if (j - i) % 2 else -1}\n"
            for i in range(n) for j in range(i + 1, n)),
    }

    # --- generated: no orbital file, every real two-body family ---------------
    n = 6
    cases["ring_no_orbital"] = {
        "namelist.def": namelist([
            ("ModPara", "modpara.def"), ("LocSpin", "locspn.def"), ("Trans", "trans.def"),
            ("CoulombIntra", "coulombintra.def"), ("CoulombInter", "coulombinter.def"),
            ("Hund", "hund.def"), ("Exchange", "exchange.def"), ("PairHop", "pairhop.def")]),
        "modpara.def": modpara(n, n, extra=["Mix            0.5", "EPS            9",
                                            "IterationMax   1500", "RndSeed        2024"]),
        "locspn.def": locspn(n),
        "trans.def": trans(n, [(i, j, 1.0) for i, j in ring(n)], flux=0.2),
        "coulombintra.def": coulomb_intra(n, 2.5),
        "coulombinter.def": pair_rows("NCoulombInter", ring(n), 0.8),
        "hund.def": pair_rows("NHund", ring(n), 0.3),
        "exchange.def": pair_rows("NExchange", ring(n), 0.2),
        "pairhop.def": pair_rows("NPairhop", ring(n), 0.1),
    }
    return cases
