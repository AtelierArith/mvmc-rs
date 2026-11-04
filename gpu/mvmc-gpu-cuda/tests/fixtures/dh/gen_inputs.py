"""Generate the resident GPU doublon-holon inputs (#665, #572 milestone 2).

uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/dh/gen_inputs.py

Writes two deterministic Expert-mode Hubbard chains next to this script:

* hubbard_dh_L16: 16 sites, Ncond=12 (2Ne=12, doped), NExUpdatePath=1
  (hopping/exchange mixture), two DH2 tables and one DH4 table.
* hubbard_dh_L128: 128 sites, Ncond=128 (2Ne=128: Opt delayed sampling queue
  and paired inverse), NExUpdatePath=0, one DH2 and one DH4 table.

Both have periodic nearest-neighbour Transfer t=1, CoulombIntra U=4, and
nearest-neighbour PairHop and Exchange terms, so the local energy exercises
one- and two-electron DH ratios. Real normal translational orbitals,
Gutzwiller/Jastrow, identity TransSym and NSPGaussLeg=8. InGutzwiller,
InDH2 and InDH4 give nonzero initial projection factors. Every optimization
flag is 1, so C SetFlagShift enables FlagShiftGJ, FlagShiftDH2 and
FlagShiftDH4. These are engine-test inputs, not physics benchmarks.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = "=" * 45


def header(name, count, extra=()):
    """Five header lines: an index file carries ComplexType instead of a title."""
    title = BAR if extra else f"{'=' * 10} {name} {'=' * 10}"
    return [BAR, f"{name} {count:>10}", *extra, BAR, title, BAR][:5]


def write(directory, name, lines):
    (directory / name).write_text("\n".join(lines) + "\n")


def modpara(nsite, ncond, path):
    return f"""--------------------
Model_Parameters   0
--------------------
VMC_Cal_Parameters
--------------------
CDataFileHead  zvo
CParaFileHead  zqp
--------------------
NVMCCalMode    0
NLanczosMode   0
--------------------
NDataIdxStart  1
NDataQtySmp    1
--------------------
Nsite          {nsite}
Ncond          {ncond}
2Sz            0
NSPGaussLeg    8
NSPStot        0
NMPTrans       1
NSROptItrStep  20
NSROptItrSmp   20
DSROptRedCut   0.0000000100
DSROptStaDel   0.0100000000
DSROptStepDt   0.0030000000
NVMCWarmUp     10
NVMCInterval   1
NVMCSample     64
NExUpdatePath  {path}
RndSeed        79
NSplitSize     1
NStore         1
NSRCG          0""".splitlines()


def distance(a, b, length):
    d = abs(a - b) % length
    return min(d, length - d)


def generate(name, length, ncond, path, dh2_shells, dh4_shells):
    out = HERE / name
    out.mkdir(exist_ok=True)
    ns = length
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, ncond, path))
    write(out, "locspn.def",
          header("NlocalSpin", 0) + [f"{i:5d} {0:6d}" for i in range(ns)])
    trans = []
    for i in range(ns):
        j = (i + 1) % ns
        for s in (0, 1):
            trans.append(f"{j:5d} {s:5d} {i:5d} {s:5d} {1.0:25.15f} {0.0:25.15f}")
            trans.append(f"{i:5d} {s:5d} {j:5d} {s:5d} {1.0:25.15f} {-0.0:25.15f}")
    write(out, "trans.def", header("NTransfer", len(trans)) + trans)
    write(out, "coulombintra.def",
          header("NCoulombIntra", ns) + [f"{i:5d} {4.0:25.15f}" for i in range(ns)])
    bonds = [(i, (i + 1) % ns) for i in range(ns)]
    write(out, "pairhop.def",
          header("NPairHop", len(bonds)) + [f"{i:5d} {j:5d} {0.3:25.15f}" for i, j in bonds])
    write(out, "exchange.def",
          header("NExchange", len(bonds)) + [f"{i:5d} {j:5d} {0.25:25.15f}" for i, j in bonds])
    files += [("Trans", "trans.def"), ("CoulombIntra", "coulombintra.def"),
              ("PairHop", "pairhop.def"), ("Exchange", "exchange.def")]
    one = [(i, s, i, s) for i in (0, 1, 2) for s in (0, 1)]
    one += [(0, 0, 1, 0), (1, 1, 0, 1), (2, 0, 0, 0), (0, 1, 3, 1)]
    write(out, "greenone.def",
          header("NCisAjs", len(one)) + ["".join(f"{v:6d}" for v in t) for t in one])
    # Diagonal and off-diagonal (two-move) density/hopping products.
    two = [
        (0, 0, 0, 0, 1, 1, 1, 1),
        (0, 0, 1, 0, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 1, 0, 1),
        (0, 0, 1, 0, 2, 1, 3, 1),
        (2, 1, 2, 1, 0, 0, 1, 0),
        (3, 0, 2, 0, 3, 1, 2, 1),
    ]
    write(out, "greentwo.def",
          header("NCisAjsCktAltDC", len(two)) + ["".join(f"{v:6d}" for v in t) for t in two])
    files += [("OneBodyG", "greenone.def"), ("TwoBodyG", "greentwo.def")]
    write(out, "gutzwilleridx.def",
          header("NGutzwillerIdx", 1, ["ComplexType          0"])
          + [f"{i:5d} {0:6d}" for i in range(ns)] + [f"{0:5d} {1:6d}"])
    half = length // 2
    rows = [f"{i:5d} {j:6d} {distance(i, j, length) - 1:6d}"
            for i in range(ns) for j in range(ns) if i != j]
    write(out, "jastrowidx.def",
          header("NJastrowIdx", half, ["ComplexType          0"])
          + rows + [f"{k:5d} {1:6d}" for k in range(half)])
    # DH2 table x: neighbours at distance shell[x]; DH4: both shells.
    n2 = len(dh2_shells)
    rows = [f"{r:5d} {(r - d) % ns:5d} {(r + d) % ns:5d} {x:5d}"
            for x, d in enumerate(dh2_shells) for r in range(ns)]
    write(out, "dh2.def",
          header("NDoublonHolon2siteIdx", n2, ["ComplexType          0"])
          + rows + [f"{k:5d} {1:5d}" for k in range(6 * n2)])
    n4 = len(dh4_shells)
    rows = [f"{r:5d} {(r - a) % ns:5d} {(r + a) % ns:5d} {(r - b) % ns:5d} {(r + b) % ns:5d} {x:5d}"
            for x, (a, b) in enumerate(dh4_shells) for r in range(ns)]
    write(out, "dh4.def",
          header("NDoublonHolon4siteIdx", n4, ["ComplexType          0"])
          + rows + [f"{k:5d} {1:5d}" for k in range(10 * n4)])
    files += [("Gutzwiller", "gutzwilleridx.def"), ("Jastrow", "jastrowidx.def"),
              ("DH2", "dh2.def"), ("DH4", "dh4.def")]
    write(out, "orbitalidx.def",
          header("NOrbitalIdx", length, ["ComplexType          0"])
          + [f"{i:5d} {j:6d} {(j - i) % length:6d}" for i in range(ns) for j in range(ns)]
          + [f"{k:5d} {1:6d}" for k in range(length)])
    write(out, "qptransidx.def",
          header("NQPTrans", 1) + ["0    1.00000"]
          + [f"{0:5d} {i:6d} {i:6d} {1:6d}" for i in range(ns)])
    files += [("Orbital", "orbitalidx.def"), ("TransSym", "qptransidx.def")]
    # Initial projection factors, exact binary fractions of moderate size.
    write(out, "ingutzwiller.def",
          header("NGutzwillerIdx", 1) + [f"{0:5d} {-0.375:25.15f} {0.0:25.15f}"])
    write(out, "indh2.def",
          header("NDoublonHolon2siteIdx", n2)
          + [f"{k:5d} {((-1) ** k) * (k % 6 + 1) / 32:25.15f} {0.0:25.15f}" for k in range(6 * n2)])
    write(out, "indh4.def",
          header("NDoublonHolon4siteIdx", n4)
          + [f"{k:5d} {((-1) ** (k // 2)) * (k % 10 + 1) / 64:25.15f} {0.0:25.15f}"
             for k in range(10 * n4)])
    files += [("InGutzwiller", "ingutzwiller.def"), ("InDH2", "indh2.def"),
              ("InDH4", "indh4.def")]
    write(out, "namelist.def", [f"{k:>16}  {v}" for k, v in files])


generate("hubbard_dh_L16", 16, 12, 1, dh2_shells=[1, 2], dh4_shells=[(1, 2)])
generate("hubbard_dh_L128", 128, 128, 0, dh2_shells=[1], dh4_shells=[(1, 2)])
