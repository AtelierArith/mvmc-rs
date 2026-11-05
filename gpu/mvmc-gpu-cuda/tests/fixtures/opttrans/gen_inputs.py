"""Generate the resident GPU OptTrans inputs (#674, #572 milestone 5).

uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/opttrans/gen_inputs.py

Writes two deterministic Expert-mode Hubbard chains next to this script:

* hubbard_opttrans_L16: 16 sites, Ncond=12 (2Ne=12, doped), NExUpdatePath=1
  (hopping/exchange mixture).
* hubbard_opttrans_L128: 128 sites, Ncond=128 (2Ne=128: Opt delayed sampling
  queue and paired inverse), NExUpdatePath=0.

Both declare NQPOptTrans=3 optimized translation sectors (C readdef.c
GetInfoOptTrans): the identity, the reflection r -> L-1-r and the affine
permutation r -> (5r+3) mod L. Every sign is +1, as C forces for periodic
TransSym (APFlag=0). The initial weights 0.75, 0.5 and -0.375 are binary
fractions; the resident and C-order setups normalize them to maximum amplitude
one (C SyncModifiedParameter). The orbitals depend on (j-i) mod L only, so a
pure translation sector would reproduce the identity plane; the reflection and
the affine map give distinct Slater planes and so distinct OptTrans
derivatives.

The chains are periodic with Transfer t=1, CoulombIntra U=4, PairHop 0.3 and
Exchange 0.25. Gutzwiller/Jastrow factors, identity TransSym and
NSPGaussLeg=8 give NQPFull = 8 x 1 x 3 = 24 planes. These are engine-test
inputs, not physics benchmarks.
"""
import hashlib
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


def sectors(length):
    return [
        list(range(length)),
        [length - 1 - r for r in range(length)],
        [(5 * r + 3) % length for r in range(length)],
    ]


WEIGHTS = [0.75, 0.5, -0.375]


def generate(name, length, ncond, path):
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
    files += [("Gutzwiller", "gutzwilleridx.def"), ("Jastrow", "jastrowidx.def")]
    write(out, "orbitalidx.def",
          header("NOrbitalIdx", length, ["ComplexType          0"])
          + [f"{i:5d} {j:6d} {(j - i) % length:6d}" for i in range(ns) for j in range(ns)]
          + [f"{k:5d} {1:6d}" for k in range(length)])
    write(out, "qptransidx.def",
          header("NQPTrans", 1) + ["0    1.00000"]
          + [f"{0:5d} {i:6d} {i:6d} {1:6d}" for i in range(ns)])
    files += [("Orbital", "orbitalidx.def"), ("TransSym", "qptransidx.def")]
    maps = sectors(length)
    for m in maps:
        assert sorted(m) == list(range(length))
    write(out, "opttrans.def",
          header("NQPOptTrans", len(maps), ["ComplexType 0"])
          + [f"{k} {w}" for k, w in enumerate(WEIGHTS)]
          + [f"{k:5d} {r:6d} {m[r]:6d} {1:6d}" for k, m in enumerate(maps) for r in range(ns)])
    files.append(("OptTrans", "opttrans.def"))
    # Initial factors: exact binary fractions of moderate size.
    write(out, "ingutzwiller.def",
          header("NGutzwillerIdx", 1) + [f"{0:5d} {-0.375:25.15f} {0.0:25.15f}"])
    files.append(("InGutzwiller", "ingutzwiller.def"))
    write(out, "namelist.def", [f"{k:>24}  {v}" for k, v in files])
    names = sorted(p.name for p in out.iterdir() if p.name != "inputs.sha256")
    write(out, "inputs.sha256",
          [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in names])


generate("hubbard_opttrans_L16", 16, 12, 1)
generate("hubbard_opttrans_L128", 128, 128, 0)
