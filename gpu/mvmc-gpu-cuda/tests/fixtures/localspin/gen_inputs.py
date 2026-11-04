"""Generate the resident GPU local-spin inputs (#572 milestone 1).

uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/localspin/gen_inputs.py

Writes two deterministic Expert-mode inputs next to this script:

* heisenberg_chain_L64: 64 local spins on a periodic chain (Ncond=0,
  NExUpdatePath=2, every proposal is an exchange), 2Ne=64.
* kondo_chain_L64: 64 conduction sites (0..63, periodic hopping, U=4) and 64
  local spins (64..127) coupled site by site, Ncond=64, NExUpdatePath=1
  (hopping/exchange mixture), 2Ne=128 (delayed sampling updates in Opt).

Real normal orbitals, Gutzwiller/Jastrow, identity TransSym and NSPGaussLeg=8.
The spin couplings follow the StdFace Heisenberg sign convention of
tests/fixtures/original_heisenberg_parser_184 (CoulombInter -J/4, Hund -J/2,
Exchange -J/2). These are engine-test inputs, not physics benchmarks.
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


def chain_distance(a, b, length):
    d = abs(a - b) % length
    return min(d, length - d)


def generate(name, length, kondo):
    out = HERE / name
    out.mkdir(exist_ok=True)
    ns = 2 * length if kondo else length
    local = [(i >= length) if kondo else True for i in range(ns)]
    cls = [1 if (loc and kondo) else 0 for loc in local]
    x = [i % length for i in range(ns)]
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, length if kondo else 0, 1 if kondo else 2))
    write(
        out,
        "locspn.def",
        header("NlocalSpin", sum(local)) + [f"{i:5d} {int(loc):6d}" for i, loc in enumerate(local)],
    )
    if kondo:
        trans = []
        for i in range(length):
            j = (i + 1) % length
            for s in (0, 1):
                trans.append(f"{j:5d} {s:5d} {i:5d} {s:5d} {1.0:25.15f} {0.0:25.15f}")
                trans.append(f"{i:5d} {s:5d} {j:5d} {s:5d} {1.0:25.15f} {-0.0:25.15f}")
        write(out, "trans.def", header("NTransfer", len(trans)) + trans)
        write(
            out,
            "coulombintra.def",
            header("NCoulombIntra", length) + [f"{i:5d} {4.0:25.15f}" for i in range(length)],
        )
        files += [("Trans", "trans.def"), ("CoulombIntra", "coulombintra.def")]
        bonds = [(i, length + i) for i in range(length)]
    else:
        bonds = [(i, (i + 1) % length) for i in range(length)]
    for keyword, filename, value in [
        ("NCoulombInter", "coulombinter.def", -0.25),
        ("NHund", "hund.def", -0.5),
        ("NExchange", "exchange.def", -0.5),
    ]:
        write(
            out,
            filename,
            header(keyword, len(bonds)) + [f"{i:5d} {j:5d} {value:25.15f}" for i, j in bonds],
        )
    files += [("CoulombInter", "coulombinter.def"), ("Hund", "hund.def"),
              ("Exchange", "exchange.def")]
    a, b = (0, length) if kondo else (0, 1)
    one = [(i, s, i, s) for i in sorted({0, 1, a + 2, b + 2}) for s in (0, 1)]
    one += [(0, 0, 1, 0), (1, 1, 0, 1), (a, 0, b, 0), (b, 1, a, 1)]
    write(out, "greenone.def",
          header("NCisAjs", len(one)) + ["".join(f"{v:6d}" for v in t) for t in one])
    two = [
        (a, 0, a, 0, b, 0, b, 0),
        (a, 0, a, 0, b, 1, b, 1),
        (a, 0, b, 0, b, 1, a, 1),
        (b, 1, a, 1, a, 0, b, 0),
        (0, 1, 0, 1, 2, 1, 2, 1),
        (1, 0, 1, 0, 1, 1, 1, 1),
    ]
    write(out, "greentwo.def",
          header("NCisAjsCktAltDC", len(two)) + ["".join(f"{v:6d}" for v in t) for t in two])
    files += [("OneBodyG", "greenone.def"), ("TwoBodyG", "greentwo.def")]
    ngutz = 2 if kondo else 1
    write(
        out,
        "gutzwilleridx.def",
        header("NGutzwillerIdx", ngutz, ["ComplexType          0"])
        + [f"{i:5d} {cls[i]:6d}" for i in range(ns)]
        + [f"{k:5d} {1:6d}" for k in range(ngutz)],
    )
    half = length // 2

    def jastrow(i, j):
        d = chain_distance(x[i], x[j], length)
        if not kondo:
            return d - 1
        return [d - 1, half + d, 2 * half + 1 + d - 1][cls[i] + cls[j]]

    nj = 3 * half + 1 if kondo else half
    rows = [f"{i:5d} {j:6d} {jastrow(i, j):6d}" for i in range(ns) for j in range(ns) if i != j]
    write(
        out,
        "jastrowidx.def",
        header("NJastrowIdx", nj, ["ComplexType          0"])
        + rows
        + [f"{k:5d} {1:6d}" for k in range(nj)],
    )
    no = (4 if kondo else 1) * length
    rows = [
        f"{i:5d} {j:6d} {(2 * cls[i] + cls[j]) * length + (x[j] - x[i]) % length:6d}"
        for i in range(ns)
        for j in range(ns)
    ]
    write(
        out,
        "orbitalidx.def",
        header("NOrbitalIdx", no, ["ComplexType          0"])
        + rows
        + [f"{k:5d} {1:6d}" for k in range(no)],
    )
    write(
        out,
        "qptransidx.def",
        header("NQPTrans", 1) + ["0    1.00000"] + [f"{0:5d} {i:6d} {i:6d} {1:6d}" for i in range(ns)],
    )
    files += [("Gutzwiller", "gutzwilleridx.def"), ("Jastrow", "jastrowidx.def"),
              ("Orbital", "orbitalidx.def"), ("TransSym", "qptransidx.def")]
    write(out, "namelist.def", [f"{k:>16}  {v}" for k, v in files])


generate("heisenberg_chain_L64", 64, kondo=False)
generate("kondo_chain_L64", 64, kondo=True)
