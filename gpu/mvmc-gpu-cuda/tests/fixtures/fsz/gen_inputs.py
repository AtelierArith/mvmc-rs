"""Generate the resident GPU FSZ / general-orbital inputs (#714, #572 milestone 7).

uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/fsz/gen_inputs.py

Writes six deterministic Expert-mode inputs next to this script:

* hubbard_fsz_L16: 16-site Hubbard chain, Ncond=12 (2Ne=12, doped), 2Sz not
  set (C's default -1, Sz not conserved), NExUpdatePath=1 (hop/exchange
  mixture), OrbitalGeneral.
* hubbard_fsz_L128: 128 sites, Ncond=128 (2Ne=128), 2Sz not set,
  NExUpdatePath=0, OrbitalGeneral. 2Ne >= 128 exercises the Opt delayed queue
  and the paired factor.
* kondo_fsz_2sz2: Kondo chain with 8 conduction sites (0-7) and 8 local spins
  (8-15), Ncond=6 (2Ne=14), 2Sz=2 (8 up, 6 down), NExUpdatePath=1,
  OrbitalAntiParallel + OrbitalParallel (C selects general orbitals).
* kondo_fsz_kgc: the same Kondo chain with 2Sz not set and NExUpdatePath=3
  (C KondoGC: hop 1/2, exchange 1/4, local spin flip 1/4), OrbitalGeneral,
  and a spin-flip conduction hopping.
* kondo_fsz_kgc_L128: 64 conduction sites and 64 local spins, Ncond=64
  (2Ne=128), as kondo_fsz_kgc, so local spin flips run through the Opt
  delayed queue and the paired factor.
* heisenberg_fsz_L16: 16 local spins (2Ne=16), 2Sz not set,
  NExUpdatePath=2 (C: exchange or local spin flip for 2Sz=-1),
  OrbitalGeneral, Heisenberg exchange plus a Hermitian pair of
  spin-changing InterAll terms (S+_i S+_j type) that break Sz conservation.

C reads 2Sz=-1 only as the default: an explicit "2Sz -1" is an error
(readdef.c:2215-2218), so the Sz-free inputs omit the 2Sz line.

The Sz-free Hubbard chains carry a spin-flip Transfer
lambda (c+_{j dn} c_{i up} - c+_{j up} c_{i dn} + h.c.) on every bond, and a
Hermitian pair of spin-changing InterAll terms (S+_0 S-_1 + S-_0 S+_1 type)
next to a spin-conserving pair. OneBodyG, TwoBodyG and TwoBodyGEx include
spin-flip entries. General orbitals are indexed by the class (spin_i, spin_j,
(j - i) mod L) of each pair rsi < rsj, so 3L-2 Slater parameters. Two
TransSym elements: the identity (weight 1) and the affine permutation
r -> (5r+3) mod L (weight 0.5), as in ../complex. The fixed-2Sz Kondo chain
has spin-conserving Hamiltonian terms (C GetInfoInterAll requires them for a
fixed 2Sz) but measures spin-flip Green functions. NSPGaussLeg=1: C forces it
for general orbitals (readdef.c:703-707). Real Gutzwiller/Jastrow factors.
These are engine-test inputs, not physics benchmarks.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = "=" * 45
LAMBDA = 0.35
QP_WEIGHT = 0.5


def header(name, count, extra=()):
    """Five header lines: an index file carries ComplexType instead of a title."""
    title = BAR if extra else f"{'=' * 10} {name} {'=' * 10}"
    return [BAR, f"{name} {count:>10}", *extra, BAR, title, BAR][:5]


def write(directory, name, lines):
    (directory / name).write_text("\n".join(lines) + "\n")


def modpara(nsite, ncond, path, two_sz):
    sz = [] if two_sz is None else [f"2Sz            {two_sz}"]
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
Ncond          {ncond}""".splitlines() + sz + f"""NSPGaussLeg    1
NSPStot        0
NMPTrans       2
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


def rows(items):
    return ["".join(f"{v:6d}" for v in t) for t in items]


def projection(out, ns, files, ring):
    write(out, "gutzwilleridx.def",
          header("NGutzwillerIdx", 1, ["ComplexType          0"])
          + [f"{i:5d} {0:6d}" for i in range(ns)] + [f"{0:5d} {1:6d}"])
    half = max(ring // 2, 1)
    jast = [f"{i:5d} {j:6d} {(distance(i % ring, j % ring, ring) + ring - 1) % half:6d}"
            for i in range(ns) for j in range(ns) if i != j]
    write(out, "jastrowidx.def",
          header("NJastrowIdx", half, ["ComplexType          0"])
          + jast + [f"{k:5d} {1:6d}" for k in range(half)])
    files += [("Gutzwiller", "gutzwilleridx.def"), ("Jastrow", "jastrowidx.def")]
    write(out, "ingutzwiller.def",
          header("NGutzwillerIdx", 1) + [f"{0:5d} {-0.375:25.15f} {0.0:25.15f}"])
    files.append(("InGutzwiller", "ingutzwiller.def"))


def transsym(out, ns, files):
    affine = [(5 * r + 3) % ns for r in range(ns)]
    write(out, "qptransidx.def",
          header("NQPTrans", 2) + [f"0 {1.0:.5f} {0.0:.5f}", f"1 {QP_WEIGHT:.5f} {0.0:.5f}"]
          + [f"{0:5d} {i:6d} {i:6d} {1:6d}" for i in range(ns)]
          + [f"{1:5d} {i:6d} {affine[i]:6d} {1:6d}" for i in range(ns)])
    files.append(("TransSym", "qptransidx.def"))


def finish(out, files):
    write(out, "namelist.def", [f"{k:>24}  {v}" for k, v in files])
    names = sorted(p.name for p in out.iterdir() if p.name != "inputs.sha256")
    write(out, "inputs.sha256",
          [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in names])


def hubbard(name, length, ncond, path):
    out = HERE / name
    out.mkdir(exist_ok=True)
    ns = length
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, ncond, path, None))
    write(out, "locspn.def",
          header("NlocalSpin", 0) + [f"{i:5d} {0:6d}" for i in range(ns)])
    trans = []
    for i in range(ns):
        j = (i + 1) % ns
        for sp in (0, 1):
            trans.append(f"{j:5d} {sp:5d} {i:5d} {sp:5d} {1.0:25.15f} {0.0:25.15f}")
            trans.append(f"{i:5d} {sp:5d} {j:5d} {sp:5d} {1.0:25.15f} {0.0:25.15f}")
        # Real spin-flip hopping and its Hermitian conjugate (C GreenFunc1_fsz2).
        for (a, sa, b, sb, v) in ((j, 1, i, 0, LAMBDA), (j, 0, i, 1, -LAMBDA)):
            trans.append(f"{a:5d} {sa:5d} {b:5d} {sb:5d} {v:25.15f} {0.0:25.15f}")
            trans.append(f"{b:5d} {sb:5d} {a:5d} {sa:5d} {v:25.15f} {0.0:25.15f}")
    write(out, "trans.def", header("NTransfer", len(trans)) + trans)
    write(out, "coulombintra.def",
          header("NCoulombIntra", ns) + [f"{i:5d} {4.0:25.15f}" for i in range(ns)])
    bonds = [(i, (i + 1) % ns) for i in range(ns)]
    write(out, "pairhop.def",
          header("NPairHop", len(bonds)) + [f"{i:5d} {j:5d} {0.3:25.15f}" for i, j in bonds])
    write(out, "exchange.def",
          header("NExchange", len(bonds)) + [f"{i:5d} {j:5d} {0.25:25.15f}" for i, j in bonds])
    # c+_{0 up} c_{0 dn} c+_{1 dn} c_{1 up} (S+_0 S-_1) with 0.2 and its
    # Hermitian conjugate, plus a spin-conserving Hermitian pair.
    inter = [
        f"{0:5d} {0:5d} {0:5d} {1:5d} {1:5d} {1:5d} {1:5d} {0:5d} {0.2:25.15f} {0.0:25.15f}",
        f"{1:5d} {0:5d} {1:5d} {1:5d} {0:5d} {1:5d} {0:5d} {0:5d} {0.2:25.15f} {0.0:25.15f}",
        f"{0:5d} {0:5d} {1:5d} {0:5d} {2:5d} {1:5d} {3:5d} {1:5d} {0.15:25.15f} {0.0:25.15f}",
        f"{3:5d} {1:5d} {2:5d} {1:5d} {1:5d} {0:5d} {0:5d} {0:5d} {0.15:25.15f} {0.0:25.15f}",
    ]
    write(out, "interall.def", header("NInterAll", len(inter)) + inter)
    files += [("Trans", "trans.def"), ("CoulombIntra", "coulombintra.def"),
              ("PairHop", "pairhop.def"), ("Exchange", "exchange.def"),
              ("InterAll", "interall.def")]
    one = [(i, s_, i, s_) for i in (0, 1, 2) for s_ in (0, 1)]
    one += [(0, 0, 1, 0), (1, 1, 0, 1), (0, 0, 0, 1), (1, 1, 1, 0), (2, 0, 1, 1), (0, 1, 3, 0)]
    write(out, "greenone.def", header("NCisAjs", len(one)) + rows(one))
    two = [
        (0, 0, 0, 0, 1, 1, 1, 1),
        (0, 0, 1, 0, 1, 1, 0, 1),
        (0, 0, 0, 1, 1, 1, 1, 0),
        (0, 1, 1, 0, 2, 0, 3, 1),
        (2, 1, 2, 0, 0, 0, 1, 0),
        (3, 0, 2, 0, 3, 1, 2, 1),
    ]
    write(out, "greentwo.def", header("NCisAjsCktAltDC", len(two)) + rows(two))
    # Factored products <c+_i c_j> conj(<c+_k c_l>) (C calgrn_fsz.c), with
    # spin-flip factors.
    gex = [
        (0, 0, 1, 0, 1, 1, 0, 1),
        (0, 0, 0, 1, 1, 1, 1, 0),
        (2, 1, 1, 0, 0, 0, 3, 1),
    ]
    write(out, "greentwoex.def", header("NTwoBodyGEx", len(gex)) + rows(gex))
    files += [("OneBodyG", "greenone.def"), ("TwoBodyG", "greentwo.def"),
              ("TwoBodyGEx", "greentwoex.def")]
    projection(out, ns, files, length)
    general_orbitals(out, ns, files)
    transsym(out, ns, files)
    finish(out, files)


def general_orbitals(out, ns, files):
    """One Slater class per (spin_i, spin_j, (j - i) mod ns) of the pairs
    rsi < rsj, numbered in order of first appearance."""
    classes = {}
    mapping = []
    for rsi in range(2 * ns):
        for rsj in range(rsi + 1, 2 * ns):
            key = (rsi // ns, rsj // ns, (rsj % ns - rsi % ns) % ns)
            k = classes.setdefault(key, len(classes))
            mapping.append(f"{rsi % ns:5d} {rsi // ns:3d} {rsj % ns:6d} {rsj // ns:3d} {k:6d} {1:3d}")
    write(out, "orbitalidxgen.def",
          header("NOrbitalIdx", len(classes), ["ComplexType          0"])
          + mapping + [f"{k:5d} {1:6d}" for k in range(len(classes))])
    files.append(("OrbitalGeneral", "orbitalidxgen.def"))


def kondo(name, ring, ncond, path, two_sz, general=False):
    out = HERE / name
    out.mkdir(exist_ok=True)
    ns = 2 * ring
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, ncond, path, two_sz))
    write(out, "locspn.def",
          header("NlocalSpin", ring) + [f"{i:5d} {int(i >= ring):6d}" for i in range(ns)])
    trans = []
    for i in range(ring):
        j = (i + 1) % ring
        for sp in (0, 1):
            trans.append(f"{j:5d} {sp:5d} {i:5d} {sp:5d} {1.0:25.15f} {0.0:25.15f}")
            trans.append(f"{i:5d} {sp:5d} {j:5d} {sp:5d} {1.0:25.15f} {0.0:25.15f}")
        if two_sz is None:
            # Sz-free inputs: the spin-flip conduction hopping of the chains.
            for (a, sa, b, sb, v) in ((j, 1, i, 0, LAMBDA), (j, 0, i, 1, -LAMBDA)):
                trans.append(f"{a:5d} {sa:5d} {b:5d} {sb:5d} {v:25.15f} {0.0:25.15f}")
                trans.append(f"{b:5d} {sb:5d} {a:5d} {sa:5d} {v:25.15f} {0.0:25.15f}")
    write(out, "trans.def", header("NTransfer", len(trans)) + trans)
    write(out, "coulombintra.def",
          header("NCoulombIntra", ring) + [f"{i:5d} {4.0:25.15f}" for i in range(ring)])
    kondo_bonds = [(i, i + ring) for i in range(ring)]
    write(out, "exchange.def",
          header("NExchange", ring) + [f"{i:5d} {j:5d} {0.5:25.15f}" for i, j in kondo_bonds])
    write(out, "hund.def",
          header("NHund", ring) + [f"{i:5d} {j:5d} {0.25:25.15f}" for i, j in kondo_bonds])
    write(out, "coulombinter.def",
          header("NCoulombInter", ring) + [f"{i:5d} {j:5d} {-0.125:25.15f}" for i, j in kondo_bonds])
    files += [("Trans", "trans.def"), ("CoulombIntra", "coulombintra.def"),
              ("Exchange", "exchange.def"), ("Hund", "hund.def"),
              ("CoulombInter", "coulombinter.def")]
    r = ring
    one = [(i, s_, i, s_) for i in (0, 1, r) for s_ in (0, 1)]
    one += [(0, 0, 1, 0), (1, 1, 0, 1), (r, 0, r, 1), (0, 1, 0, 0), (r + 1, 0, 1, 1)]
    write(out, "greenone.def", header("NCisAjs", len(one)) + rows(one))
    two = [
        (0, 0, 0, 0, r, 1, r, 1),
        (0, 0, r, 0, r, 1, 0, 1),
        (r, 0, r, 1, 0, 1, 0, 0),
        (1, 0, 0, 0, r + 1, 1, r + 1, 1),
    ]
    write(out, "greentwo.def", header("NCisAjsCktAltDC", len(two)) + rows(two))
    files += [("OneBodyG", "greenone.def"), ("TwoBodyG", "greentwo.def")]
    projection(out, ns, files, ns)
    if general:
        general_orbitals(out, ns, files)
        transsym(out, ns, files)
        finish(out, files)
        return
    write(out, "orbitalidx.def",
          header("NOrbitalIdx", ns, ["ComplexType          0"])
          + [f"{i:5d} {j:6d} {(j - i) % ns:6d} {1:3d}" for i in range(ns) for j in range(ns)]
          + [f"{k:5d} {1:6d}" for k in range(ns)])
    write(out, "orbitalidxpara.def",
          header("NOrbitalIdx", ns - 1, ["ComplexType          0"])
          + [f"{i:5d} {j:6d} {j - i - 1:6d} {1:3d}" for i in range(ns) for j in range(i + 1, ns)]
          + [f"{k:5d} {1:6d}" for k in range(ns - 1)])
    files += [("Orbital", "orbitalidx.def"), ("OrbitalParallel", "orbitalidxpara.def")]
    transsym(out, ns, files)
    finish(out, files)


def heisenberg(name, ns, path):
    out = HERE / name
    out.mkdir(exist_ok=True)
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, 0, path, None))
    write(out, "locspn.def",
          header("NlocalSpin", ns) + [f"{i:5d} {1:6d}" for i in range(ns)])
    bonds = [(i, (i + 1) % ns) for i in range(ns)]
    write(out, "exchange.def",
          header("NExchange", ns) + [f"{i:5d} {j:5d} {0.5:25.15f}" for i, j in bonds])
    write(out, "hund.def",
          header("NHund", ns) + [f"{i:5d} {j:5d} {0.5:25.15f}" for i, j in bonds])
    write(out, "coulombinter.def",
          header("NCoulombInter", ns) + [f"{i:5d} {j:5d} {-0.25:25.15f}" for i, j in bonds])
    # c+_{i up} c_{i dn} c+_{j up} c_{j dn} (S+_i S+_j) with 0.15 and its
    # Hermitian conjugate (S-_j S-_i) on two bonds: Sz is not conserved.
    inter = []
    for i, j in bonds[:2]:
        inter.append(f"{i:5d} {0:5d} {i:5d} {1:5d} {j:5d} {0:5d} {j:5d} {1:5d} {0.15:25.15f} {0.0:25.15f}")
        inter.append(f"{j:5d} {1:5d} {j:5d} {0:5d} {i:5d} {1:5d} {i:5d} {0:5d} {0.15:25.15f} {0.0:25.15f}")
    write(out, "interall.def", header("NInterAll", len(inter)) + inter)
    files += [("Exchange", "exchange.def"), ("Hund", "hund.def"),
              ("CoulombInter", "coulombinter.def"), ("InterAll", "interall.def")]
    one = [(i, s_, i, s_) for i in (0, 1) for s_ in (0, 1)] + [(0, 0, 0, 1), (1, 1, 1, 0)]
    write(out, "greenone.def", header("NCisAjs", len(one)) + rows(one))
    two = [(0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1, 1, 0), (0, 0, 0, 1, 1, 0, 1, 1)]
    write(out, "greentwo.def", header("NCisAjsCktAltDC", len(two)) + rows(two))
    files += [("OneBodyG", "greenone.def"), ("TwoBodyG", "greentwo.def")]
    projection(out, ns, files, ns)
    general_orbitals(out, ns, files)
    transsym(out, ns, files)
    finish(out, files)


hubbard("hubbard_fsz_L16", 16, 12, 1)
hubbard("hubbard_fsz_L128", 128, 128, 0)
kondo("kondo_fsz_2sz2", 8, 6, 1, 2)
kondo("kondo_fsz_kgc", 8, 6, 3, None, general=True)
kondo("kondo_fsz_kgc_L128", 64, 64, 3, None, general=True)
heisenberg("heisenberg_fsz_L16", 16, 2)
