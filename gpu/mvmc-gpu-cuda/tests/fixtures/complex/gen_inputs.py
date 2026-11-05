"""Generate the resident GPU complex normal-orbital inputs (#678, #572 milestone 6).

uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/complex/gen_inputs.py

Writes two deterministic Expert-mode Hubbard chains next to this script:

* hubbard_complex_L16: 16 sites, Ncond=12 (2Ne=12, doped), NExUpdatePath=1
  (hopping/exchange mixture).
* hubbard_complex_L128: 128 sites, Ncond=128 (2Ne=128), NExUpdatePath=0.

Both declare complex orbitals (Orbital ComplexType 1, so C AllComplexFlag=1),
a complex Transfer phase t exp(i phi) with phi=0.25 on every bond (Hermitian
reverse entry), a Hermitian pair of complex spin-conserving InterAll terms, and
two TransSym elements: the identity with weight 1 and the affine permutation
r -> (5r+3) mod L with the complex weight 0.6+0.8i (C ParaQPTrans, readdef.c
GetInfoTransSym). The orbitals depend on (j-i) mod L only, so a translation
would repeat the identity plane, and the reflection r -> L-1-r transposes the
orbital, which the total-spin projection maps back onto the identity planes
(the weight would then only scale the overlap). The affine map gives an
independent plane, so the weight's phase changes Metropolis decisions.
NSPGaussLeg=8 gives 8 x 2 = 16 QP planes.
CoulombIntra U=4, PairHop 0.3 and Exchange 0.25; real Gutzwiller/Jastrow
factors; OneBodyG, TwoBodyG and TwoBodyGEx measurements. These are
engine-test inputs, not physics benchmarks.
"""
import hashlib
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = "=" * 45
PHI = 0.25
QP_WEIGHT = (0.6, 0.8)
INTER_ALL = (0.2, 0.1)


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


def generate(name, length, ncond, path):
    out = HERE / name
    out.mkdir(exist_ok=True)
    ns = length
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, ncond, path))
    write(out, "locspn.def",
          header("NlocalSpin", 0) + [f"{i:5d} {0:6d}" for i in range(ns)])
    c, s = math.cos(PHI), math.sin(PHI)
    trans = []
    for i in range(ns):
        j = (i + 1) % ns
        for sp in (0, 1):
            trans.append(f"{j:5d} {sp:5d} {i:5d} {sp:5d} {c:25.15f} {s:25.15f}")
            trans.append(f"{i:5d} {sp:5d} {j:5d} {sp:5d} {c:25.15f} {-s:25.15f}")
    write(out, "trans.def", header("NTransfer", len(trans)) + trans)
    write(out, "coulombintra.def",
          header("NCoulombIntra", ns) + [f"{i:5d} {4.0:25.15f}" for i in range(ns)])
    bonds = [(i, (i + 1) % ns) for i in range(ns)]
    write(out, "pairhop.def",
          header("NPairHop", len(bonds)) + [f"{i:5d} {j:5d} {0.3:25.15f}" for i, j in bonds])
    write(out, "exchange.def",
          header("NExchange", len(bonds)) + [f"{i:5d} {j:5d} {0.25:25.15f}" for i, j in bonds])
    # c+_{0 up} c_{1 up} c+_{2 dn} c_{3 dn} with V, and its Hermitian conjugate
    # c+_{3 dn} c_{2 dn} c+_{1 up} c_{0 up} with conj(V).
    vr, vi = INTER_ALL
    inter = [
        f"{0:5d} {0:5d} {1:5d} {0:5d} {2:5d} {1:5d} {3:5d} {1:5d} {vr:25.15f} {vi:25.15f}",
        f"{3:5d} {1:5d} {2:5d} {1:5d} {1:5d} {0:5d} {0:5d} {0:5d} {vr:25.15f} {-vi:25.15f}",
    ]
    write(out, "interall.def", header("NInterAll", len(inter)) + inter)
    files += [("Trans", "trans.def"), ("CoulombIntra", "coulombintra.def"),
              ("PairHop", "pairhop.def"), ("Exchange", "exchange.def"),
              ("InterAll", "interall.def")]
    one = [(i, s_, i, s_) for i in (0, 1, 2) for s_ in (0, 1)]
    one += [(0, 0, 1, 0), (1, 1, 0, 1), (2, 0, 0, 0), (0, 1, 3, 1)]
    write(out, "greenone.def", header("NCisAjs", len(one)) + rows(one))
    two = [
        (0, 0, 0, 0, 1, 1, 1, 1),
        (0, 0, 1, 0, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 1, 0, 1),
        (0, 0, 1, 0, 2, 1, 3, 1),
        (2, 1, 2, 1, 0, 0, 1, 0),
        (3, 0, 2, 0, 3, 1, 2, 1),
    ]
    write(out, "greentwo.def", header("NCisAjsCktAltDC", len(two)) + rows(two))
    # Factored products <c+_i c_j> conj(<c+_k c_l>) (C calgrn.c).
    gex = [
        (0, 0, 1, 0, 1, 1, 0, 1),
        (1, 0, 2, 0, 3, 1, 2, 1),
        (2, 1, 2, 1, 0, 0, 1, 0),
    ]
    write(out, "greentwoex.def", header("NTwoBodyGEx", len(gex)) + rows(gex))
    files += [("OneBodyG", "greenone.def"), ("TwoBodyG", "greentwo.def"),
              ("TwoBodyGEx", "greentwoex.def")]
    write(out, "gutzwilleridx.def",
          header("NGutzwillerIdx", 1, ["ComplexType          0"])
          + [f"{i:5d} {0:6d}" for i in range(ns)] + [f"{0:5d} {1:6d}"])
    half = length // 2
    jast = [f"{i:5d} {j:6d} {distance(i, j, length) - 1:6d}"
            for i in range(ns) for j in range(ns) if i != j]
    write(out, "jastrowidx.def",
          header("NJastrowIdx", half, ["ComplexType          0"])
          + jast + [f"{k:5d} {1:6d}" for k in range(half)])
    files += [("Gutzwiller", "gutzwilleridx.def"), ("Jastrow", "jastrowidx.def")]
    write(out, "orbitalidx.def",
          header("NOrbitalIdx", length, ["ComplexType          1"])
          + [f"{i:5d} {j:6d} {(j - i) % length:6d}" for i in range(ns) for j in range(ns)]
          + [f"{k:5d} {1:6d}" for k in range(length)])
    wr, wi = QP_WEIGHT
    affine = [(5 * r + 3) % length for r in range(length)]
    write(out, "qptransidx.def",
          header("NQPTrans", 2) + [f"0 {1.0:.5f} {0.0:.5f}", f"1 {wr:.5f} {wi:.5f}"]
          + [f"{0:5d} {i:6d} {i:6d} {1:6d}" for i in range(ns)]
          + [f"{1:5d} {i:6d} {affine[i]:6d} {1:6d}" for i in range(ns)])
    files += [("Orbital", "orbitalidx.def"), ("TransSym", "qptransidx.def")]
    write(out, "ingutzwiller.def",
          header("NGutzwillerIdx", 1) + [f"{0:5d} {-0.375:25.15f} {0.0:25.15f}"])
    files.append(("InGutzwiller", "ingutzwiller.def"))
    write(out, "namelist.def", [f"{k:>24}  {v}" for k, v in files])
    names = sorted(p.name for p in out.iterdir() if p.name != "inputs.sha256")
    write(out, "inputs.sha256",
          [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in names])


generate("hubbard_complex_L16", 16, 12, 1)
generate("hubbard_complex_L128", 128, 128, 0)
