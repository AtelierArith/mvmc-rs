"""Generate the resident GPU RBM inputs (#667, #572 milestone 3).

uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/rbm/gen_inputs.py

Writes two deterministic Expert-mode Hubbard chains next to this script:

* hubbard_rbm_L16: 16 sites, Ncond=12 (2Ne=12, doped), NExUpdatePath=1
  (hopping/exchange mixture); 4 charge, 4 spin and 2 general hidden neurons.
* hubbard_rbm_L128: 128 sites, Ncond=128 (2Ne=128: Opt delayed sampling queue
  and paired inverse), NExUpdatePath=0; 4 charge, 4 spin and 2 general neurons.

Every one of the nine RBM sections is present and completely mapped, so the
C reader (readdef.c GetInfoRBM_*) accepts them: a physical-layer row for
every site (General: every site and spin), a hidden-layer row for every
neuron, a physical-hidden row for every (site[, spin], neuron), and one
optimization flag per declared coefficient. Indices are tied on purpose:

* ChargeRBM_PhysLayer: one coefficient; SpinRBM_PhysLayer: r % 2;
  GeneralRBM_PhysLayer: the spin.
* Hidden layers: Charge h % 2, Spin h, General 0.
* ChargeRBM_PhysHidden: (r + h*L/4) % L (translated filters, tied across
  neurons); SpinRBM_PhysHidden: (r + h*L/4) % L; GeneralRBM_PhysHidden:
  (r + s*L + h) % (2L).

All coefficients are real (ComplexType 0) and every flag is 1. The In*RBM
overlays give moderate real values (binary fractions), so the RBM factor
changes Metropolis decisions and operator ratios; C's 1e-2/Nneuron random
initialization would barely move them. The chains have periodic
nearest-neighbour Transfer t=1, CoulombIntra U=4, PairHop 0.3 and Exchange
0.25, so the local energy contains one- and two-electron RBM ratios. There is
no InterAll: C's real InterAll Green kernel omits RBM factors. Real normal
translational orbitals, Gutzwiller/Jastrow, identity TransSym and
NSPGaussLeg=8. These are engine-test inputs, not physics benchmarks.
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


def modpara(nsite, ncond, path, neurons):
    charge, spin, general = neurons
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
NSRCG          0
NneuronCharge  {charge}
NneuronSpin    {spin}
NneuronGeneral {general}""".splitlines()


def distance(a, b, length):
    d = abs(a - b) % length
    return min(d, length - d)


def index_file(out, name, keyword, rows, width):
    write(out, name, header(keyword, width, ["ComplexType          0"])
          + rows + [f"{k:5d} {1:5d}" for k in range(width)])


def overlay(out, name, keyword, values):
    write(out, name, header(keyword, len(values))
          + [f"{k:5d} {v:25.15f} {0.0:25.15f}" for k, v in enumerate(values)])


def generate(name, length, ncond, path, neurons):
    out = HERE / name
    out.mkdir(exist_ok=True)
    ns = length
    nc, nspin, ngen = neurons
    files = [("ModPara", "modpara.def"), ("LocSpin", "locspn.def")]
    write(out, "modpara.def", modpara(ns, ncond, path, neurons))
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

    # Nine complete RBM sections (C readdef.c GetInfoRBM_Layer/_PhysHidden).
    shift = max(1, length // 4)
    sections = [
        ("ChargeRBM_PhysLayer", "rbm_charge_phys.def",
         [f"{r:5d} {0:5d}" for r in range(ns)], 1),
        ("SpinRBM_PhysLayer", "rbm_spin_phys.def",
         [f"{r:5d} {r % 2:5d}" for r in range(ns)], 2),
        ("GeneralRBM_PhysLayer", "rbm_general_phys.def",
         [f"{r:5d} {s:5d} {s:5d}" for s in (0, 1) for r in range(ns)], 2),
        ("ChargeRBM_HiddenLayer", "rbm_charge_hidden.def",
         [f"{h:5d} {h % 2:5d}" for h in range(nc)], 2),
        ("SpinRBM_HiddenLayer", "rbm_spin_hidden.def",
         [f"{h:5d} {h:5d}" for h in range(nspin)], nspin),
        ("GeneralRBM_HiddenLayer", "rbm_general_hidden.def",
         [f"{h:5d} {0:5d}" for h in range(ngen)], 1),
        ("ChargeRBM_PhysHidden", "rbm_charge_phys_hidden.def",
         [f"{r:5d} {h:5d} {(r + h * shift) % ns:5d}" for r in range(ns) for h in range(nc)], ns),
        ("SpinRBM_PhysHidden", "rbm_spin_phys_hidden.def",
         [f"{r:5d} {h:5d} {(r + h * shift) % ns:5d}" for r in range(ns) for h in range(nspin)],
         ns),
        ("GeneralRBM_PhysHidden", "rbm_general_phys_hidden.def",
         [f"{r:5d} {s:5d} {h:5d} {(r + s * ns + h) % (2 * ns):5d}"
          for s in (0, 1) for r in range(ns) for h in range(ngen)], 2 * ns),
    ]
    for keyword, filename, rows, width in sections:
        index_file(out, filename, keyword, rows, width)
        files.append((keyword, filename))

    write(out, "orbitalidx.def",
          header("NOrbitalIdx", length, ["ComplexType          0"])
          + [f"{i:5d} {j:6d} {(j - i) % length:6d}" for i in range(ns) for j in range(ns)]
          + [f"{k:5d} {1:6d}" for k in range(length)])
    write(out, "qptransidx.def",
          header("NQPTrans", 1) + ["0    1.00000"]
          + [f"{0:5d} {i:6d} {i:6d} {1:6d}" for i in range(ns)])
    files += [("Orbital", "orbitalidx.def"), ("TransSym", "qptransidx.def")]

    # Initial factors: exact binary fractions of moderate size.
    write(out, "ingutzwiller.def",
          header("NGutzwillerIdx", 1) + [f"{0:5d} {-0.375:25.15f} {0.0:25.15f}"])
    files.append(("InGutzwiller", "ingutzwiller.def"))
    # Couplings of magnitude <= 0.25 give hidden fields |theta| of order one,
    # with both signs, so cosh(theta) varies strongly between configurations.
    values = {
        "ChargeRBM_PhysLayer": [0.125],
        "SpinRBM_PhysLayer": [0.0625, -0.09375],
        "GeneralRBM_PhysLayer": [0.046875, -0.078125],
        "ChargeRBM_HiddenLayer": [0.1875, -0.15625],
        "SpinRBM_HiddenLayer": [((-1) ** h) * (h + 1) / 32 for h in range(nspin)],
        "GeneralRBM_HiddenLayer": [-0.109375],
        "ChargeRBM_PhysHidden": [((-1) ** (k // 3)) * (k % 7 + 1) / 32 for k in range(ns)],
        "SpinRBM_PhysHidden": [((-1) ** k) * (k % 5 + 2) / 32 for k in range(ns)],
        "GeneralRBM_PhysHidden": [((-1) ** (k // 2)) * (k % 6 + 1) / 64
                                  for k in range(2 * ns)],
    }
    for keyword, filename, _, width in sections:
        assert len(values[keyword]) == width
        overlay(out, "in_" + filename, keyword, values[keyword])
        files.append(("In" + keyword, "in_" + filename))
    write(out, "namelist.def", [f"{k:>24}  {v}" for k, v in files])


generate("hubbard_rbm_L16", 16, 12, 1, neurons=(4, 4, 2))
generate("hubbard_rbm_L128", 128, 128, 0, neurons=(4, 4, 2))
