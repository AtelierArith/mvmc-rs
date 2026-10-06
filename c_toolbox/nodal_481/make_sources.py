#!/usr/bin/env python3
"""Write the hand-made nodal-configuration inputs for the Lanczos checkGF/calHCA2 branch (#481).

The wavefunction is a Neel-type pairing state on an 8-site periodic chain with 3 up and 3 down
electrons. The orbital parameter couples only (up on an even site, down on an odd site); every
other pair uses a parameter fixed at zero. Every configuration with non-zero weight therefore has
all up electrons on the even sublattice and all down electrons on the odd sublattice. Any
nearest-neighbour hop of one electron, and the exchange of an up and a down electron on a
nearest-neighbour bond, leaves that manifold: the Pfaffian of the hopped configuration is zero, so
`checkGF1`/`checkGF2` return a value far below 1e-12 and C takes `calHCA2`/`calHCACA2`. The
next-nearest-neighbour transfers stay inside the manifold and exercise the ordinary branch in the
same runs.

    python3 c_toolbox/nodal_481/make_sources.py tests/fixtures/native_c_physcal_181/_sources/nodal_neel

writes `<dir>/inputs/*.def` and `<dir>/zqp_opt.dat` (fixed parameters, header zeros, (re, im, err)
triples: Gutzwiller, Orbital).
"""
import os
import sys

N = 8
NE = 3
NN_HOP = 1.0
NNN_HOP = -0.35
ORBITAL_VALUES = [1.0, 0.35, -0.22, 0.12]  # (j - i) mod 8 = 1, 3, 5, 7 for i even, j odd
GUTZWILLER = 0.15


def write(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def main(out):
    inputs = os.path.join(out, "inputs")
    os.makedirs(inputs, exist_ok=True)
    write(os.path.join(inputs, "namelist.def"), """         ModPara  modpara.def
         LocSpin  locspn.def
           Trans  trans.def
    CoulombIntra  coulombintra.def
        OneBodyG  greenone.def
        TwoBodyG  greentwo.def
      Gutzwiller  gutzwilleridx.def
         Orbital  orbitalidx.def
        TransSym  qptransidx.def
      TwoBodyGEx  greentwoex.def
CoulombInter coulombinter.def
Hund hund.def
Exchange exchange.def
PairHop pairhop.def
""")
    write(os.path.join(inputs, "modpara.def"), f"""--------------------
Model_Parameters   0
--------------------
VMC_Cal_Parameters
--------------------
CDataFileHead  zvo
CParaFileHead  zqp
--------------------
NVMCCalMode    1
NLanczosMode 1
--------------------
NDataIdxStart  1
NDataQtySmp    1
--------------------
Nsite          {N}
Ncond          {2 * NE}
2Sz            0
NSPGaussLeg    1
NSPStot        0
NMPTrans       -1
NSROptItrStep  500
NSROptItrSmp   50
DSROptRedCut   0.0000000100
DSROptStaDel   0.0100000000
DSROptStepDt   0.0030000000
NVMCWarmUp     10
NVMCInterval   1
NVMCSample     40
NExUpdatePath  0
RndSeed        1
NSplitSize     1
NStore         1
NSRCG          0
""")
    write(os.path.join(inputs, "locspn.def"),
          "================================ \nNlocalSpin     0  \n"
          "================================ \n========i_1LocSpn_0IteElc ====== \n"
          "================================ \n"
          + "".join(f"{i:5d} {0:5d}\n" for i in range(N)))
    rows = []
    for i in range(N):
        for s in (0, 1):
            for j, value in (((i + 1) % N, NN_HOP), ((i - 1) % N, NN_HOP),
                             ((i + 2) % N, NNN_HOP), ((i - 2) % N, NNN_HOP)):
                rows.append(f"{i:5d} {s:5d} {j:5d} {s:5d} {value:25.15f} {0.0:25.15f}\n")
    write(os.path.join(inputs, "trans.def"),
          f"======================== \nNTransfer      {len(rows)}  \n======================== \n"
          "========i_j_s_tijs====== \n======================== \n" + "".join(rows))
    write(os.path.join(inputs, "coulombintra.def"),
          "=============================================\n"
          f"NCoulombIntra          {N}\n=============================================\n"
          "================== CoulombIntra ================\n"
          "=============================================\n"
          + "".join(f"{i:5d} {4.0:25.15f}\n" for i in range(N)))
    write(os.path.join(inputs, "coulombinter.def"),
          "======================\nNCoulombInter 2\n======================\nCoulombInter\n"
          "======================\n0 1 0.375\n2 3 0.2\n")
    write(os.path.join(inputs, "hund.def"),
          "======================\nNHund 2\n======================\nHund\n"
          "======================\n0 1 -0.125\n1 2 -0.05\n")
    write(os.path.join(inputs, "exchange.def"),
          "======================\nNExchange 3\n======================\nExchange\n"
          "======================\n0 1 0.25\n2 3 0.25\n1 2 0.15\n")
    write(os.path.join(inputs, "pairhop.def"),
          "======================\nNPairHop 2\n======================\nPairHop\n"
          "======================\n0 1 -0.0625\n1 2 -0.04\n")
    write(os.path.join(inputs, "gutzwilleridx.def"),
          "=============================================\nNGutzwillerIdx          1\n"
          "ComplexType          0\n=============================================\n"
          "=============================================\n"
          + "".join(f"{i:5d} {0:5d}\n" for i in range(N)) + "    0      1\n")
    orbital = []
    for i in range(N):
        for j in range(N):
            if i % 2 == 0 and j % 2 == 1:
                index = ((j - i) % N - 1) // 2
            else:
                index = 4  # fixed at zero
            orbital.append(f"{i:5d} {j:5d} {index:5d}\n")
    write(os.path.join(inputs, "orbitalidx.def"),
          "=============================================\nNOrbitalIdx         5\n"
          "ComplexType          0\n=============================================\n"
          "=============================================\n" + "".join(orbital)
          + "".join(f"{k:5d} {1:5d}\n" for k in range(5)))
    write(os.path.join(inputs, "qptransidx.def"),
          "=============================================\nNQPTrans          1\n"
          "=============================================\n"
          "======== TrIdx_TrWeight_and_TrIdx_i_xi ======\n"
          "=============================================\n0    1.00000\n"
          + "".join(f"    0 {i:6d} {i:6d} {1:6d}\n" for i in range(N)))
    one = [f"{i:5d} {s:5d} {j:5d} {s:5d}\n" for s in (0, 1) for i in range(N) for j in range(N)]
    write(os.path.join(inputs, "greenone.def"),
          "===============================\n"
          f"NCisAjs         {len(one)}\n===============================\n"
          "======== Green functions ======\n===============================\n" + "".join(one))
    two = []
    for (si, sk) in ((0, 1), (1, 0), (0, 0)):
        for ri in (1, 3):
            for rj in (0, 2):
                for rk in (4, 6, 5):
                    for rl in (5, 7, 4):
                        if len(two) % 1 == 0 and len(two) < 60:
                            two.append(f"{ri:5d} {si:5d} {rj:5d} {si:5d} {rk:5d} {sk:5d} {rl:5d} {sk:5d}\n")
    # a few density-density and ordinary entries
    for entry in ((0, 0, 0, 0, 0, 1, 0, 1), (0, 0, 2, 0, 4, 1, 6, 1), (2, 0, 0, 0, 0, 0, 2, 0)):
        two.append("".join(f"{v:5d} " for v in entry) + "\n")
    write(os.path.join(inputs, "greentwo.def"),
          "=============================================\n"
          f"NCisAjsCktAltDC         {len(two)}\n=============================================\n"
          "======== Green functions for Sq AND Nq ======\n"
          "=============================================\n" + "".join(two))
    write(os.path.join(inputs, "greentwoex.def"),
          "=============================================\nNTwoBodyGEx          3\n"
          "=============================================\n"
          "======== Factored two-body Green (TwoBodyGEx) =\n"
          "=============================================\n"
          "    0     0     1     0     1     1     0     1\n"
          "    1     0     2     0     2     1     1     1\n"
          "    0     0     2     0     2     0     0     0\n")
    params = [GUTZWILLER] + ORBITAL_VALUES + [0.0]
    tokens = ["0.0"] * 6
    for value in params:
        tokens += [f"{value: .18e}", f"{0.0: .18e}", f"{1e-3: .18e}"]
    write(os.path.join(out, "zqp_opt.dat"), "  ".join(tokens) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
