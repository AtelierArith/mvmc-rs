"""Ladder-lattice fixture inputs (issue #354); merged into the case list by generate_fixtures.py.

C defect recorded here: `StdFace_Ladder` calls `StdFace_NotUsed_{c,d,J}` on `t`, `t'`, `V`, `V'`,
`J`, `J'` before it reads them (`Ladder.c`, "(2) check & store parameters"), so the isotropic
names are rejected and hopping must be given as t0, t1, t2, t1', t2' (likewise V*, J*). In
particular the Kondo coupling `J` can never be set (it is always 0). The Rust port reproduces it.
"""

H = (
    'model = "Hubbard"\nlattice = "ladder"\nW = 2\nL = 3\nU = 4.0\nt0 = 1.0\nt1 = 1.0\n'
    "ncond = 6\n"
)
S = 'model = "Spin"\nlattice = "ladder"\nW = 2\nL = 3\n2Sz = 0\n'
K = 'model = "Kondo"\nlattice = "ladder"\nW = 2\nL = 2\nt0 = 1.0\nt1 = 1.0\nncond = 4\n'

CASES = {
    "ladder_hubbard": H + "Lsub = 3\n",
    "ladder_hubbard_w3_anisotropic": (
        'model = "Hubbard"\nlattice = "ladder"\nW = 3\nL = 2\nU = 3.0\nt0 = 0.5\nt1 = 1.0\n'
        "t2 = 0.2\nt1' = 0.1\nt2' = 0.3\nV0 = 0.4\nV1 = 0.6\nV2 = 0.1\nV1' = 0.05\n"
        "V2' = 0.07\nmu = -0.3\nncond = 6\n"
    ),
    "ladder_hubbard_complex_hopping": (
        'model = "Hubbard"\nlattice = "ladder"\nW = 2\nL = 3\nU = 2.0\nt0 = 1.0\n'
        "t1 = 1.0, 0.4\nV1 = 0.3\nncond = 6\nComplexType = 1\n"
    ),
    "ladder_hubbard_gc_fields_gamma_y": (
        'model = "HubbardGC"\nlattice = "ladder"\nW = 2\nL = 2\nU = 3.0\nt0 = 1.0\nt1 = 1.0\n'
        "h = -0.5\nGamma = 0.3\nGamma_y = 0.2\nncond = 4\n"
    ),
    "ladder_hubbard_antiperiodic": H + "phase0 = 180\n",
    "ladder_hubbard_nmptrans_0": H + "NMPTrans = 0\nLsub = 3\n",
    "ladder_hubbard_w1": (
        'model = "Hubbard"\nlattice = "ladder"\nW = 1\nL = 4\nU = 2.0\nt1 = 1.0\nncond = 4\n'
    ),
    "ladder_hubbard_wx_overridden": H + "Wx = 5.0\n",
    "ladder_spin": S + "J0 = 1.0\nJ1 = 0.8\nJ2 = 0.3\nJ1' = 0.2\nJ2' = 0.1\n",
    "ladder_spin_components": (
        S + "J0x = 1.0\nJ0y = 1.0\nJ0z = 0.5\nJ1 = 1.0\nJ2xy = 0.2\nJ2' = 0.1\nD = 0.3\n"
        "ComplexType = 1\n"
    ),
    "ladder_spin_gc_fields": (
        'model = "SpinGC"\nlattice = "ladder"\nW = 2\nL = 2\nJ0 = 1.0\nJ1 = 1.0\nh = 0.3\n'
        "Gamma = 0.2\nGamma_y = -0.4\n"
    ),
    "ladder_spin_antiperiodic": S + "J0 = 1.0\nJ1 = 1.0\nphase0 = 180\n",
    "err_ladder_spin_J1_conflict": S + "J1 = 1.0\nJ1x = 1.0\n",
    "ladder_kondo": K,
    "ladder_kondo_gc_fields": (
        'model = "KondoGC"\nlattice = "ladder"\nW = 2\nL = 2\nt0 = 1.0\nt1 = 1.0\nncond = 4\n'
        "h = -0.4\nGamma = 0.3\nGamma_y = 0.2\n"
    ),
    "err_ladder_kondo_J_not_used": K + "J = 2.0\n",
    "err_ladder_kondo_Jx_not_used": K + "Jx = 2.0\n",
    "err_ladder_hubbard_t_not_used": H + "t = 1.0\n",
    "err_ladder_hubbard_tp_not_used": H + "t' = 1.0\n",
    "err_ladder_hubbard_V_not_used": H + "V = 1.0\n",
    "err_ladder_missing_W": (
        'model = "Hubbard"\nlattice = "ladder"\nL = 3\nU = 4.0\nt1 = 1.0\nncond = 6\n'
    ),
    "err_ladder_missing_L": (
        'model = "Hubbard"\nlattice = "ladder"\nW = 2\nU = 4.0\nt1 = 1.0\nncond = 6\n'
    ),
    "err_ladder_a0W_not_used": H + "a0W = 2\n",
    "err_ladder_phase1_not_used": H + "phase1 = 90\n",
    "err_ladder_K_not_used": H + "K = 1.0\n",
    "err_ladder_hubbard_J_not_used": H + "J = 1.0\n",
    "err_ladder_hubbard_J1p_not_used": H + "J1'x = 1.0\n",
    "err_ladder_hubbard_J0_not_used": H + "J0 = 1.0\n",
    "err_ladder_spin_t_not_used": S + "t = 1.0\n",
    "err_ladder_spin_t1_not_used": S + "t1 = 1.0\n",
    "err_ladder_spin_U_not_used": S + "U = 1.0\n",
    "err_ladder_spin_J_not_used": S + "J = 1.0\n",
    "err_ladder_hubbard_V_with_V1_not_used": H + "V1 = 1.0\nV = 2.0\n",
    "err_ladder_missing_ncond": (
        'model = "Hubbard"\nlattice = "ladder"\nW = 2\nL = 3\nU = 4.0\nt1 = 1.0\n'
    ),
}
