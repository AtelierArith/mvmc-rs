"""2D-lattice fixture inputs (issue #355): square/tetragonal, triangular, honeycomb, kagome."""

LATTICES = {
    # key: (lattice keyword, W, L, hubbard ncond for that size)
    "square": ("square", 2, 2, 4),
    "triangular": ("triangular", 3, 3, 9),
    "honeycomb": ("honeycomb", 2, 2, 8),
    "kagome": ("kagome", 2, 2, 12),
}


def head(model, lat, w, l):
    return f'model = "{model}"\nlattice = "{lat}"\nW = {w}\nL = {l}\n'


CASES = {}
for key, (lat, w, l, nc) in LATTICES.items():
    H = head("Hubbard", lat, w, l) + f"U = 4.0\nt = 1.0\nncond = {nc}\n"
    S = head("Spin", lat, w, l) + "J = 1.0\n2Sz = 0\n"
    K = head("Kondo", lat, w, l) + f"t = 1.0\nJ = 2.0\nncond = {nc}\n"
    CASES[f"{key}_hubbard"] = H
    CASES[f"{key}_hubbard_sub"] = H + f"Wsub = 1\nLsub = {l}\n"
    CASES[f"{key}_hubbard_long_range"] = (
        H + "mu = -0.4\nt' = 0.3\nt'' = 0.1\nV = 0.5\nV' = 0.2\nV'' = 0.1\n"
    )
    CASES[f"{key}_hubbard_complex_hopping"] = (
        head("Hubbard", lat, w, l)
        + f"U = 2.0\nt = 1.0, 0.5\nt' = -0.3, 0.2\nncond = {nc}\nComplexType = 1\n"
    )
    CASES[f"{key}_hubbard_gc_fields"] = (
        head("HubbardGC", lat, w, l)
        + f"U = 3.0\nt = 1.0\nh = -0.7\nGamma = 0.4\nGamma_y = 0.25\nncond = {nc}\n"
    )
    CASES[f"{key}_hubbard_antiperiodic"] = H + "phase0 = 180\nphase1 = 180\n"
    CASES[f"{key}_hubbard_antiperiodic_w"] = H + "phase0 = 180\n"
    CASES[f"{key}_hubbard_box_tilted"] = (
        f'model = "Hubbard"\nlattice = "{lat}"\na0W = 2\na0L = 1\na1W = -1\na1L = 2\n'
        f"U = 4.0\nt = 1.0\nncond = 5\n"
    )
    CASES[f"{key}_hubbard_box_negative_det"] = (
        f'model = "Hubbard"\nlattice = "{lat}"\na0W = 1\na0L = 2\na1W = 2\na1L = 0\n'
        f"U = 4.0\nt = 1.0\nncond = 4\n"
    )
    CASES[f"{key}_hubbard_nmptrans_0"] = H + "NMPTrans = 0\nWsub = 1\nLsub = 1\n"
    CASES[f"{key}_spin"] = S + f"Wsub = 1\nLsub = {l}\n"
    CASES[f"{key}_spin_long_range"] = S + "J' = 0.4\nJ'' = 0.1\n"
    CASES[f"{key}_spin_anisotropic"] = (
        head("Spin", lat, w, l) + "Jx = 1.0\nJy = 0.5\nJz = 2.0\nD = 0.2\n2Sz = 0\n"
    )
    CASES[f"{key}_spin_bonds"] = (
        head("Spin", lat, w, l)
        + "J0 = 1.0\nJ1x = 0.5\nJ1y = 0.6\nJ1z = 0.7\nJ0' = 0.2\nJ1' = 0.3\n2Sz = 0\n"
        "ComplexType = 1\n"
    )
    CASES[f"{key}_spin_gc_fields"] = (
        head("SpinGC", lat, w, l) + "J = 1.0\nh = 0.3\nGamma = 0.2\nGamma_y = -0.4\n"
    )
    CASES[f"{key}_spin_antiperiodic"] = S + "phase0 = 180\nphase1 = 180\n"
    CASES[f"{key}_kondo"] = K
    CASES[f"{key}_kondo_gc_fields"] = (
        head("KondoGC", lat, w, l)
        + f"t = 1.0\nJ = 1.0\nncond = {nc}\nh = -0.4\nGamma = 0.3\nGamma_y = 0.2\nU = 2.0\n"
    )
    CASES[f"{key}_kondo_anisotropic_j"] = (
        head("Kondo", lat, w, l) + f"t = 1.0\nJx = 2.0\nJy = 1.5\nJz = 1.0\nncond = {nc}\n"
    )
    CASES[f"{key}_kondo_antiperiodic"] = K + "phase0 = 180\n"
    # --- errors ---
    CASES[f"err_{key}_L_box_conflict"] = H + "a0W = 2\n"
    CASES[f"err_{key}_zero_cell"] = (
        f'model = "Hubbard"\nlattice = "{lat}"\na0W = 1\na0L = 1\na1W = 2\na1L = 2\n'
        f"U = 1.0\nt = 1.0\nncond = 1\n"
    )
    CASES[f"err_{key}_K_not_used"] = H + "K = 1.0\n"
    CASES[f"err_{key}_missing_ncond"] = head("Hubbard", lat, w, l) + "U = 4.0\nt = 1.0\n"
    CASES[f"err_{key}_hubbard_J_not_used"] = H + "J = 1.0\n"
    CASES[f"err_{key}_hubbard_J0_not_used"] = H + "J0 = 1.0\n"
    CASES[f"err_{key}_hubbard_D_not_used"] = H + "D = 1.0\n"
    CASES[f"err_{key}_hubbard_2S_unknown"] = H + "2S = 1\n"
    CASES[f"err_{key}_hubbard_t_conflict"] = H + "t0 = 2.0\n"
    CASES[f"err_{key}_hubbard_V_conflict"] = H + "V = 1.0\nV0 = 2.0\n"
    CASES[f"err_{key}_spin_U_not_used"] = S + "U = 1.0\n"
    CASES[f"err_{key}_spin_mu_not_used"] = S + "mu = 1.0\n"
    CASES[f"err_{key}_spin_t_not_used"] = S + "t = 1.0\n"
    CASES[f"err_{key}_spin_t0_not_used"] = S + "t0 = 1.0\n"
    CASES[f"err_{key}_spin_V_not_used"] = S + "V = 1.0\n"
    CASES[f"err_{key}_spin_J_conflict"] = S + "J0 = 2.0\n"
    CASES[f"err_{key}_spin_J0_component_conflict"] = (
        head("Spin", lat, w, l) + "J0 = 1.0\nJ0x = 2.0\n2Sz = 0\n"
    )
    CASES[f"err_{key}_spin_missing_2sz"] = head("Spin", lat, w, l) + "J = 1.0\n"
    CASES[f"err_{key}_sublattice_incommensurate"] = H + f"Wsub = {w + 1}\nLsub = 1\n"
    CASES[f"err_{key}_sublattice_conflict"] = H + "Wsub = 1\na0Wsub = 1\n"

# Square-only keyword restrictions (t2, V2, J2, K are rejected by StdFace_Tetragonal).
CASES["err_square_t2_not_used"] = (
    head("Hubbard", "square", 2, 2) + "U = 4.0\nt = 1.0\nt2 = 1.0\nncond = 4\n"
)
CASES["err_square_V2_not_used"] = (
    head("Hubbard", "square", 2, 2) + "U = 4.0\nt = 1.0\nV2 = 1.0\nncond = 4\n"
)
CASES["err_square_J2_not_used"] = head("Spin", "square", 2, 2) + "J = 1.0\nJ2 = 1.0\n2Sz = 0\n"
CASES["err_square_t2p_not_used"] = (
    head("Hubbard", "square", 2, 2) + "U = 4.0\nt = 1.0\nt2' = 1.0\nncond = 4\n"
)

# Triangular/honeycomb/kagome use J2/t2 (third bond direction); honeycomb and kagome also
# accept bond-specific keywords.
CASES["triangular_hubbard_bond_specific"] = (
    head("Hubbard", "triangular", 3, 3)
    + "U = 4.0\nt0 = 1.0\nt1 = 0.8\nt2 = 0.6\nt0' = 0.1\nt1' = 0.2\nt2' = 0.3\n"
    "t0'' = 0.05\nt1'' = 0.06\nt2'' = 0.07\nV0 = 0.3\nV1' = 0.1\nV2'' = 0.2\nncond = 9\n"
)
CASES["honeycomb_hubbard_bond_specific"] = (
    head("Hubbard", "honeycomb", 2, 2)
    + "U = 4.0\nt0 = 1.0\nt1 = 0.8\nt2 = 0.6\nt0' = 0.1\nt1' = 0.2\nt2' = 0.3\n"
    "t0'' = 0.05\nt1'' = 0.06\nt2'' = 0.07\nV0 = 0.3\nV1' = 0.1\nV2'' = 0.2\nncond = 8\n"
)
CASES["kagome_hubbard_bond_specific"] = (
    head("Hubbard", "kagome", 2, 2)
    + "U = 4.0\nt0 = 1.0\nt1 = 0.8\nt2 = 0.6\nt0' = 0.1\nt1' = 0.2\nt2' = 0.3\n"
    "V0 = 0.3\nV1' = 0.1\nV2' = 0.2\nncond = 12\n"
)
# C quirks (see PR): copy-paste bugs in the NotUsed checks of the spin branch.
CASES["triangular_spin_tpp_unchecked"] = (
    head("Spin", "triangular", 3, 3) + "J = 1.0\n2Sz = 1\nt'' = 1.0\n"
)
CASES["honeycomb_spin_tpp_Vpp_unchecked"] = (
    head("Spin", "honeycomb", 2, 2) + "J = 1.0\n2Sz = 0\nt'' = 1.0\nV'' = 0.5\n"
)
CASES["err_kagome_spin_t0_not_used"] = head("Spin", "kagome", 2, 2) + "J = 1.0\n2Sz = 0\nt0 = 1.0\n"
CASES["kagome_hubbard_tpp_unknown_to_lattice"] = (
    head("Hubbard", "kagome", 2, 2) + "U = 4.0\nt = 1.0\nt'' = 0.5\nncond = 12\n"
)
CASES["square_hubbard_aliases"] = (
    'model = "FermionHubbard"\nlattice = "SquareLattice"\nW = 2\nL = 2\nU = 4.0\nt = 1.0\n'
    "ncond = 4\n"
)
CASES["triangular_alias"] = (
    'model = "Hubbard"\nlattice = "TriangularLattice"\nW = 2\nL = 2\nU = 4.0\nt = 1.0\n'
    "ncond = 4\n"
)
CASES["honeycomb_alias"] = (
    'model = "Hubbard"\nlattice = "HoneycombLattice"\nW = 2\nL = 2\nU = 4.0\nt = 1.0\n'
    "ncond = 8\n"
)
CASES["kagome_alias"] = (
    'model = "Hubbard"\nlattice = "KagomeLattice"\nW = 2\nL = 2\nU = 4.0\nt = 1.0\n'
    "ncond = 12\n"
)
CASES["square_hubbard_anisotropic_lengths"] = (
    head("Hubbard", "square", 2, 2)
    + "U = 4.0\nt = 1.0\nncond = 4\na = 2.0\nWlength = 1.5\nLlength = 0.5\n"
)
CASES["triangular_hubbard_directions"] = (
    head("Hubbard", "triangular", 2, 2)
    + "U = 4.0\nt = 1.0\nncond = 4\nWx = 1.0\nWy = 0.1\nLx = 0.4\nLy = 0.9\n"
)
CASES["tetragonal_hubbard_3x2"] = (
    head("Hubbard", "tetragonal", 3, 2) + "U = 4.0\nt = 1.0\nncond = 6\nWsub = 1\nLsub = 2\n"
)
CASES["tetragonal_spin_3x2"] = head("Spin", "tetragonal", 3, 2) + "J = 1.0\n2Sz = 0\n"
CASES["err_unknown_lattice_2d"] = head("Hubbard", "rectangle", 2, 2)
