"""Inputs that exercise the corrections of issue #404 (C defects reported as Julia-mVMC#66).

For these the C program and the corrected build differ; the fixture directory keeps both
(`expected/` = C, `expected_fixed/` = corrected build, see tests/fixtures/stdface/README.md).
"""

LH = 'model = "Hubbard"\nlattice = "ladder"\nW = 2\nL = 3\nU = 4.0\n'
LK = 'model = "Kondo"\nlattice = "ladder"\nW = 2\nL = 2\nncond = 4\n'
LS = 'model = "Spin"\nlattice = "ladder"\nW = 2\nL = 3\n2Sz = 0\n'


def head(model, lat, w=2, l=2):
    return f'model = "{model}"\nlattice = "{lat}"\nW = {w}\nL = {l}\n'


CASES = {
    # --- Ladder: isotropic t, V, J and the Kondo coupling ---------------------------------
    "ladder_hubbard_isotropic_t_V": LH + "t = 1.0\nV = 0.3\nncond = 6\n",
    "ladder_kondo_isotropic": LK + "t = 1.0\nV = 0.2\nJ = 2.0\n",
    "ladder_kondo_J_components": LK + "t = 1.0\nJx = 2.0\nJy = 1.5\nJz = 1.0\n",
    "ladder_kondo_J_with_bonds": LK + "t0 = 1.0\nt1 = 0.5\nJ = 1.0\nGamma_y = 0.2\n",
    "err_ladder_spin_t_V_J_not_used_t": LS + "t = 1.0\n",
    "err_ladder_spin_V_not_used": LS + "V = 1.0\n",
    "err_ladder_spin_J_not_used_fixed": LS + "J = 1.0\n",
    "err_ladder_kondo_tp_not_used": LK + "t = 1.0\nt' = 1.0\n",
    "err_ladder_kondo_Vp_not_used": LK + "t = 1.0\nV' = 1.0\n",
    "err_ladder_kondo_Jp_not_used": LK + "t = 1.0\nJ' = 1.0\n",
    # --- Ladder: Wx, Wy and the box keywords ---------------------------------------------
    "ladder_hubbard_a_scaled": (
        'model = "Hubbard"\nlattice = "ladder"\nW = 3\nL = 2\nU = 4.0\nt1 = 1.0\nt0 = 1.0\n'
        "a = 2.0\nncond = 6\n"
    ),
    "ladder_hubbard_wy": LH + "t1 = 1.0\nWy = 0.5\nncond = 6\n",
    "ladder_hubbard_wx_given": LH + "t1 = 1.0\nWx = 0.5\nncond = 6\n",
    "err_ladder_a0H_not_used": LH + "t1 = 1.0\nncond = 6\na0H = 1\n",
    "err_ladder_a1H_not_used": LH + "t1 = 1.0\nncond = 6\na1H = 1\n",
    "err_ladder_a2W_not_used": LH + "t1 = 1.0\nncond = 6\na2W = 1\n",
    "err_ladder_a2H_not_used": LH + "t1 = 1.0\nncond = 6\na2H = 1\n",
    "err_ladder_hubbard_J1p_not_used_label": LH + "t1 = 1.0\nncond = 6\nJ2'x = 1.0\n",
    # --- Chain / Triangular / Honeycomb: J'' labels ---------------------------------------
    "chain_spin_Jpp_labels": (
        'model = "Spin"\nlattice = "chain"\nL = 6\nJ = 1.0\nJ\'\' = 0.2\n2Sz = 0\n'
    ),
    "triangular_spin_Jpp_labels": head("Spin", "triangular", 3, 3) + "J = 1.0\nJ'' = 0.2\n2Sz = 1\n",
    "honeycomb_spin_Jpp_labels": head("Spin", "honeycomb") + "J = 1.0\nJ'' = 0.2\n2Sz = 0\n",
    "triangular_spin_Jpp_components": (
        head("Spin", "triangular", 3, 3) + "J = 1.0\nJ0''x = 0.1\nJ1'' = 0.2\n2Sz = 1\n"
    ),
    "err_triangular_spin_Jpp_conflict": (
        head("Spin", "triangular", 3, 3) + "J = 1.0\nJ'' = 0.2\nJ0'' = 0.1\n2Sz = 1\n"
    ),
    # --- Triangular: t'' check ----------------------------------------------------------------
    "err_triangular_spin_tpp_not_used": head("Spin", "triangular", 3, 3) + "J = 1.0\n2Sz = 1\nt'' = 1.0\n",
    "err_triangular_spin_t2pp_not_used": head("Spin", "triangular", 3, 3) + "J = 1.0\n2Sz = 1\nt2'' = 1.0\n",
    "err_triangular_hubbard_Jp_not_used": (
        head("Hubbard", "triangular", 3, 3) + "U = 4.0\nt = 1.0\nncond = 9\nJ' = 1.0\n"
    ),
    "err_triangular_hubbard_Jpp_not_used": (
        head("Hubbard", "triangular", 3, 3) + "U = 4.0\nt = 1.0\nncond = 9\nJ'' = 1.0\n"
    ),
    # --- Honeycomb: t'', V'', J'' --------------------------------------------------------------
    "err_honeycomb_spin_tpp_not_used": head("Spin", "honeycomb") + "J = 1.0\n2Sz = 0\nt'' = 1.0\n",
    "err_honeycomb_spin_t1pp_not_used": head("Spin", "honeycomb") + "J = 1.0\n2Sz = 0\nt1'' = 1.0\n",
    "err_honeycomb_spin_Vpp_not_used": head("Spin", "honeycomb") + "J = 1.0\n2Sz = 0\nV'' = 0.5\n",
    "err_honeycomb_spin_V0pp_not_used": head("Spin", "honeycomb") + "J = 1.0\n2Sz = 0\nV0'' = 0.5\n",
    "err_honeycomb_hubbard_J0p_not_used": (
        head("Hubbard", "honeycomb") + "U = 4.0\nt = 1.0\nncond = 8\nJ0' = 1.0\n"
    ),
    "err_honeycomb_hubbard_Jpp_not_used": (
        head("Hubbard", "honeycomb") + "U = 4.0\nt = 1.0\nncond = 8\nJ'' = 1.0\n"
    ),
    "err_honeycomb_hubbard_J2pp_not_used": (
        head("Hubbard", "honeycomb") + "U = 4.0\nt = 1.0\nncond = 8\nJ2''x = 1.0\n"
    ),
    # --- Kagome: duplicate check and t'', V'', J'' --------------------------------------------
    "err_kagome_hubbard_tpp_not_used": head("Hubbard", "kagome") + "U = 4.0\nt = 1.0\nt'' = 0.5\nncond = 12\n",
    "err_kagome_hubbard_Vpp_not_used": head("Hubbard", "kagome") + "U = 4.0\nt = 1.0\nV'' = 0.5\nncond = 12\n",
    "err_kagome_hubbard_Jpp_not_used": head("Hubbard", "kagome") + "U = 4.0\nt = 1.0\nJ'' = 0.5\nncond = 12\n",
    "err_kagome_spin_tpp_not_used": head("Spin", "kagome") + "J = 1.0\n2Sz = 0\nt'' = 1.0\n",
    "err_kagome_spin_Vpp_not_used": head("Spin", "kagome") + "J = 1.0\n2Sz = 0\nV'' = 1.0\n",
    "err_kagome_spin_Jpp_not_used": head("Spin", "kagome") + "J = 1.0\n2Sz = 0\nJ'' = 1.0\n",
    "err_kagome_kondo_Jpp_not_used": head("Kondo", "kagome") + "t = 1.0\nJ = 1.0\nncond = 12\nJ0'' = 1.0\n",
}
