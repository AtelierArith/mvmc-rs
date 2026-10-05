"""3D-lattice fixture inputs (issue #356): orthorhombic, face-centered orthorhombic, pyrochlore.

Merged into the case list by generate_fixtures.py. Each case is run through the unmodified C
StdFace (historical behaviour) and through the C StdFace with `3d_defects.patch` (corrected
behaviour, see `tests/fixtures/stdface/README_3d_defects.md`). Where the two differ the case is
listed in DEFECT_CASES: `expected/` holds the corrected output (what Rust must produce) and
`c_historical/` the unmodified C output. Everywhere else both builds agree and `expected/` is
the C output, byte for byte.

C defects recorded here (not reproduced by the Rust port):
- `Pyrochlore.c` Kondo coupling is applied as `GeneralJ(.., isite + 3, jsite + isiteUC)`:
  every localized spin of a cell couples to conduction site 3 and conduction sites 0-2 have no
  Kondo coupling.
- `FCOrtho.c` Kondo: the localized spin does not feel `h`, `Gamma`, `Gamma_y` (the other
  lattices apply them).
- `ntransMax` omits the `Gamma_y` terms of `StdFace_HubbardLocal` (up to 6 transfer terms per
  site, 4 counted): with `Gamma` and `Gamma_y` nonzero and no unused hopping slack (pyrochlore)
  `StdFace_trans` writes past the allocation (heap overflow, exit status -11 = SIGSEGV).
- Unused parameters accepted silently: V2, V0', V1', V2', V'' in the spin model of all three
  lattices; J', J'' in the Hubbard/Kondo model of Orthorhombic and Pyrochlore (Orthorhombic
  rejects the nonexistent J0'', J1'', J2'' instead); t'', V'' and J0''..J2'' in FCOrtho
  Hubbard/Kondo and J'' in FCOrtho spin (read and dropped); t', t'', V', V'' in Pyrochlore
  Hubbard/Kondo; J', J'' in Pyrochlore spin.
"""

BOX3 = "L = 2\nW = 2\nHeight = 2\n"


def H(lat, extra="", size=BOX3, ncond=8):
    return f'model = "Hubbard"\nlattice = "{lat}"\n{size}U = 4.0\nt = 1.0\nncond = {ncond}\n{extra}'


def Hn(lat, extra="", size=BOX3, ncond=8):
    """Hubbard model without the isotropic `t` (so that t0, t1, ... may be given)."""
    return f'model = "Hubbard"\nlattice = "{lat}"\n{size}U = 4.0\nncond = {ncond}\n{extra}'


def S(lat, extra="", size=BOX3):
    return f'model = "Spin"\nlattice = "{lat}"\n{size}2Sz = 0\n{extra}'


def K(lat, extra="", size=BOX3, ncond=8):
    return f'model = "Kondo"\nlattice = "{lat}"\n{size}t = 1.0\nJ = 2.0\nncond = {ncond}\n{extra}'


PYRO = "L = 2\nW = 2\nHeight = 1\n"  # 4 sites per cell: 16 sites
CASES = {}

# ----------------------------------------------------------------------------- orthorhombic
CASES.update({
    "ortho_hubbard": H("orthorhombic"),
    "ortho_hubbard_anisotropic": Hn(
        "orthorhombic",
        "t0 = 1.0\nt1 = 0.7\nt2 = 0.4\nt0' = 0.2\nt1' = 0.1\nt2' = 0.05\nt'' = 0.03\n"
        "V0 = 0.5\nV1 = 0.4\nV2 = 0.3\nV0' = 0.2\nV1' = 0.1\nV2' = 0.05\nV'' = 0.02\n"
        "mu = -0.3\nWlength = 1.5\nLlength = 2.0\nHlength = 0.8\n",
    ),
    "ortho_hubbard_isotropic_defaults": H("orthorhombic", "t' = 0.3\nV = 0.5\nV' = 0.2\n"),
    "ortho_hubbard_antiperiodic": H("orthorhombic", "phase0 = 180\nphase1 = 180\nphase2 = 180\n"),
    "ortho_hubbard_complex_hopping": H(
        "orthorhombic", "ComplexType = 1\n", ncond=8
    ).replace("t = 1.0", "t = 1.0, 0.4"),
    "ortho_hubbard_gc_fields_gamma_y": (
        'model = "HubbardGC"\nlattice = "orthorhombic"\n' + BOX3 + "U = 3.0\nt = 1.0\nh = -0.7\n"
        "Gamma = 0.4\nGamma_y = 0.25\nncond = 8\n"
    ),
    "ortho_hubbard_box_skewed": (
        'model = "Hubbard"\nlattice = "orthorhombic"\na0W = 2\na0L = 0\na0H = 0\na1W = 1\na1L = 2\n'
        "a1H = 0\na2W = 0\na2L = 0\na2H = 2\nU = 4.0\nt = 1.0\nncond = 8\n"
    ),
    "ortho_hubbard_open_dimensions": H("orthorhombic", size="L = 3\nW = 2\nHeight = 1\n", ncond=6),
    "ortho_alias_cubic": H("cubic"),
    "ortho_alias_simple_orthorhombic": H("simpleorthorhombic"),
    "ortho_alias_simple_cubic": H("SimpleCubic"),
    "ortho_spin": S(
        "orthorhombic",
        "J0 = 1.0\nJ1 = 0.8\nJ2 = 0.6\nJ0' = 0.4\nJ1' = 0.3\nJ2' = 0.2\nJ'' = 0.1\n",
    ),
    "ortho_spin_isotropic": S("orthorhombic", "J = 1.0\nJ' = 0.5\nJ'' = 0.2\n"),
    "ortho_spin_components": S(
        "orthorhombic",
        "J0x = 1.0\nJ0y = 1.0\nJ0z = 0.5\nJ1 = 1.0\nJ2xy = 0.2\nJ0'x = 0.1\nJ'' = 0.1\nD = 0.3\n"
        "ComplexType = 1\n",
    ),
    "err_ortho_hphi_2s_keyword": S("orthorhombic", "2S = 2\nJ = 1.0\n"),
    "ortho_spin_gc_fields": (
        'model = "SpinGC"\nlattice = "orthorhombic"\n' + BOX3 + "J = 1.0\nh = 0.3\nGamma = 0.2\n"
        "Gamma_y = -0.4\n"
    ),
    "ortho_spin_antiperiodic": S("orthorhombic", "J = 1.0\nphase0 = 180\n"),
    "ortho_kondo": K("orthorhombic"),
    "ortho_kondo_fields": K("orthorhombic", "h = 0.4\nGamma = 0.2\nU = 1.0\nmu = 0.1\n"),
    "ortho_kondo_gc_fields": (
        'model = "KondoGC"\nlattice = "orthorhombic"\n' + BOX3 + "t = 1.0\nJ = 1.0\nncond = 8\n"
        "h = -0.4\nGamma = 0.3\nGamma_y = 0.2\n"
    ),
    "ortho_kondo_third_neighbour": K("orthorhombic", "t'' = 0.2\nV'' = 0.1\n"),
    # errors (identical in C and in the corrected build)
    "err_ortho_hubbard_K_not_used": H("orthorhombic", "K = 1.0\n"),
    "err_ortho_hubbard_J_not_used": H("orthorhombic", "J = 1.0\n"),
    "err_ortho_hubbard_J0_not_used": H("orthorhombic", "J0 = 1.0\n"),
    "err_ortho_hubbard_D_not_used": H("orthorhombic", "D = 1.0\n"),
    "err_ortho_hubbard_2S_not_used": H("orthorhombic", "2S = 2\n"),
    "err_ortho_hubbard_t_conflict": H("orthorhombic", "t0 = 2.0\n"),
    "err_ortho_hubbard_V_conflict": H("orthorhombic", "V = 1.0\nV0 = 2.0\n"),
    "err_ortho_spin_U_not_used": S("orthorhombic", "U = 1.0\n"),
    "err_ortho_spin_t_not_used": S("orthorhombic", "t = 1.0\n"),
    "err_ortho_spin_t0_not_used": S("orthorhombic", "t0 = 1.0\n"),
    "err_ortho_spin_V_not_used": S("orthorhombic", "V = 1.0\n"),
    "err_ortho_spin_V0_not_used": S("orthorhombic", "V0 = 1.0\n"),
    "err_ortho_spin_J_conflict": S("orthorhombic", "J = 1.0\nJ0 = 2.0\n"),
    "err_ortho_spin_J0_component_conflict": S("orthorhombic", "J0 = 1.0\nJ0x = 2.0\n"),
    "err_ortho_missing_2sz": 'model = "Spin"\nlattice = "orthorhombic"\n' + BOX3 + "J = 1.0\n",
    "err_ortho_missing_ncond": 'model = "Hubbard"\nlattice = "orthorhombic"\n' + BOX3 + "U = 1.0\nt = 1.0\n",
    "err_3d_lwh_box_conflict": H("orthorhombic", "a0W = 2\n"),
    # C accepts these silently, the corrected build rejects them (DEFECT_CASES)
    "ortho_spin_V2_accepted_by_c": S("orthorhombic", "J = 1.0\nV2 = 1.0\n"),
    "ortho_spin_Vp_component_accepted_by_c": S("orthorhombic", "J = 1.0\nV1' = 1.0\n"),
    "ortho_spin_Vpp_accepted_by_c": S("orthorhombic", "J = 1.0\nV'' = 1.0\n"),
    "ortho_hubbard_Jp_accepted_by_c": H("orthorhombic", "J' = 1.0\n"),
    "ortho_hubbard_Jpp_accepted_by_c": H("orthorhombic", "J'' = 1.0\n"),
    "ortho_kondo_Jp_accepted_by_c": K("orthorhombic", "J' = 1.0\n"),
})

# ----------------------------------------------------------------------------- fc orthorhombic
CASES.update({
    "fcc_hubbard": H("fcc"),
    "fcc_hubbard_anisotropic": Hn(
        "fcorthorhombic",
        "t0 = 1.0\nt1 = 0.7\nt2 = 0.4\nt0' = 0.2\nt1' = 0.1\nt2' = 0.05\n"
        "V0 = 0.5\nV1 = 0.4\nV2 = 0.3\nV0' = 0.2\nV1' = 0.1\nV2' = 0.05\n"
        "mu = -0.3\nWlength = 1.5\nLlength = 2.0\nHlength = 0.8\n",
    ),
    "fcc_hubbard_antiperiodic": H("fcc", "phase0 = 180\nphase1 = 180\nphase2 = 180\n"),
    "fcc_hubbard_complex_hopping": H("fcc", "ComplexType = 1\n").replace("t = 1.0", "t = 1.0, 0.4"),
    "fcc_hubbard_gc_fields_gamma_y": (
        'model = "HubbardGC"\nlattice = "fcc"\n' + BOX3 + "U = 3.0\nt = 1.0\nh = -0.7\n"
        "Gamma = 0.4\nGamma_y = 0.25\nncond = 8\n"
    ),
    "fcc_hubbard_box_skewed": (
        'model = "Hubbard"\nlattice = "fco"\na0W = 2\na0L = 0\na0H = 0\na1W = 1\na1L = 2\n'
        "a1H = 0\na2W = 0\na2L = 0\na2H = 2\nU = 4.0\nt = 1.0\nncond = 8\n"
    ),
    "fcc_alias_face_centered_cubic": H("face-centeredcubic"),
    "fcc_alias_fccubic": H("fccubic"),
    "fcc_alias_face_centered_orthorhombic": H("face-centeredorthorhombic"),
    "fcc_spin": S(
        "fcc", "J0 = 1.0\nJ1 = 0.8\nJ2 = 0.6\nJ0' = 0.4\nJ1' = 0.3\nJ2' = 0.2\n"
    ),
    "fcc_spin_isotropic": S("fcc", "J = 1.0\nJ' = 0.5\n"),
    "fcc_spin_components": S(
        "fcc", "J0x = 1.0\nJ0y = 1.0\nJ0z = 0.5\nJ1 = 1.0\nJ2xy = 0.2\nJ0'x = 0.1\nD = 0.3\nComplexType = 1\n"
    ),
    "fcc_spin_gc_fields": (
        'model = "SpinGC"\nlattice = "fcc"\n' + BOX3 + "J = 1.0\nh = 0.3\nGamma = 0.2\n"
        "Gamma_y = -0.4\n"
    ),
    "fcc_kondo": K("fcc"),
    "fcc_kondo_gc": (
        'model = "KondoGC"\nlattice = "fcc"\n' + BOX3 + "t = 1.0\nJ = 1.0\nncond = 8\n"
    ),
    "err_fcc_hubbard_K_not_used": H("fcc", "K = 1.0\n"),
    "err_fcc_hubbard_J_not_used": H("fcc", "J = 1.0\n"),
    "err_fcc_spin_t_not_used": S("fcc", "t = 1.0\n"),
    "err_fcc_spin_V_not_used": S("fcc", "V = 1.0\n"),
    "err_fcc_spin_J_conflict": S("fcc", "J = 1.0\nJ0 = 2.0\n"),
    # C accepts / mishandles these, the corrected build does not (DEFECT_CASES)
    "fcc_kondo_fields_defect": K("fcc", "h = 0.4\nGamma = 0.2\nGamma_y = 0.1\n"),
    "fcc_kondo_gc_fields_defect": (
        'model = "KondoGC"\nlattice = "fcc"\n' + BOX3 + "t = 1.0\nJ = 1.0\nncond = 8\n"
        "h = -0.4\nGamma = 0.3\nGamma_y = 0.2\n"
    ),
    "fcc_hubbard_tpp_accepted_by_c": H("fcc", "t'' = 0.2\n"),
    "fcc_hubbard_Vpp_accepted_by_c": H("fcc", "V'' = 0.2\n"),
    "fcc_hubbard_J0pp_accepted_by_c": H("fcc", "J0'' = 0.2\n"),
    "fcc_spin_Jpp_accepted_by_c": S("fcc", "J = 1.0\nJ'' = 0.3\n"),
    "fcc_spin_J0pp_accepted_by_c": S("fcc", "J = 1.0\nJ0'' = 0.3\n"),
    "fcc_spin_V2_accepted_by_c": S("fcc", "J = 1.0\nV2 = 1.0\n"),
    "fcc_spin_Vpp_accepted_by_c": S("fcc", "J = 1.0\nV'' = 1.0\n"),
})

# ----------------------------------------------------------------------------- pyrochlore
CASES.update({
    "pyrochlore_hubbard": H("pyrochlore", size=PYRO, ncond=16),
    "pyrochlore_hubbard_anisotropic": Hn(
        "pyrochlore",
        "t0 = 1.0\nt1 = 0.7\nt2 = 0.4\nt0' = 0.2\nt1' = 0.1\nt2' = 0.05\n"
        "V0 = 0.5\nV1 = 0.4\nV2 = 0.3\nV0' = 0.2\nV1' = 0.1\nV2' = 0.05\n"
        "mu = -0.3\n",
        size=PYRO,
        ncond=16,
    ),
    "pyrochlore_hubbard_antiperiodic": H(
        "pyrochlore", "phase0 = 180\nphase1 = 180\nphase2 = 180\n", size=PYRO, ncond=16
    ),
    "pyrochlore_hubbard_complex_hopping": H(
        "pyrochlore", "ComplexType = 1\n", size=PYRO, ncond=16
    ).replace("t = 1.0", "t = 1.0, 0.4"),
    "pyrochlore_hubbard_gc_fields_gamma_y": (
        'model = "HubbardGC"\nlattice = "pyrochlore"\n' + PYRO + "U = 3.0\nt = 1.0\nh = -0.7\n"
        "Gamma = 0.4\nGamma_y = 0.25\nncond = 16\n"
    ),
    "pyrochlore_spin": S(
        "pyrochlore", "J0 = 1.0\nJ1 = 0.8\nJ2 = 0.6\nJ0' = 0.4\nJ1' = 0.3\nJ2' = 0.2\n", size=PYRO
    ),
    "pyrochlore_spin_isotropic": S("pyrochlore", "J = 1.0\n", size=PYRO),
    "pyrochlore_spin_components": S(
        "pyrochlore",
        "J0x = 1.0\nJ0y = 1.0\nJ0z = 0.5\nJ1 = 1.0\nJ2xy = 0.2\nJ0'x = 0.1\nD = 0.3\n"
        "ComplexType = 1\n",
        size=PYRO,
    ),
    "pyrochlore_spin_gc_fields": (
        'model = "SpinGC"\nlattice = "pyrochlore"\n' + PYRO + "J = 1.0\nh = 0.3\nGamma = 0.2\n"
        "Gamma_y = -0.4\n"
    ),
    "pyrochlore_kondo_gc_fields_defect": (
        'model = "KondoGC"\nlattice = "pyrochlore"\n' + PYRO + "t = 1.0\nJ = 1.0\nncond = 16\n"
        "h = -0.4\nGamma = 0.3\nGamma_y = 0.2\n"
    ),
    "err_pyrochlore_hubbard_K_not_used": H("pyrochlore", "K = 1.0\n", size=PYRO, ncond=16),
    "err_pyrochlore_hubbard_J_not_used": H("pyrochlore", "J = 1.0\n", size=PYRO, ncond=16),
    "err_pyrochlore_spin_t_not_used": S("pyrochlore", "t = 1.0\n", size=PYRO),
    "err_pyrochlore_spin_V_not_used": S("pyrochlore", "V = 1.0\n", size=PYRO),
    "err_pyrochlore_spin_J_conflict": S("pyrochlore", "J = 1.0\nJ0 = 2.0\n", size=PYRO),
    # C accepts / mishandles these, the corrected build does not (DEFECT_CASES)
    "pyrochlore_kondo_defect": K("pyrochlore", size=PYRO, ncond=16),
    "pyrochlore_kondo_fields_defect": K(
        "pyrochlore", "h = 0.4\nGamma = 0.2\n", size=PYRO, ncond=16
    ),
    "pyrochlore_hubbard_tp_accepted_by_c": H(
        "pyrochlore", "t' = 0.2\n", size=PYRO, ncond=16
    ),
    "pyrochlore_hubbard_tpp_accepted_by_c": H(
        "pyrochlore", "t'' = 0.2\n", size=PYRO, ncond=16
    ),
    "pyrochlore_hubbard_Vp_accepted_by_c": H(
        "pyrochlore", "V' = 0.2\n", size=PYRO, ncond=16
    ),
    "pyrochlore_hubbard_Vpp_accepted_by_c": H(
        "pyrochlore", "V'' = 0.2\n", size=PYRO, ncond=16
    ),
    "pyrochlore_hubbard_Jp_accepted_by_c": H(
        "pyrochlore", "J' = 0.2\n", size=PYRO, ncond=16
    ),
    "pyrochlore_spin_Jp_accepted_by_c": S("pyrochlore", "J = 1.0\nJ' = 0.3\n", size=PYRO),
    "pyrochlore_spin_Jpp_accepted_by_c": S("pyrochlore", "J = 1.0\nJ'' = 0.3\n", size=PYRO),
    "pyrochlore_spin_V2_accepted_by_c": S("pyrochlore", "J = 1.0\nV2 = 1.0\n", size=PYRO),
    "pyrochlore_spin_Vpp_accepted_by_c": S("pyrochlore", "J = 1.0\nV'' = 1.0\n", size=PYRO),
})

# Cases where the unmodified C and the corrected build differ. The generator verifies that this
# set equals the set of cases whose outputs actually differ.
DEFECT_CASES = {
    "ortho_spin_V2_accepted_by_c",
    "ortho_spin_Vp_component_accepted_by_c",
    "ortho_spin_Vpp_accepted_by_c",
    "ortho_hubbard_Jp_accepted_by_c",
    "ortho_hubbard_Jpp_accepted_by_c",
    "ortho_kondo_Jp_accepted_by_c",
    "fcc_kondo_fields_defect",
    "fcc_kondo_gc_fields_defect",
    "fcc_hubbard_tpp_accepted_by_c",
    "fcc_hubbard_Vpp_accepted_by_c",
    "fcc_hubbard_J0pp_accepted_by_c",
    "fcc_spin_Jpp_accepted_by_c",
    "fcc_spin_J0pp_accepted_by_c",
    "fcc_spin_V2_accepted_by_c",
    "fcc_spin_Vpp_accepted_by_c",
    "pyrochlore_kondo_defect",
    "pyrochlore_hubbard_gc_fields_gamma_y",
    "pyrochlore_kondo_gc_fields_defect",
    "pyrochlore_kondo_fields_defect",
    "pyrochlore_hubbard_tp_accepted_by_c",
    "pyrochlore_hubbard_tpp_accepted_by_c",
    "pyrochlore_hubbard_Vp_accepted_by_c",
    "pyrochlore_hubbard_Vpp_accepted_by_c",
    "pyrochlore_hubbard_Jp_accepted_by_c",
    "pyrochlore_spin_Jp_accepted_by_c",
    "pyrochlore_spin_Jpp_accepted_by_c",
    "pyrochlore_spin_V2_accepted_by_c",
    "pyrochlore_spin_Vpp_accepted_by_c",
}
