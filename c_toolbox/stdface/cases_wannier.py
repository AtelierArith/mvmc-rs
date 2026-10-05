"""Wannier90-lattice fixture inputs (issue #357).

Each case is `(StdFace.def text, {data file name: text})`; the data files are the Wannier90
`zvo_geom.dat`, `zvo_hr.dat`, `zvo_ur.dat`, `zvo_jr.dat` and RESPACK `zvo_dr.dat` that C reads
from the working directory (the mVMC build cannot change the `zvo` prefix: `CDataFileHead` is an
HPhi-only keyword).
"""


def geom(direct, taus):
    lines = ["  %.10f  %.10f  %.10f" % tuple(v) for v in direct]
    lines.append("  %d" % len(taus))
    lines += ["  %.10f %.10f %.10f" % tuple(t) for t in taus]
    return "\n".join(lines) + "\n"


def hr(nwan, blocks, header="wannier90 format for StdFace tests"):
    """`blocks`: list of (R, {(i, j): (re, im)}) with 1-based band indices."""
    out = [header, "%10d" % nwan, "%10d" % len(blocks)]
    out.append("".join("%5d" % 1 for _ in blocks))
    for r, mat in blocks:
        for j in range(1, nwan + 1):
            for i in range(1, nwan + 1):
                re, im = mat.get((i, j), (0.0, 0.0))
                out.append("%5d%5d%5d%5d%5d%12.6f%12.6f" % (r[0], r[1], r[2], i, j, re, im))
    return "\n".join(out) + "\n"


NN = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
SQUARE_R = [(a, b, 0) for a in (-1, 0, 1) for b in (-1, 0, 1)]


def square_files(with_u=True, with_j=False, with_dr=False, nn_u=False):
    files = {
        "zvo_geom.dat": geom([(1, 0, 0), (0, 1, 0), (0, 0, 1)], [(0.0, 0.0, 0.0)]),
        "zvo_hr.dat": hr(1, [(r, {(1, 1): (-1.0, 0.0)} if r in NN else {}) for r in SQUARE_R]),
    }
    if with_u:
        files["zvo_ur.dat"] = hr(
            1,
            [
                (r, {(1, 1): (4.0, 0.0) if r == (0, 0, 0) else (1.0, 0.0) if (nn_u and r in NN) else (0.0, 0.0)})
                for r in SQUARE_R
            ],
        )
    if with_j:
        files["zvo_jr.dat"] = hr(1, [(r, {(1, 1): (0.2, 0.0)} if r in NN else {}) for r in SQUARE_R])
    if with_dr:
        dens = {(1, 0, 0): (-0.1, 0.02), (-1, 0, 0): (-0.1, -0.02), (0, 1, 0): (-0.1, 0.0), (0, -1, 0): (-0.1, 0.0)}
        files["zvo_dr.dat"] = hr(
            1,
            [(r, {(1, 1): (0.5, 0.0) if r == (0, 0, 0) else dens.get(r, (0.0, 0.0))}) for r in SQUARE_R],
        )
    return files


HONEY_R = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]


def honeycomb_files(with_j=True, with_dr=False, spin=False):
    t = (-1.0, 0.0)
    hop = {
        (0, 0, 0): {(1, 2): t, (2, 1): t, (1, 1): (0.1, 0.0), (2, 2): (-0.1, 0.0)},
        (-1, 0, 0): {(1, 2): t},
        (1, 0, 0): {(2, 1): t},
        (0, -1, 0): {(1, 2): (-0.5, 0.2)},
        (0, 1, 0): {(2, 1): (-0.5, -0.2)},
    }
    u0 = {(1, 1): (4.0, 0.0), (2, 2): (3.0, 0.0), (1, 2): (1.0, 0.0), (2, 1): (1.0, 0.0)}
    j0 = {(1, 2): (0.3, 0.0), (2, 1): (0.3, 0.0)}
    den = {
        (0, 0, 0): {(1, 1): (0.5, 0.0), (2, 2): (0.5, 0.0), (1, 2): (0.1, 0.05), (2, 1): (0.1, -0.05)},
        (-1, 0, 0): {(1, 2): (0.02, 0.0)},
        (1, 0, 0): {(2, 1): (0.02, 0.0)},
    }
    files = {
        "zvo_geom.dat": geom(
            [(1, 0, 0), (0.5, 0.8660254038, 0), (0, 0, 1)],
            [(0.0, 0.0, 0.0), (0.3333333333, 0.3333333333, 0.0)],
        ),
        "zvo_hr.dat": hr(2, [(r, hop.get(r, {})) for r in HONEY_R]),
        "zvo_ur.dat": hr(2, [(r, u0 if r == (0, 0, 0) else {}) for r in HONEY_R]),
    }
    if with_j:
        files["zvo_jr.dat"] = hr(2, [(r, j0 if r == (0, 0, 0) else {}) for r in HONEY_R])
    if with_dr:
        files["zvo_dr.dat"] = hr(2, [(r, den.get(r, {})) for r in HONEY_R])
    return files


def cases(std):
    sq = 'model = "Hubbard"\nlattice = "wannier90"\nW = 2\nL = 2\nncond = 4\n2Sz = 0\n'
    hc = 'model = "Hubbard"\nlattice = "wannier90"\nW = 2\nL = 2\nncond = 8\n2Sz = 0\n'
    sp = 'model = "Spin"\nlattice = "wannier90"\nW = 2\nL = 2\n2Sz = 0\n'
    c = {}
    # --- upstream sample ------------------------------------------------------------------
    sample = std / "samples/hubbard/wannier"
    c["sample_hubbard_wannier"] = (
        (sample / "stan.in").read_text(),
        {n: (sample / n).read_text() for n in ("zvo_geom.dat", "zvo_hr.dat", "zvo_ur.dat")},
    )
    c["wannier_sample_mvmc_keywords"] = (
        'model = "Hubbard"\nlattice = "wannier90"\nW = 2\nL = 2\n2Sz = 0\nncond = 4\n',
        {n: (sample / n).read_text() for n in ("zvo_geom.dat", "zvo_hr.dat", "zvo_ur.dat")},
    )
    # --- single orbital square lattice -----------------------------------------------------
    c["wannier_square_hubbard"] = (sq, square_files())
    c["wannier_square_hubbard_no_ur"] = (sq, square_files(with_u=False))
    c["wannier_square_hubbard_UJ"] = (
        sq + "cutoff_length_U = 1.5\ncutoff_length_J = 1.5\n",
        square_files(with_j=True, nn_u=True),
    )
    c["wannier_square_lambda"] = (
        sq + "cutoff_length_U = 1.5\ncutoff_length_J = 1.5\nlambda = 0.5\n",
        square_files(with_j=True, nn_u=True),
    )
    c["wannier_square_lambda_UJ"] = (
        sq + "cutoff_length_U = 1.5\ncutoff_length_J = 1.5\nlambda_U = 0.8\nlambda_J = 0.3\n",
        square_files(with_j=True, nn_u=True),
    )
    for mode in ("hartree", "hartree_u", "full", "none"):
        c[f"wannier_square_dc_{mode}"] = (
            sq + f'cutoff_length_U = 1.5\ncutoff_length_J = 1.5\ndoublecounting = "{mode}"\n',
            square_files(with_j=True, with_dr=True, nn_u=True),
        )
    c["wannier_square_dc_full_alpha"] = (
        sq + 'cutoff_length_U = 1.5\ncutoff_length_J = 1.5\ndoublecounting = full\nalpha = 0.25\n',
        square_files(with_j=True, with_dr=True, nn_u=True),
    )
    c["wannier_square_cutoff_t"] = (sq + "cutoff_t = 1.5\n", square_files())
    c["wannier_square_cutoff_length_t"] = (sq + "cutoff_length_t = 0.5\n", square_files())
    c["wannier_square_cutoff_vec"] = (
        sq + "cutoff_t_a0w = 1.0\ncutoff_t_a1l = 1.0\n",
        square_files(),
    )
    c["wannier_square_w3"] = (
        'model = "Hubbard"\nlattice = "wannier90"\nW = 3\nL = 2\nncond = 6\n2Sz = 0\n',
        square_files(),
    )
    c["wannier_square_box"] = (
        'model = "Hubbard"\nlattice = "wannier90"\na0W = 2\na0L = 1\na1W = -1\na1L = 2\nncond = 5\n2Sz = 1\n',
        square_files(),
    )
    c["wannier_square_height"] = (
        'model = "Hubbard"\nlattice = "wannier90"\nW = 2\nL = 2\nHeight = 2\nncond = 8\n2Sz = 0\n',
        square_files(),
    )
    c["wannier_square_antiperiodic"] = (sq + "phase0 = 180\nphase1 = 180\n", square_files())
    c["wannier_square_gc_fields"] = (
        'model = "HubbardGC"\nlattice = "wannier90"\nW = 2\nL = 2\nncond = 4\nh = -0.5\n'
        "Gamma = 0.3\nGamma_y = 0.2\nmu = 0.4\n",
        square_files(),
    )
    # GC model with every on-site term and a complete data set (ntransMax check, see #404):
    # HubbardLocal appends six on-site transfers per site while C reserves four.
    c["wannier_square_gc_all_terms"] = (
        'model = "HubbardGC"\nlattice = "wannier90"\nW = 2\nL = 2\nncond = 4\nh = -0.5\n'
        "Gamma = 0.3\nGamma_y = 0.2\nmu = 0.4\ncutoff_length_U = 1.5\ncutoff_length_J = 1.5\n"
        "doublecounting = full\n",
        square_files(with_j=True, with_dr=True, nn_u=True),
    )
    c["wannier_honeycomb_gc_all_terms"] = (
        'model = "HubbardGC"\nlattice = "wannier90"\nW = 2\nL = 2\nncond = 8\nh = -0.5\n'
        "Gamma = 0.3\nGamma_y = 0.2\nmu = 0.4\ncutoff_length_U = 1.0\ncutoff_length_J = 1.0\n"
        "doublecounting = full\n",
        honeycomb_files(with_dr=True),
    )
    c["wannier_square_sublattice"] = (sq + "Wsub = 1\nLsub = 2\n", square_files())
    c["wannier_square_nmptrans_0"] = (sq + "NMPTrans = 0\nWsub = 1\nLsub = 1\n", square_files())
    c["wannier_square_mu"] = (sq + "mu = -0.3\n", square_files())
    # --- two-orbital (honeycomb-like) ------------------------------------------------------
    c["wannier_honeycomb_hubbard"] = (hc + "cutoff_length_U = 1.0\ncutoff_length_J = 1.0\n", honeycomb_files())
    c["wannier_honeycomb_dc_full"] = (
        hc + "cutoff_length_U = 1.0\ncutoff_length_J = 1.0\ndoublecounting = full\n",
        honeycomb_files(with_dr=True),
    )
    c["wannier_honeycomb_dc_hartree"] = (
        hc + "cutoff_length_U = 1.0\ncutoff_length_J = 1.0\ndoublecounting = hartree\nalpha = 0.7\n",
        honeycomb_files(with_dr=True),
    )
    c["wannier_honeycomb_dc_hartree_u"] = (
        hc + "cutoff_length_U = 1.0\ncutoff_length_J = 1.0\ndoublecounting = hartree_u\n",
        honeycomb_files(with_dr=True),
    )
    c["wannier_honeycomb_no_jr"] = (hc + "cutoff_length_U = 1.0\n", honeycomb_files(with_j=False))
    c["wannier_honeycomb_spin"] = (sp + "cutoff_length_U = 1.0\n", honeycomb_files(with_j=False))
    c["wannier_honeycomb_spin_s1"] = (sp + "cutoff_length_U = 1.0\n", honeycomb_files(with_j=True))
    c["wannier_honeycomb_spin_gc"] = (
        'model = "SpinGC"\nlattice = "wannier90"\nW = 2\nL = 2\ncutoff_length_U = 1.0\nh = 0.3\nGamma = 0.2\n',
        honeycomb_files(with_j=False),
    )
    # --- errors ---------------------------------------------------------------------------
    c["err_wannier_kondo"] = (
        'model = "Kondo"\nlattice = "wannier90"\nW = 2\nL = 2\nncond = 4\n', square_files()
    )
    c["err_wannier_missing_geom"] = (sq, {n: v for n, v in square_files().items() if n != "zvo_geom.dat"})
    c["err_wannier_lambda_negative"] = (sq + "lambda_U = -1.0\n", square_files())
    c["err_wannier_lambda_J_negative"] = (sq + "lambda = 0.5\nlambda_J = -0.1\n", square_files())
    c["err_wannier_bad_doublecounting"] = (sq + "doublecounting = bogus\n", square_files())
    c["err_wannier_alpha_range"] = (sq + "alpha = 1.5\n", square_files())
    c["err_wannier_missing_dr"] = (
        sq + "doublecounting = full\n",
        {n: v for n, v in square_files(with_dr=True).items() if n != "zvo_dr.dat"},
    )
    c["err_wannier_U_not_used"] = (sq + "U = 4.0\n", square_files())
    c["err_wannier_K_not_used"] = (sq + "K = 1.0\n", square_files())
    c["err_wannier_missing_ncond"] = (
        'model = "Hubbard"\nlattice = "wannier90"\nW = 2\nL = 2\n', square_files()
    )
    c["err_wannier_L_box_conflict"] = (sq + "a0W = 2\n", square_files())
    c["err_wannier_zero_cell"] = (
        'model = "Hubbard"\nlattice = "wannier90"\na0W = 1\na0L = 1\na1W = 2\na1L = 2\nncond = 4\n',
        square_files(),
    )
    c["wannier_square_J_ignored"] = (sq + "J = 1.0\n", square_files())
    return c
