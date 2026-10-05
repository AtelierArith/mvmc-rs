#!/usr/bin/env python3
"""Generate the greenr2k parity fixtures (issue #351).

Usage (Linux, repository root, after `c_toolbox/greenr2k/build.sh <bin>`):

    uv run --no-project c_toolbox/greenr2k/generate_fixtures.py <bin> tests/fixtures/greenr2k

For every case this script writes `<case>/inputs/` (namelist, ModPara, geometry,
index files and the correlation files that greenr2k reads) and `<case>/expected/`
(everything the authoritative Fortran `greenr2k` printed or wrote when run in a
copy of `inputs/`).  Geometry and index files of the lattice cases come from the
authoritative C StdFace (`mvmc_dry.out`); the chain6_mvmc case reuses the mVMC
index files and correlation values of tests/fixtures/physcal_181.  All other
correlation values are synthetic (seeded `random.Random`): greenr2k is a pure
linear post-processing tool, so only the file contract matters.
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile

BIN = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHYS = os.path.join(
    ROOT, "tests/fixtures/physcal_181/two-samples/hubbard_chain_dh_real"
)

CHAIN_STAN = 'model = "Hubbard"\nlattice = "chain"\nL = 6\nt = 1.0\nU = 4.0\nnelec = 6\n2Sz = 0\n'
HC_STAN = (
    'model = "Hubbard"\nlattice = "honeycomb"\na0W = 2\na0L = 0\na1W = 0\na1L = 2\n'
    "t = 1.0\nU = 4.0\nnelec = 8\n2Sz = 0\nphase0 = 30.0\nphase1 = 45.0\n"
)
CHAIN_KPATH = "2 4\nG 0 0 0\nX 0 0.5 0\n1 6 1\n"
HC_KPATH = "3 4\nG 0 0 0\nM 0.5 0 0\nK 0.3333333333 0.3333333333 0\n4 4 1\n"


def stdface(stan, workdir):
    os.makedirs(workdir)
    with open(os.path.join(workdir, "stan.in"), "w") as f:
        f.write(stan)
    subprocess.run(
        [os.path.join(BIN, "mvmc_dry.out"), "stan.in"],
        cwd=workdir,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def sites_of(geometry_text):
    lines = geometry_text.strip().splitlines()
    return [tuple(map(int, l.split())) for l in lines[7:]]


def hphi_indices(sites):
    """One-/two-body index sets in the HPhi style greenr2k expects (calctype != 4)."""
    r0 = [i for i, s in enumerate(sites) if s[:3] == (0, 0, 0)]
    one = []
    two = []
    for i in r0:
        for j in range(len(sites)):
            for s in (0, 1):
                one.append((i, s, j, s))
            for s1 in (0, 1):
                for s2 in (0, 1):
                    two.append((i, s1, i, s1, j, s2, j, s2))
            two.append((i, 0, i, 1, j, 1, j, 0))
            two.append((i, 1, i, 0, j, 0, j, 1))
    return one, two


def write_def(path, name, rows, title):
    with open(path, "w") as f:
        f.write("=" * 15 + "\n")
        f.write(f"{name} {len(rows):8d}\n")
        f.write("=" * 15 + "\n")
        f.write(f"======== {title} ======\n")
        f.write("=" * 15 + "\n")
        for r in rows:
            f.write(" ".join(f"{x:5d}" for x in r) + "\n")


def read_def_rows(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f):
            if n >= 5 and line.strip():
                rows.append(tuple(map(int, line.split())))
    return rows


def write_corr(path, rows, rng):
    with open(path, "w") as f:
        for r in rows:
            re = rng.uniform(-0.5, 0.5)
            im = rng.uniform(-0.01, 0.01)
            f.write(" ".join(str(x) for x in r) + f" {re:.18e}  {im:.18e} \n")
        f.write("\n")


def write_modpara(path, **kw):
    with open(path, "w") as f:
        f.write("--------------------\nModel_Parameters   0\n--------------------\n")
        f.write("CDataFileHead  zvo\nCParaFileHead  zqp\n--------------------\n")
        for k, v in kw.items():
            f.write(f"{k} {v}\n")


def run_case(name, setup, may_fail=False):
    case = os.path.join(OUT, name)
    shutil.rmtree(case, ignore_errors=True)
    inputs = os.path.join(case, "inputs")
    os.makedirs(os.path.join(inputs, "output"))
    setup(inputs)
    work = tempfile.mkdtemp()
    shutil.copytree(inputs, os.path.join(work, "run"))
    run = os.path.join(work, "run")
    proc = subprocess.run(
        [os.path.join(BIN, "greenr2k"), "namelist.def", "geometry.dat"],
        cwd=run,
        capture_output=True,
        text=True,
        check=not may_fail,
    )
    exp = os.path.join(case, "expected")
    os.makedirs(os.path.join(exp, "output"))
    with open(os.path.join(exp, "stdout.txt"), "w") as f:
        f.write(proc.stdout)
    if may_fail:
        with open(os.path.join(exp, "exit_code.txt"), "w") as f:
            f.write(f"{proc.returncode}\n")
        with open(os.path.join(exp, "stderr.txt"), "w") as f:
            f.write(proc.stderr)
    else:
        assert proc.stderr == "", proc.stderr
    for rel in sorted(os.listdir(run)):
        if rel == "kpath.gp":
            shutil.copy(os.path.join(run, rel), exp)
    for rel in sorted(os.listdir(os.path.join(run, "output"))):
        if rel.startswith("zvo_corr"):
            shutil.copy(os.path.join(run, "output", rel), os.path.join(exp, "output"))
    shutil.rmtree(work)


def sd_dir(label, stan):
    d = os.path.join(tempfile.mkdtemp(), label)
    stdface(stan, d)
    return d


def geometry_with(sd, kpath):
    with open(os.path.join(sd, "geometry.dat")) as f:
        return f.read() + kpath


def namelist(extra=""):
    return (
        "         ModPara  modpara.def\n         LocSpin  locspn.def\n"
        "        OneBodyG  greenone.def\n        TwoBodyG  greentwo.def\n" + extra
    )


def case_chain6_mvmc(inputs):
    sd = sd_dir("chain", CHAIN_STAN)
    with open(os.path.join(inputs, "geometry.dat"), "w") as f:
        f.write(geometry_with(sd, CHAIN_KPATH))
    for n in ("namelist.def", "modpara.def", "greenone.def", "greentwo.def"):
        shutil.copy(os.path.join(PHYS, "inputs", n), inputs)
    for idx in ("007", "008"):
        for h in ("cisajs", "cisajscktalt"):
            shutil.copy(
                os.path.join(PHYS, "expected", f"zvo_{h}_{idx}.dat"),
                os.path.join(inputs, "output"),
            )


def case_chain6_missing_index(inputs):
    """The last two-body index of chain6_mvmc is removed: greenr2k must stop."""
    case_chain6_mvmc(inputs)
    path = os.path.join(inputs, "greentwo.def")
    lines = open(path).read().splitlines()
    assert lines[1].split()[1] == "36"
    lines[1] = lines[1].replace("36", "35")
    open(path, "w").write("\n".join(lines[:-1]) + "\n")


def case_honeycomb_mvmc(inputs):
    sd = sd_dir("hc", HC_STAN)
    with open(os.path.join(inputs, "geometry.dat"), "w") as f:
        f.write(geometry_with(sd, HC_KPATH))
    for n in ("namelist.def", "greenone.def", "greentwo.def"):
        shutil.copy(os.path.join(sd, n), inputs)
    with open(os.path.join(sd, "modpara.def")) as f:
        mp = f.read()
    mp = mp.replace("NDataIdxStart  1", "NDataIdxStart  3").replace(
        "NDataQtySmp    1", "NDataQtySmp    3"
    )
    assert "NDataIdxStart  3" in mp and "NDataQtySmp    3" in mp
    with open(os.path.join(inputs, "modpara.def"), "w") as f:
        f.write(mp)
    rng = random.Random(351)
    one = read_def_rows(os.path.join(inputs, "greenone.def"))
    two = read_def_rows(os.path.join(inputs, "greentwo.def"))
    for idx in (3, 4, 5):
        write_corr(os.path.join(inputs, "output", f"zvo_cisajs_{idx:03d}.dat"), one, rng)
        write_corr(os.path.join(inputs, "output", f"zvo_cisajscktalt_{idx:03d}.dat"), two, rng)


def hphi_case(stan, kpath, calctype, modpara_kw, tails, seed, extra_files=None):
    def setup(inputs):
        sd = sd_dir("sd", stan)
        with open(os.path.join(inputs, "geometry.dat"), "w") as f:
            f.write(geometry_with(sd, kpath))
        sites = sites_of(open(os.path.join(sd, "geometry.dat")).read())
        one, two = hphi_indices(sites)
        write_def(os.path.join(inputs, "greenone.def"), "NCisAjs", one, "Green functions")
        write_def(
            os.path.join(inputs, "greentwo.def"), "NCisAjsCktAltDC", two, "Green functions"
        )
        write_modpara(os.path.join(inputs, "modpara.def"), Nsite=len(sites), **modpara_kw)
        open(os.path.join(inputs, "locspn.def"), "w").write("dummy\n")
        with open(os.path.join(inputs, "calcmod.def"), "w") as f:
            f.write(f"#CalcType\nCalcType {calctype}\nCalcModel 1\n")
        with open(os.path.join(inputs, "namelist.def"), "w") as f:
            f.write(namelist("         CalcMod  calcmod.def\n"))
        if extra_files:
            for rel, text in extra_files.items():
                open(os.path.join(inputs, rel), "w").write(text)
        rng = random.Random(seed)
        for t in tails:
            write_corr(os.path.join(inputs, "output", f"zvo_cisajs{t}"), one, rng)
            write_corr(os.path.join(inputs, "output", f"zvo_cisajscktalt{t}"), two, rng)

    return setup


def main():
    os.makedirs(OUT, exist_ok=True)
    run_case("chain6_mvmc", case_chain6_mvmc)
    run_case("honeycomb_mvmc", case_honeycomb_mvmc)
    run_case("chain6_missing_index", case_chain6_missing_index, may_fail=True)
    run_case(
        "honeycomb_lanczos",
        hphi_case(HC_STAN, HC_KPATH, 0, {}, [".dat"], 7),
    )
    run_case(
        "chain6_tpq",
        hphi_case(
            CHAIN_STAN,
            CHAIN_KPATH,
            1,
            {"NumAve": 2, "Lanczos_max": 3, "ExpecInterval": 2},
            [f"_set{r}step{s}.dat" for s in (0, 2) for r in (0, 1)],
            11,
        ),
    )
    run_case(
        "chain6_lobcg",
        hphi_case(CHAIN_STAN, CHAIN_KPATH, 3, {"Exct": 2}, ["_eigen0.dat", "_eigen1.dat"], 17),
    )


main()
