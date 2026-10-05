"""Generate tests/fixtures/stdface/<case>/ from the C StdFace (mVMC solver, mvmc_dry.out build).

Explicit developer command (Rust builds and tests never run this):

    c_toolbox/stdface/build_reference.sh /tmp/stdface-c
    uv run --no-project python c_toolbox/stdface/generate_fixtures.py /tmp/stdface-c/mvmc_dry.out

For every case the input `StdFace.def` is run through the C binary in an empty directory. The
generated Expert files, the C `stdout` and the exit status are stored under
`tests/fixtures/stdface/<case>/expected/`. `tests/fixtures/stdface/PROVENANCE.md` records the
source hashes, compiler, flags and commands.
"""
import hashlib
import cases_3d
import cases_2d
import cases_defects
import cases_ladder
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STD = ROOT / "extern/mVMC-1.3.0/src/StdFace"
UPSTREAM = STD / "test/mvmc"
FIXTURES = ROOT / "tests/fixtures/stdface"

# Upstream cases (src/StdFace/test/mvmc) that use the chain lattice.
UPSTREAM_CASES = [
    "HubbardChain",
    "HubbardChain_cmp",
    "HubbardChain_fsz",
    "HubbardChainLanczos",
    "HeisenbergChain",
    "HeisenbergChain_cmp",
    "HeisenbergChain_fsz",
    "SpinChainLanczos",
    "KondoChain",
    "KondoChain_cmp",
    "KondoChain_fsz",
    "KondoChain_Stot1_cmp",
    "HubbardTetragonal",
    "HubbardTetragonal_MomentumProjection",
]

HUBBARD = 'model = "Hubbard"\nlattice = "chain"\nL = 6\nU = 4.0\nt = 1.0\nncond = 6\n'
SPIN = 'model = "Spin"\nlattice = "chain"\nL = 6\nJ = 1.0\n2Sz = 0\n'
KONDO = 'model = "KondoLattice"\nlattice = "chain"\nL = 4\nt = 1.0\nJ = 4.0\nncond = 4\n'

# Additional cases. `None` means "the input file does not exist".
EXTRA_CASES = {
    # --- valid models: parameter families the upstream tests do not cover -------------------
    "hubbard_chain_antiperiodic": HUBBARD + "phase0 = 180\nLsub = 3\n",
    "hubbard_chain_long_range": (
        HUBBARD + "mu = -0.5\nt' = 0.5\nt'' = 0.2\nV = 1.0\nV' = 0.3\nV'' = 0.1\n"
    ),
    "hubbard_chain_complex_hopping": (
        'model = "Hubbard"\nlattice = "chain"\nL = 4\nU = 2.0\nt = 1.0, 0.5\nt\' = -0.3, 0.2\n'
        "ncond = 4\nComplexType = 1\n"
    ),
    "hubbard_gc_fields_gamma_y": (
        'model = "HubbardGC"\nlattice = "chain"\nL = 4\nU = 3.0\nt = 1.0\nh = -0.7\n'
        "Gamma = 0.4\nGamma_y = 0.25\nLsub = 2\nncond = 4\n"
    ),
    "hubbard_chain_nmptrans_0": HUBBARD + "NMPTrans = 0\nLsub = 3\n",
    "hubbard_chain_nmptrans_1": HUBBARD + "NMPTrans = 1\n",
    "hubbard_chain_outputmode_none": HUBBARD + 'outputmode = "none"\n',
    "hubbard_chain_outputmode_raw": (
        'model = "Hubbard"\nlattice = "chain"\nL = 2\nU = 4.0\nt = 1.0\nncond = 2\n'
        'outputmode = "raw"\n'
    ),
    "hubbard_chain_modpara_overrides": (
        HUBBARD
        + "CParaFileHead = 'opt'\nNVMCCalMode = 1\nNDataQtySmp = 3\n"
        "NLanczosMode = 2\nNDataIdxStart = 4\nNSROptItrStep = 77\nNVMCWarmUp = 5\n"
        "NVMCInterval = 2\nNVMCSample = 13\nNSplitSize = 2\nNStore = 0\nNSRCG = 1\n"
        "RndSeed = 42\nDSROptRedCut = 1e-3\n"
    ),
    "hubbard_chain_open_dimensions": (
        'model = "FermionHubbard"\nlattice = "ChainLattice"\nL = 5\nU = 1\nt = 2\nncond = 5\n'
        "2Sz = 1\n"
    ),
    "spin_chain_long_range": SPIN + "J' = 0.4\nJ'' = 0.1\nLsub = 2\n",
    "spin_chain_anisotropic": (
        'model = "Spin"\nlattice = "chain"\nL = 4\nJx = 1.0\nJy = 0.5\nJz = 2.0\n2Sz = 0\n'
        "NVMCSample = 10\n"
    ),
    "spin_chain_offdiagonal": (
        'model = "Spin"\nlattice = "chain"\nL = 4\nJx = 1.0\nJy = 1.0\nJz = 1.0\nJxy = 0.3\n'
        "Jyx = -0.2\nJxz = 0.1\nJyz = 0.4\nJzx = -0.15\nJzy = 0.35\n2Sz = 0\n"
        "ComplexType = 1\n"
    ),
    "spin_gc_fields_gamma_y": (
        'model = "SpinGC"\nlattice = "chain"\nL = 4\nJ = 1.0\nh = 0.3\nGamma = 0.2\n'
        "Gamma_y = -0.6\n"
    ),
    "spin_chain_antiperiodic": SPIN + "phase0 = 180\n",
    "spin_chain_isotropic_alias": (
        'model = "Spin"\nlattice = "chain"\nL = 6\nJ0 = 1.0\nJ0\' = 0.5\n2Sz = 2\n'
    ),
    "kondo_chain_anisotropic_j": (
        'model = "KondoLattice"\nlattice = "chain"\nL = 4\nt = 1.0\nJx = 2.0\nJy = 1.5\nJz = 1.0\n'
        "ncond = 4\n"
    ),
    "kondo_gc_fields_gamma_y": (
        'model = "KondoGC"\nlattice = "chain"\nL = 3\nt = 1.0\nJ = 1.0\nncond = 3\nh = -0.4\n'
        "Gamma = 0.3\nGamma_y = 0.2\nU = 2.0\nmu = 0.1\n"
    ),
    "kondo_chain_antiperiodic": KONDO + "phase0 = 180\nLsub = 2\n",
    "kondo_chain_nmptrans_0": KONDO + "NMPTrans = 0\nLsub = 2\n",
    # --- reader behaviour -------------------------------------------------------------------
    "reader_comments_quotes_case": (
        "// leading comment\n\n   MODEL = \"hubbard\" ;\nLATTICE = chain\n  L=6\nu = 4.0\n"
        "T = 1.0\nNCOND = 6\n  // another comment\nCParaFileHead = \"MixedCase\"\n"
    ),
    "reader_tab_and_trailing_text": (
        HUBBARD + "NSROptItrStep = 12abc\nRndSeed = 7 # trailing\nNVMCSample = 3.9\n"
    ),
    "reader_long_line_split_value": (
        HUBBARD + "// " + "x" * 200 + "\nNVMCSample = 4\n"
    ),
    "reader_extra_equals": HUBBARD + "NSROptItrStep == 33\nRndSeed = 5 = 6\n",
    # --- error paths ------------------------------------------------------------------------
    "err_unknown_keyword": HUBBARD + "FooBar = 1\n",
    "err_cdatafilehead_keyword": HUBBARD + "CDataFileHead = out\n",
    "err_colon_separator": 'model : "Hubbard"\n',
    "err_long_comment_split": HUBBARD + "// " + "x" * 300 + "\nNVMCSample = 4\n",
    "err_hphi_keyword": HUBBARD + "method = lanczos\n",
    "err_hphi_2s_keyword": SPIN + "2S = 1\n",
    "err_missing_equals": HUBBARD + "NSROptItrStep 100\n",
    "err_empty_value": HUBBARD + "NSROptItrStep =\n",
    "err_duplicate_keyword": HUBBARD + "U = 5.0\n",
    "err_duplicate_complex": HUBBARD + "t = 2.0\n",
    "err_duplicate_string": HUBBARD + 'lattice = "chain"\n',
    "err_missing_L": 'model = "Hubbard"\nlattice = "chain"\nU = 4.0\nt = 1.0\nncond = 6\n',
    "err_missing_ncond": 'model = "Hubbard"\nlattice = "chain"\nL = 6\nU = 4.0\nt = 1.0\n',
    "err_missing_2sz": 'model = "Spin"\nlattice = "chain"\nL = 6\nJ = 1.0\n',
    "err_empty_input": "",
    "err_missing_model": 'lattice = "chain"\nL = 6\n',
    "err_unsupported_model": 'model = "Boson"\nlattice = "chain"\nL = 6\n',
    "err_unsupported_lattice": 'model = "Hubbard"\nlattice = "mobius"\nL = 6\n',
    "err_unsupported_outputmode": HUBBARD + "outputmode = bogus\n",
    "err_chain_width_not_used": HUBBARD + "W = 2\n",
    "err_chain_phase1_not_used": HUBBARD + "phase1 = 90\n",
    "err_chain_t1_not_used": HUBBARD + "t1 = 1.0\n",
    "err_chain_V1_not_used": HUBBARD + "V1 = 1.0\n",
    "err_chain_K_not_used": HUBBARD + "K = 1.0\n",
    "err_hubbard_J_not_used": HUBBARD + "J = 1.0\n",
    "err_hubbard_D_not_used": HUBBARD + "D = 1.0\n",
    "err_spin_U_not_used": SPIN + "U = 1.0\n",
    "err_spin_t_not_used": SPIN + "t = 1.0\n",
    "err_spin_ncond_not_used": SPIN + "ncond = 6\n",
    "err_spin_J_conflict": SPIN + "J0 = 2.0\n",
    "err_spin_J_component_conflict": SPIN + "Jx = 2.0\n",
    "err_spin_J0_component_conflict": 'model = "Spin"\nlattice = "chain"\nL = 6\nJ0 = 1.0\nJ0x = 2.0\n2Sz = 0\n',
    "err_hubbard_t_conflict": HUBBARD + "t0 = 2.0\n",
    "err_hubbard_V_conflict": HUBBARD + "V = 1.0\nV0 = 2.0\n",
    "err_L_box_conflict": HUBBARD + "a0W = 6\n",
    "err_ndataqtysmp_unused": HUBBARD + "NDataQtySmp = 2\n",
    "err_nsroptitrsmp_unused": HUBBARD + "NVMCCalMode = 1\nNSROptItrSmp = 2\n",
    "err_gc_2sz_not_used": 'model = "HubbardGC"\nlattice = "chain"\nL = 4\nU = 1.0\nt = 1.0\nncond = 4\n2Sz = 0\n',
    "err_gc_nspgaussleg_not_used": 'model = "HubbardGC"\nlattice = "chain"\nL = 4\nU = 1.0\nt = 1.0\nncond = 4\nNSPGaussLeg = 4\n',
    "err_sublattice_conflict": HUBBARD + "Lsub = 2\na0Wsub = 2\n",
    "err_sublattice_incommensurate": HUBBARD + "Lsub = 4\n",
    "err_sublattice_zero": HUBBARD + "Lsub = 0\n",
    "err_zero_cell": 'model = "Hubbard"\nlattice = "chain"\nL = 0\nU = 1.0\nt = 1.0\nncond = 0\n',
    "err_missing_input_file": None,
}


EXTRA_CASES.update(cases_ladder.CASES)
EXTRA_CASES.update(cases_3d.CASES)
EXTRA_CASES.update(cases_2d.CASES)
EXTRA_CASES.update(cases_defects.CASES)
# Upstream sample (HPhi keywords: rejected by the mVMC build of StdFace).
EXTRA_CASES["sample_hubbard_default_model"] = (STD / "samples/hubbard/default_model/stan.in").read_text()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(binary, case_dir, input_text, subdir="expected"):
    expected = case_dir / subdir
    if expected.exists():
        shutil.rmtree(expected)
    expected.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        if input_text is not None:
            shutil.copyfile(case_dir / "StdFace.def", work / "StdFace.def")
        proc = subprocess.run(
            [str(binary), "StdFace.def"], cwd=work, capture_output=True, check=False
        )
        (expected / "stdout.txt").write_bytes(proc.stdout)
        (expected / "exit_status").write_text(f"{proc.returncode}\n")
        if proc.stderr:
            (expected / "stderr.txt").write_bytes(proc.stderr)
        for produced in sorted(work.iterdir()):
            if produced.name == "StdFace.def":
                continue
            shutil.copyfile(produced, expected / produced.name)
    return proc.returncode


def write_provenance_3d(rows_3d, sources, fixed_3d):
    patch = ROOT / "c_toolbox/stdface/3d_defects.patch"
    lines = [
        "# StdFace 3D lattices (#356): fixture provenance",
        "",
        "Cases from `c_toolbox/stdface/cases_3d.py`, generated by `generate_fixtures.py` with two C",
        "builds of the vendored StdFace (flags and environment as in `PROVENANCE.md`):",
        "",
        "- `mvmc_dry.out`: unmodified C (`build_reference.sh`), the historical behaviour;",
        "- `mvmc_dry_3d_fixed.out`: the same sources with `c_toolbox/stdface/3d_defects.patch`",
        f"  (SHA-256 `{sha256(patch)}`) applied to a copy (`build_reference_3d_fixed.sh`).",
        "",
        "`<case>/expected/` is the corrected-build output (what Rust must reproduce byte for byte);",
        "where it differs from the unmodified C output the latter is kept in `<case>/c_historical/`",
        "(the C defects are listed in `README_3d_defects.md`). Cases without `c_historical/` are",
        "byte-identical in both builds, i.e. identical to the unmodified C.",
        "",
        f"Fixed binary SHA-256: `{sha256(fixed_3d)}`.",
        "",
        "```",
        "c_toolbox/stdface/build_reference.sh /tmp/stdface-c",
        "c_toolbox/stdface/build_reference_3d_fixed.sh /tmp/stdface-c",
        "uv run --no-project python c_toolbox/stdface/generate_fixtures.py \\",
        "  /tmp/stdface-c/mvmc_dry.out /tmp/stdface-c/mvmc_dry_3d_fixed.out",
        "```",
        "",
        "| case | origin | exit status (corrected) | has c_historical | input SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case, origin, status in rows_3d:
        input_path = FIXTURES / case / "StdFace.def"
        historical = "yes" if (FIXTURES / case / "c_historical").exists() else "no"
        lines.append(
            f"| `{case}` | {origin} | {status} | {historical} | `{sha256(input_path)}` |"
        )
    (FIXTURES / "PROVENANCE_3d.md").write_text("\n".join(lines) + "\n")


def tree_files(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def run_case_3d(historical, fixed, case_dir, text, declared_defect):
    """`expected/` = corrected build; `c_historical/` = unmodified C, kept only where they differ."""
    status = run_case(fixed, case_dir, text)
    historical_dir = case_dir / "c_historical"
    if historical_dir.exists():
        shutil.rmtree(historical_dir)
    run_case(historical, case_dir, text, subdir="c_historical")
    differs = tree_files(case_dir / "expected") != tree_files(historical_dir)
    if differs != declared_defect:
        sys.exit(
            f"{case_dir.name}: unmodified and corrected C {'differ' if differs else 'agree'} "
            f"but cases_3d.DEFECT_CASES says the opposite"
        )
    if not differs:
        shutil.rmtree(historical_dir)
    return status


def same_tree(a, b):
    names_a = sorted(p.name for p in a.iterdir())
    names_b = sorted(p.name for p in b.iterdir())
    return names_a == names_b and all((a / n).read_bytes() == (b / n).read_bytes() for n in names_a)


def run_both(binary, fixed, case_dir, input_text):
    """C output in `expected/`; if the corrected build differs, also `expected_fixed/` (#404)."""
    status = run_case(binary, case_dir, input_text)
    fixed_dir = case_dir / "expected_fixed"
    if fixed_dir.exists():
        shutil.rmtree(fixed_dir)
    if fixed is not None:
        run_case(fixed, case_dir, input_text, "expected_fixed")
        if same_tree(case_dir / "expected", fixed_dir):
            shutil.rmtree(fixed_dir)
    return status


def main():
    binary = Path(sys.argv[1]).resolve()
    # Argument 2: the build with 3d_defects.patch (build_reference_3d_fixed.sh), see #356.
    fixed_3d = Path(sys.argv[2]).resolve()
    # Argument 3 (optional): the build with lattice_defects.patch (`build_reference.sh --fixed`).
    fixed = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rows = []
    rows_3d = []
    for case in UPSTREAM_CASES:
        case_dir = FIXTURES / case
        case_dir.mkdir(exist_ok=True)
        shutil.copyfile(UPSTREAM / case / "StdFace.def", case_dir / "StdFace.def")
        status = run_both(binary, fixed, case_dir, "present")
        rows.append((case, "upstream test/mvmc", status))
    for case, text in EXTRA_CASES.items():
        case_dir = FIXTURES / case
        case_dir.mkdir(exist_ok=True)
        if text is None:
            (case_dir / "StdFace.def").unlink(missing_ok=True)
        else:
            (case_dir / "StdFace.def").write_text(text)
        if case in cases_3d.CASES:
            status = run_case_3d(binary, fixed_3d, case_dir, text, case in cases_3d.DEFECT_CASES)
            rows_3d.append((case, "generated (3D, #356)", status))
            continue
        status = run_both(binary, fixed, case_dir, text)
        rows.append((case, "generated", status))

    sources = sorted((STD / "src").glob("*.[ch]"))
    lines = [
        "# StdFace C reference fixtures: provenance",
        "",
        "Generated by `c_toolbox/stdface/generate_fixtures.py` from the C StdFace program",
        "(mVMC solver build, `mvmc_dry.out` = `dry.c` + `StdFace_main` with `-D_mVMC`).",
        "Normal Rust tests only read these checked-in files; they never run C.",
        "",
        "## Reproduction",
        "",
        "```",
        "c_toolbox/stdface/build_reference.sh /tmp/stdface-c",
        "uv run --no-project python c_toolbox/stdface/generate_fixtures.py /tmp/stdface-c/mvmc_dry.out",
        "```",
        "",
        "## Environment",
        "",
        f"- Platform: {platform.platform()} ({platform.machine()})",
        f"- Compiler: {subprocess.run(['gcc', '--version'], capture_output=True, text=True).stdout.splitlines()[0]}",
        "- Flags: `-O3 -DNDEBUG -ffp-contract=off -w -D_mVMC -DMEXP=19937` (the upstream CMake",
        "  Release flags are `-O3 -DNDEBUG`; contraction is disabled to match the baseline x86-64",
        "  target and the portable numerical policy). `build_reference.sh` documents the exact line.",
        "- Upstream: `extern/mVMC-1.3.0` (git tag v1.3.0), StdFace submodule",
        "  `6fa4ef1f6809001a24b6501bd08f988bdc8cb7c4`.",
        "- BLAS/MPI: none (StdFace is plain C; `StdFace_exit` calls `MPI_Abort` only with `-DMPI`).",
        "",
        "## C source SHA-256 (`extern/mVMC-1.3.0/src/StdFace/src/`)",
        "",
    ]
    for source in sources:
        lines.append(f"- `{source.name}`: `{sha256(source)}`")
    lines += [
        "",
        "## Layout",
        "",
        "`<case>/StdFace.def` is the input (absent for `err_missing_input_file`);",
        "`<case>/expected/` holds every file the C program wrote plus `stdout.txt` (the complete",
        "C `stdout`), `exit_status` (255 = `StdFace_exit(-1)`) and, if any, `stderr.txt`.",
        "`<case>/expected_fixed/` (only where it differs) is the output of the *corrected* build",
        "(`build_reference.sh --fixed`: `lattice_defects.patch` applied to a copy of the C sources, see",
        "`tests/fixtures/stdface/README.md`). `expected/` stays the historical C output; the Rust",
        "port is compared with `expected_fixed/` when present, else with `expected/`.",
        f"- `c_toolbox/stdface/lattice_defects.patch` SHA-256: `{sha256(ROOT / 'c_toolbox/stdface/lattice_defects.patch')}`",
        "",
        "## Cases",
        "",
        "| case | origin | C exit status | input SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    corrected = sorted(r[0] for r in rows if (FIXTURES / r[0] / "expected_fixed").exists())
    lines.insert(
        lines.index("## Cases"),
        "## Cases whose C output differs from the corrected build (`expected_fixed/`)\n\n"
        + "\n".join(f"- `{c}`" for c in corrected)
        + "\n",
    )
    for case, origin, status in rows:
        input_path = FIXTURES / case / "StdFace.def"
        digest = sha256(input_path) if input_path.exists() else "(no input file)"
        lines.append(f"| `{case}` | {origin} | {status} | `{digest}` |")
    (FIXTURES / "PROVENANCE.md").write_text("\n".join(lines) + "\n")
    write_provenance_3d(rows_3d, sources, fixed_3d)
    print(f"generated {len(rows) + len(rows_3d)} cases")


if __name__ == "__main__":
    main()
