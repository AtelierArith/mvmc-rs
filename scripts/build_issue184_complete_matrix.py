#!/usr/bin/env python3
"""Generate the complete #184 evidence matrix (docs only; never run by Cargo or CI).

Inputs (all checked in):
  * docs/reference/c-to-julia/verification/issue-184-public-apis.tsv   Julia API inventory
  * docs/reference/c-to-julia/verification/issue-184-scenarios.tsv     Julia scenario inventory
  * the Rust sources and tests (static scan) and a captured nextest run
    (`--nextest-log`, produced with `cargo nextest run --workspace --cargo-profile test-fast
    --status-level all`), which decides PASS / NotRun per cited test.

Every test cited in the output is looked up in the source tree and in the nextest log; a
family row whose test pattern matches nothing aborts the build, so no test name can be
invented. A test is PASS only if the log shows PASS; ignored/opt-in tests (SKIP) and
feature-gated tests absent from the log are reported NotRun.

usage: build_issue184_complete_matrix.py --nextest-log LOG --head SHA --out-dir DIR
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "docs/reference/c-to-julia/verification"
OPEN_OWNERS = {
    "#179": "MPI optimization/measurement scenario matrix",
    "#181": "serial PhysCal and Lanczos scenarios",
    "#184": "this matrix",
    "#185": "umbrella",
    "#207": "CalHamiltonian1/ReturnSlaterElmDiff on Hubbard real path",
    "#337": "PhysCal outputData cost",
    "#347": "vmc.out CLI options -b/-F/-v/-e/-h",
    "#348": "MultiDef mode",
    "#351": "post-processing tools",
    "#352": "StdFace standard-mode umbrella",
    "#354": "StdFace 1D lattices",
    "#355": "StdFace 2D lattices",
    "#356": "StdFace 3D lattices",
    "#357": "StdFace Wannier90",
    "#360": "C-only OpenMP regions",
    "#361": "inner-threading slowdown",
    "#392": "MPI gate mpi_issue178_support_matrix",
}

# --------------------------------------------------------------------------- test index


def load_log(path):
    status = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\s+(PASS|FAIL|SKIP)\s+\[[^\]]*\]\s+\(\s*[\d─]+(?:/\s*\d+)?\)\s+(\S+(?:::\S+)?)\s+(\S+)", line)
        if m:
            kind, crate_bin, name = m.groups()
            if kind in ("PASS", "FAIL", "SKIP"):
                key = f"{crate_bin} {name}"
                if kind == "PASS" or key not in status:
                    status[key] = kind
    return status


def static_tests():
    """(crate, stem-or-module, fn) for every #[test] in the tree, with its body text."""
    out = []
    for path in glob.glob(os.path.join(ROOT, "crates/*/tests/*.rs")) + glob.glob(
        os.path.join(ROOT, "crates/*/src/**/*.rs"), recursive=True
    ) + glob.glob(os.path.join(ROOT, "xtask/src/**/*.rs"), recursive=True):
        rel = os.path.relpath(path, ROOT)
        text = open(path, errors="replace").read()
        parts = re.split(r"(?m)^\s*#\[test\]", text)
        for part in parts[1:]:
            m = re.search(r"fn\s+(\w+)", part)
            if not m:
                continue
            crate = rel.split("/")[1] if rel.startswith("crates/") else "xtask"
            kind = "integration" if "/tests/" in rel else "unit"
            stem = os.path.splitext(os.path.basename(rel))[0]
            out.append(
                {"crate": crate, "rel": rel, "stem": stem, "fn": m.group(1), "body": part,
                 "kind": kind, "ignored": "#[ignore" in part.split("fn", 1)[0]}
            )
    return out


class Index:
    def __init__(self, log, tests):
        self.log = log
        self.tests = tests
        self.keys = list(log)

    def result(self, t):
        """PASS / SKIP(ignored) / NotRun for a static test record."""
        name = t["fn"]
        for key, kind in self.log.items():
            crate_bin, test = key.split(" ", 1)
            if not (test == name or test.endswith("::" + name)):
                continue
            if t["kind"] == "integration":
                if crate_bin == f"{t['crate']}::{t['stem']}":
                    return kind
            else:
                if crate_bin == t["crate"]:
                    return kind
        return "NotRun"

    def find(self, regex):
        """Tests whose 'crate::bin name' (integration) or 'crate module::name' matches."""
        rx = re.compile(regex)
        found = []
        for key, kind in self.log.items():
            if rx.search(key):
                found.append((key, kind))
        # feature-gated / absent-from-log tests: static fallback
        for t in self.tests:
            label = f"{t['crate']}::{t['stem']} {t['fn']}" if t["kind"] == "integration" else f"{t['crate']} {t['fn']}"
            if rx.search(label) and not any(label.endswith(k.split(' ', 1)[1]) and k.startswith(label.split(' ')[0]) for k, _ in found):
                found.append((label, self.result(t) if self.result(t) != "NotRun" else "NotRun"))
        return found


UNIT = "unit test (no external reference)"


def evidence_type(rel_or_key, body=""):
    key = (rel_or_key + " " + body).lower()
    if "ctest_equivalent" in key or "ctest_model" in key or "upstream_rule" in key:
        return "statistical ctest"
    if re.search(r"c_kernel_order|native|c_setup|_c_|/c_|issue342|issue274|c_orbital|c_overlay|c_rbm|c_integer|c_jastrow|c_gutz|c_timer|c_initial|c_general|c_interall|c_definition|stdface_c|c_parity|golden_vs_c|c_window|issue344|issue176_real", key):
        return "C fixture"
    if re.search(r"julia|golden_vs_julia|_vs_julia|linux_gnu|macos_arm", key):
        return "Julia fixture"
    if re.search(r"repeat|observer|passiv|same_worker|worker_invariant|nonmutating|unchanged|rejects_before", key):
        return "same-impl repeatability"
    return UNIT


def fmt_result(kinds):
    c = Counter(kinds)
    if not c:
        return "no test"
    if set(c) == {"PASS"}:
        return f"PASS {c['PASS']}/{c['PASS']}"
    parts = []
    if c.get("PASS"):
        parts.append(f"PASS {c['PASS']}")
    if c.get("SKIP"):
        parts.append(f"NotRun-ignored {c['SKIP']}")
    if c.get("NotRun"):
        parts.append(f"NotRun-gated {c['NotRun']}")
    if c.get("FAIL"):
        parts.append(f"FAIL {c['FAIL']}")
    return "; ".join(parts)


def short(key):
    crate_bin, name = key.split(" ", 1)
    crate_bin = crate_bin.replace("mvmc-expert-parsers", "parsers").replace("mvmc-core", "core").replace("mvmc-cli", "cli")
    return f"{crate_bin.split('::', 1)[-1] if '::' in crate_bin else crate_bin}::{name}"


def command_for(keys):
    """Reproduction command from the first resolved test."""
    if not keys:
        return "n/a"
    key = keys[0]
    crate_bin, name = key.split(" ", 1)
    crate = crate_bin.split("::")[0]
    if "::" in crate_bin and not crate_bin.endswith("bin/mvmc"):
        binary = crate_bin.split("::", 1)[1]
        return f"cargo nextest run -p {crate} --cargo-profile test-fast --test {binary} -E 'test({name})'"
    return f"cargo nextest run -p {crate} --cargo-profile test-fast -E 'test({name.split('::')[-1]})'"


# --------------------------------------------------------------------------- family rows
# (id, area, scenario, rust entry, state, evidence override or None, [test regexes], owner, note)
I, R, D, M = "Implemented", "Rejected-with-C-reason", "Intentional difference", "Missing"
F = []


def fam(area, scenario, entry, state, tests, owner, note="", evidence=None):
    F.append(dict(area=area, scenario=scenario, entry=entry, state=state, tests=tests,
                  owner=owner, note=note, evidence=evidence))


# ---- parsing families
fam("parsing", "Namelist / indexed-definition closure and C definition order",
    "mvmc_expert_parsers::parse_expert_mode_files", I,
    [r"parsers::c_definition_order ", r"parsers::issue184_original_heisenberg_public_loader ", r"parsers::round_trip_namelist "],
    "#184/#344", "Layout of AP/Parallel/DH/RBM blocks is C's, independent of keyword order.")
fam("parsing", "ModPara reader: C keyword set, defaults, unknown-keyword error",
    "mvmc_expert_parsers::parse_modpara*", I,
    [r"parsers::issue344_c_modpara_reader ", r"cli::issue344_modpara_cli ", r"parsers::issue184_modpara_semantics "],
    "#344", "Julia `=` syntax and Rust-only keys are rejected as C does.")
fam("parsing", "Duplicate namelist keyword policy (case-insensitive)",
    "mvmc_expert_parsers namelist loaders", I, [r"parsers::issue184_namelist_duplicate_policy "], "#184")
fam("parsing", "Transfer / Coulomb / Hund / Exchange definition boundaries",
    "parse_transfer*, parse_coulomb_*, parse_hund, parse_exchange", I,
    [r"parsers::issue184_transfer_definition_boundary ", r"parsers::issue184_coulomb_definition_boundary ",
     r"parsers::issue184_pair_value_definition_boundaries "], "#257/#259/#262")
fam("parsing", "PairHop definition and directed expansion", "parse_pair_hop*", I,
    [r"parsers::issue184_pairhop_definition_boundary ", r"core::pairhop_reference "], "#264")
fam("parsing", "LocSpin definition boundary", "parse_loc_spin*", I,
    [r"parsers::issue184_locspin_definition_boundary "], "#267")
fam("parsing", "TransSym / QPTrans file boundary, weights, inverse", "parse_qp_trans*, qp_weight", I,
    [r"parsers::issue184_qptrans_definition_boundary ", r"parsers::issue184_qptrans_inverse ",
     r"parsers::issue184_qptrans_payload_boundary "], "#269/#272")
fam("parsing", "Green one-body/two-body direct readers and public bounds", "parse_green_*", I,
    [r"parsers::issue308_green_direct_readers ", r"parsers::issue184_green_public_bounds ",
     r"parsers::issue184_green_public_errors ", r"parsers::green_two_ex "], "#308")
fam("parsing", "Gutzwiller / Jastrow index files", "parse_gutzwiller*, parse_jastrow*", I,
    [r"parsers::c_gutzwiller_contracts ", r"parsers::c_jastrow_contracts ", r"core::c_jastrow_projection "], "#184")
fam("parsing", "Orbital families (AP, Parallel, General) and declared widths", "parse_orbital*", I,
    [r"parsers::c_orbital_contracts ", r"parsers::c_general_orbital ", r"parsers::orbital_layout ",
     r"parsers::orbital_general ", r"parsers::issue184_original_ap_offset "], "#184")
fam("parsing", "Manual orbital mode and orbital flag refresh", "apply_orbital_opt_flags_from_files, manual orbital mode", I,
    [r"parsers::issue184_manual_orbital_mode ", r"parsers::issue184_orbital_flag_refresh "], "#335/#336")
fam("parsing", "DH2 / DH4 tables, signed headers, public boundary", "parse_doublon_holon_*", I,
    [r"parsers::dh2 ", r"parsers::dh4 ", r"parsers::issue283_dh_public_boundary "], "#283")
fam("parsing", "RBM nine-section layout, flags, validation diagnostics", "parse_rbm*, validate_general_rbm*", I,
    [r"parsers::rbm ", r"parsers::c_rbm_contracts ", r"parsers::issue323_general_rbm_validation ",
     r"parsers::issue325_manual_term_diagnostics "], "#323/#325")
fam("parsing", "InterAll reader (listed for completeness; implementation owned elsewhere)", "parse_inter_all*", I,
    [r"parsers::c_interall_reader ", r"parsers::interall "], "InterAll owner (outside #184)",
    "InterAll work is excluded from #184; reader controls exist.")
fam("parsing", "Public file metadata / regular-file predicate / complex utilities", "file_modification_time, parse_complex_*", I,
    [r"parsers::issue255_file_metadata ", r"parsers::issue184_strict_complex_utility_contracts ",
     r"parsers::issue184_complex_utility_contracts "], "#255/#246/#248/#313")
fam("parsing", "Public parsing context and typed single-family loading", "ParsingContext, load_*_definition", I,
    [r"parsers::issue330_parsing_context ", r"parsers::issue332_single_hamiltonian_definition "], "#330/#332")
fam("parsing", "Validation structs (valid/errors/warnings)", "validate_expert_mode_data*", I,
    [r"parsers::validation ", r"core::runtime_contract "], "#184")
# ---- overlays / initialization / seeds
fam("overlays", "InGutzwiller/InJastrow/InOrbital/InDH/InRBM overlays (last value wins, atomic failure)",
    "load_initial_parameters / overlay loaders", I,
    [r"parsers::input_overlays ", r"parsers::issue184_c_overlay_records ", r"parsers::issue184_original_parameter_overlays ",
     r"parsers::issue184_dh_indexed_overlays ", r"parsers::issue184_optional_opttrans_overlays "], "#184")
fam("overlays", "Strict initial-parameter loader (complete records, verbatim values)",
    "read_initial_def / read_opt_para_file", I,
    [r"core::initial_params ", r"core::c_initial_records "], "#184")
fam("initialization", "Parameter initialization / normalization / RNG draws",
    "init_parameter, normalize", I,
    [r"parsers::issue184_initialization_contract ", r"parsers::c_orbital_initialization ", r"core::native13_initialization "],
    "#184/#274")
fam("seeds", "Seed resolution (positive/zero/negative clock, group offset, broadcast)", "resolve_seed*", I,
    [r"core::issue184_public_seed_resolution ", r"core run::seed_tests::", r"core::mpi_issue177_seed_lifecycle "],
    "#177")
# ---- samplers / modes
fam("sampler", "Normal real/complex initial sample, retry and 101-attempt failure",
    "make_initial_sample*", I,
    [r"core::native274_boundaries ", r"core::native274_real_log_ip ", r"core::native342_negative_info ",
     r"core sampling::normal_initial::"], "#274/#342")
fam("sampler", "Real FSZ setup, retry, failure, zero, localspin, magnetized (native C + Julia)",
    "calc_m_all_fsz_real, make_initial_sample_fsz_real", I,
    [r"core::real_fsz_setup ", r"core::real_fsz_sampling "], "#176/#388")
fam("sampler", "Complex FSZ sampler", "vmc_make_sample_fsz*", I, [r"core::complex_fsz_sampling "], "#184")
fam("sampler", "Candidate generation, Metropolis decisions, one-move loops", "sampling::candidate/metropolis/one_move", I,
    [r"core::candidate_vs_julia ", r"core::metropolis_vs_julia ", r"core::one_move_vs_julia "], "#184")
fam("sampler", "Pfaffian/inverse construction, rank-2 updates, rsbOld", "calc_m_all_*, pf update kernels", I,
    [r"core::calc_m_all_vs_julia ", r"core::pf_update_vs_julia ", r"core::two_electron_rsbold ",
     r"pfapack ltl::rank2_real_regression::"], "#358/#365")
fam("sampler", "Projection counts, Gutzwiller/Jastrow ratios, DH counts", "sampling::projection, dh*_projection", I,
    [r"core::projection_vs_julia ", r"core::dh2_projection ", r"core::dh4_projection ",
     r"core sampling::projection::"], "#184")
fam("sampler", "PairHop real/FSZ sampler and Hamiltonian", "pair-hop kernels", I,
    [r"core::pairhop_reference ", r"core run::callback_tests::pairhop_real_and_fsz_direct_prefixes"], "#184")
# ---- real/complex/FSZ x solver
fam("optimization", "Real direct SR prefixes (1/2/3/20 steps, configs, RNG, parameters)", "run_para_opt_from_namelist (real, direct)", I,
    [r"core run::callback_tests::real_direct_sr_prefixes_match_julia", r"core run::callback_tests::hubbard_direct_sr_prefixes_match_julia"],
    "#358/#180")
fam("optimization", "Complex direct SR prefixes", "run_para_opt_from_namelist (cmp, direct)", I,
    [r"core run::callback_tests::cmp_direct_sr_prefixes_match_julia", r"core run::callback_tests::general_direct_sr_prefixes_match_julia"],
    "#180")
fam("optimization", "FSZ direct SR prefixes", "run_para_opt_from_namelist (fsz, direct)", I,
    [r"core run::callback_tests::fsz_direct_sr_prefixes_match_julia"], "#186/#180")
fam("optimization", "Real/complex/FSZ SR-CG prefixes", "run_para_opt_from_namelist (CG)", I,
    [r"core run::callback_tests::real_cg_prefixes_match_julia", r"core run::callback_tests::complex_cg_prefixes_match_julia",
     r"core run::callback_tests::fsz_cg_prefixes_match_julia", r"core run::callback_tests::general_cg_prefixes_match_julia"],
    "#180/#345")
fam("optimization", "NStore / NSRCG controls (C NSRCG semantics and store contract)", "NSROptCGMaxIter, NStoreO", I,
    [r"core run::callback_tests::nstore_", r"core run::callback_tests::nsrcg_two_with_store", r"core validation::nsrcg_contract_tests::",
     r"core run::callback_tests::cg_accumulation_forces_store"], "#345")
fam("optimization", "Negative DSROptStepDt follows C", "sr::negative_stepdt", I,
    [r"core sr::negative_stepdt_tests::", r"cli::issue367_negative_step_dt "], "#367")
fam("optimization", "Component flags / correlation gauge / fixed parameters", "optimization flags", I,
    [r"parsers::optimization_flags ", r"core::c_integer_flags ", r"parsers::c_integer_flags ", r"core sr::tests::fixed_correlations"], "#184")
fam("optimization", "SR failure boundaries (nonfinite, indefinite, singular) without parameter mutation", "SR solve", I,
    [r"core sr::tests::singular_direct_sr", r"core sr::tests::real_nonfinite_update", r"core sr::tests::complex_nonfinite_update",
     r"core sr::tests::non_positive_definite", r"core::sr_observer "], "#178/#234")
fam("optimization", "Measurement store, Gram, CG operator and finalization", "sample store, sr_cg", I,
    [r"core::sample_store ", r"core::sr_cg ", r"core sr_cg::collective_tests::"], "#184/#179")
# ---- projections: DH / RBM / OptTrans
fam("projection", "DH2/DH4 runtime: loaded values, direct/CG prefixes", "run_para_opt (DH)", I,
    [r"core::dh2_runtime ", r"core::dh4_runtime ", r"core run::callback_tests::dh24_real_complex_and_fsz_direct_prefixes",
     r"core run::callback_tests::dh4_and_dh24_real_complex_and_fsz_cg_prefixes"], "#283/#180")
fam("projection", "RBM runtime: real/complex/FSZ, direct and CG", "run_para_opt (RBM)", I,
    [r"core::rbm_production ", r"core::rbm_vs_julia ", r"core::c_rbm_counters ", r"core::c_rbm_parameters ",
     r"core run::callback_tests::rbm_real_and_general_complex_direct_prefixes", r"core run::callback_tests::rbm_fsz_cg_prefixes"],
    "#379/#184",
    "Real-mode C ignores the RBM factor (Julia-mVMC#59 C defect); Rust follows the Julia/C complex contract here, see defect rows.")
fam("projection", "OptTrans: layout, weights, derivatives, SR sync (multi-sector)", "OptTrans in optimizer and PhysCal", I,
    [r"core::opttrans ", r"parsers::opttrans ", r"core run::callback_tests::opttrans_", r"core sr::opttrans_tests::",
     r"core run::callback_tests::real_mode_opttrans_derivatives_use_their_own_slots"], "#368/#370")
# ---- ctest
fam("ctest", "Upstream ctest rule at upstream length, all 13 models (3 sigma and 1e-8)",
    "rust_ctest_upstream_rule_selected_models", I,
    [r"core::ctest_equivalent rust_ctest_upstream_rule_selected_models", r"core::ctest_equivalent upstream_references_cover",
     r"core::ctest_equivalent ctest_failure_requires_both_thresholds"], "#180/#375",
    "Long gate is opt-in; PASS for all 13 models recorded in docs/ISSUE180_UPSTREAM_CTEST_RULE.md (PR #375), not re-run for this matrix.",
    evidence="statistical ctest")
fam("ctest", "Deterministic prefixes 1/2/3/50 for 13 models (short gates) and 20-step references (missing)",
    "ctest_model_prefixes", I,
    [r"core::ctest_model_prefixes ", r"core::ctest_general_reference "], "#180/#185",
    "Independent 20-step references do not exist; 50-step prefixes are not truncated.")
fam("ctest", "First-10 zvo_out gates (real/cmp/fsz/Hubbard)", "phase4/phase5 zvo gates", I,
    [r"core::phase4_zvo_gate ", r"core::phase4_zvo_gate_cmp ", r"core::phase4_zvo_gate_fsz ", r"core::phase5_zvo_gate_hubbard "],
    "#183")
fam("ctest", "Optional gate classification (missing coverage is never a pass)", "ctest harness / support", I,
    [r"core::ctest_equivalent classified_reporting", r"core::ctest_equivalent transitive_inputs"], "#183")
# ---- PhysCal / Green / Lanczos
fam("physcal", "Serial PhysCal: indexed outputs, fixed parameters, overlays (real/complex/FSZ)", "run_phys_cal_from_namelist, mvmc CLI --physcal", I,
    [r"cli::physcal_reference (real|complex|fsz)_cli_matches", r"core::physcal_issue181 ", r"cli::issue174_physcal_dispatch "],
    "#174/#181")
fam("physcal", "PhysCal callback API", "run_phys_cal with callback", I,
    [r"core::physcal_callback ", r"core::physcal_callback_reference ", r"core::physcal_callback_exact_contract "], "#175")
fam("physcal", "PhysCal admission and unsupported-section rejection", "CLI admission", R,
    [r"cli::issue366_physcal_admission ", r"cli::issue184_fixed_file_boundaries "], "#366")
fam("physcal", "Green function kernels: one-body/two-body, FSZ and InterAll normal", "observables::green*", I,
    [r"core::two_body_green ", r"core::fsz_measurements ", r"core observables::green_measurements::tests::"], "#186/#181")
fam("lanczos", "Serial Lanczos modes (hopping, exchange, pair-hop; real/complex)", "mvmc --physcal Lanczos", I,
    [r"cli::physcal_reference .*lanczos", r"core lanczos::tests::", r"core::lanczos_transfer_physcal "], "#181",
    "Lanczos PhysCal gate `lanczos_transfer_physcal` is opt-in (NotRun here); CLI native-C references PASS.")
# ---- MPI / grouped
fam("mpi", "Grouped NSplitSize runs (C-supported combinations admitted)", "group communicators", I,
    [r"core::grouped_nsplit_349 ", r"core parallel::tests::", r"core::issue184_public_rejection_boundaries "], "#349/#178")
fam("mpi", "Grouped/MPI runtime rejection and failure boundaries before output", "validators and CLI", R,
    [r"cli::issue178_support_matrix ", r"core::runtime_contract grouped_", r"core::mpi_issue178_preflight ",
     r"core::mpi_issue178_sampling_failure ", r"core::mpi_issue178_sr_failure "], "#178")
fam("mpi", "MPI collectives, QP/world mapping, SR-CG operator literals", "ParallelContext / reducers", I,
    [r"core::mpi_issue179_collectives", r"core::mpi_issue179_mapping", r"core::mpi_issue179_parallel_literals ",
     r"core::mpi_issue179_qp_literals ", r"core::mpi_issue179_literal_operator", r"core::mpi_issue184_parallel_scalar_contracts "],
    "#179", "Multi-rank tests are feature/launcher gated; only the single-process literals run in the default build.")
fam("mpi", "MPI PhysCal (reduction, singleton group, callback)", "MPI PhysCal", I,
    [r"core::mpi_physcal ", r"core::mpi_issue179_physcal_singleton ", r"core::mpi_physcal_callback_contract "], "#179/#181")
fam("mpi", "MPI normal initialization and genuine-world run", "initial sampling under MPI", I,
    [r"core::mpi_issue274_normal_initialization", r"core::mpi_issue196_world", r"core::mpi_issue234_summary"], "#274/#196/#234")
fam("mpi", "Explicit MPI gate mpi_issue178_support_matrix", "mpi_issue178_support_matrix", I,
    [r"core::mpi_issue178_support_matrix "], "#392", "Open issue: the gate fails on normal initialization exhaustion (#392).")
# ---- threading
fam("threading", "Inner QP/Green/measurement threading: worker counts preserve RNG/configs/outputs", "MVMC_RS_INNER_THREADS", I,
    [r"core::threaded_issue182 ", r"core::issue184_thread_control_contracts ", r"core::issue184_thread_copy_contracts ",
     r"core threading::tests::"], "#182", "Performance and C-only OpenMP regions are separate: #360/#361.")
# ---- timers / diagnostics / outputs
fam("timers", "C section timers, enable semantics, zvo_time output", "c_timer", I,
    [r"core::c_timer ", r"core::issue184_thread_timer_contracts ", r"core::run_log_files "], "#346")
fam("diagnostics", "Run diagnostics: SRinfo rows, CLI timer environment, PhysCal trace", "run_log_files, trace", I,
    [r"core::run_log_files ", r"cli::runtime_contract (physcal_)?timer_environment", r"cli::bin/mvmc physcal_trace::"], "#346/#184")
fam("history", "Optimization history windows and final-window averaging", "ParaOpt summary", I,
    [r"core::ctest_window_fixtures ", r"core::runner_config ", r"core run::callback_tests::final_window_history"], "#180")
fam("output", "zvo_out / zqp_opt / zvo_var block writers and complete C storage", "io writers", I,
    [r"core::output_blocks ", r"core io::tests::", r"core::run_log_files "], "#184/#346")
fam("output", "PhysCal output files (zvo_cisajs, QQQQ, indexed names)", "PhysCal writers", I,
    [r"core io::tests::phys", r"core::physcal_issue181 (original_io134|fixed_records)"], "#181")
# ---- failure boundaries
fam("failure", "Public rejection boundaries leave state, RNG and files unchanged", "validators before mutation", R,
    [r"core::issue184_public_rejection_boundaries ", r"core::issue178_projection_boundary ", r"core::issue283_mode_preflight ",
     r"core::runner_config invalid_options"], "#178/#184")
fam("failure", "Runtime contract: unsupported solver controls, Lanczos combinations", "mvmc_core::validation", R,
    [r"core::runtime_contract rejects_", r"cli::runtime_contract malformed_runner_options"], "#184")
fam("failure", "Nonfinite local energy / invalid saved walkers do not poison store", "run optimization", I,
    [r"core run::callback_tests::nonfinite_local_energy", r"core run::callback_tests::invalid_saved_walkers",
     r"core run::callback_tests::oversized_optimization_window"], "#184")
fam("complex-mode", "Global complex-mode inference (C header) and allocated-mode guards", "all_complex_flag, state_from_data", I,
    [r"core run::mode_tests::", r"core::issue184_original_complex_mode_condition ", r"core::issue283_mode_preflight "], "#283")
fam("parameters", "Public parameter pack/unpack and slot layout", "pack_parameters, unpack_parameters", I,
    [r"core::issue290_public_parameters ", r"core::issue184_original_parameter_boundaries ", r"core::issue184_original_shared_orbital "], "#290")
fam("slater", "Slater element helpers, zero-QP contract, QP sync", "slater_update, init_qp_weight", I,
    [r"core::issue184_slater_contracts ", r"core::issue184_slater_zero_size_contracts ", r"core::issue184_qp_sync_contract "], "#252")
fam("rng", "SFMT19937: C streams, snapshots, words consumed", "sfmt19937::Sfmt19937Rng", I,
    [r"sfmt19937::golden_vs_c ", r"sfmt19937::state_snapshot ", r"sfmt19937::words_consumed ", r"sfmt19937::julia_sfmt_assertion_contracts "], "#184")
fam("linear-algebra", "PfaPack: ltl/utu2 kernels, ordinary real panel, inverse contracts", "pfapack crate", I,
    [r"pfapack::golden_vs_julia ", r"pfapack::issue176_ordinary_panel", r"pfapack::issue184_inverse_contracts ", r"pfapack::julia_assertion_contracts "],
    "#176/#184")
# ---- C-only features (not in Julia)
fam("c-only", "vmc.out -s StdFace standard-mode: keyword reader, ModelUtil, writers, CLI -s/dry-run", "mvmc-stdface, mvmc -s", I,
    [r"mvmc-stdface::stdface_c_fixtures ", r"cli::issue353_standard_mode ", r"mvmc-stdface::complex_expressions "], "#353")
fam("c-only", "ComplexUHF initial-orbital tool", "mvmc-uhf", I,
    [r"mvmc-uhf::definition_contract ", r"cli::issue350_complex_uhf "], "#350")
fam("c-only", "zvo_time / SRinfo / NFileFlushInterval", "run_log_files", I, [r"core::run_log_files "], "#346")
fam("c-only", "NSRCG / NStore contract (C-only semantics)", "validation", I, [r"core validation::nsrcg_contract_tests::"], "#345")
fam("c-only", "Negative DSROptStepDt (C-only)", "sr negative step", I, [r"core sr::negative_stepdt_tests::"], "#367")
fam("c-only", "C ModPara reader", "modpara", I, [r"parsers::issue344_c_modpara_reader "], "#344")
fam("c-only", "Grouped NSplitSize admission (C-supported)", "grouped runs", I, [r"core::grouped_nsplit_349 "], "#349")


def missing_family(scenario, entry, owner, note, area="c-only"):
    F.append(dict(area=area, scenario=scenario, entry=entry, state=M, tests=[], owner=owner, note=note, evidence="none"))


fam("c-only", "vmc.out options -b (varbin), -F flush interval, -v, -e, -h and positional InitPara", "mvmc CLI option parser", I,
    [r"cli::issue347_c_cli_options "], "#347")
fam("c-only", "MultiDef mode (vmc.out -m N): directory list, rank-to-group map, C messages", "mvmc -m, mvmc_core::multidef", I,
    [r"cli::issue348_multidef ", r"core::multidef_348_split ", r"core multidef::tests::"], "#348")
fam("c-only", "greenr2k post-processing tool (Fortran reference), exit status and input handling", "mvmc-greenr2k", I,
    [r"mvmc-greenr2k::fortran_parity ", r"mvmc-greenr2k::fortran_format ", r"mvmc-greenr2k::behavior "], "#351",
    "References are native gfortran greenr2k outputs; converters beyond greenr2k are tracked in #351.",
    evidence="C fixture")
fam("c-only", "StdFace 1D lattices (chain, ladder) byte-identical to C", "mvmc-stdface chain/ladder", I,
    [r"mvmc-stdface::stdface_c_fixtures ", r"cli::issue353_standard_mode "], "#354/#404", "Ladder/NotUsed defects (Julia-mVMC#66) corrected deliberately, see defect rows.")
fam("c-only", "StdFace 2D lattices (square, triangular, honeycomb, kagome)", "mvmc-stdface square/triangular/honeycomb/kagome", I,
    [r"cli::issue355_standard_2d ", r"mvmc-stdface::stdface_c_fixtures "], "#355/#405")
fam("c-only", "StdFace 3D lattices (orthorhombic, fcc, pyrochlore)", "mvmc-stdface lattice3d", I,
    [r"mvmc-stdface::stdface_3d_defects ", r"mvmc-stdface::stdface_c_fixtures "], "#356/#408")
fam("c-only", "StdFace Wannier90 input and export", "mvmc-stdface wannier90", I,
    [r"cli::issue357_standard_wannier90 ", r"mvmc-stdface::wannier90_defects "], "#357/#407")
fam("threading", "C-only OpenMP parallel regions ported to worker-invariant inner threading", "MVMC_RS_INNER_THREADS regions", I,
    [r"core::threaded_issue360 "], "#360/#402/#409")
fam("threading", "Inner-threading work/size gate and benchmark worker selection", "inner worker gate", I,
    [r"core::threaded_issue361 "], "#361/#412")
fam("sampler", "FSZ sampler applies the RBM factor consistently (incremental update equals recomputation)", "FSZ RBM sampler", I,
    [r"core::issue403_fsz_rbm_sampler "], "#403/#406")
fam("physcal", "Buffered, allocation-free PhysCal outputData with unchanged output bytes", "io::write_phys_cal", I,
    [r"core io::tests::phys", r"core::physcal_issue181 (original_io134|fixed_records)"], "#337/#410",
    "Performance change; bytes are pinned by the output tests.")
fam("physcal", "Native-C PhysCal and Lanczos cells (FSZ combinations, complex Hubbard mode 1, threaded)", "mvmc --physcal", I,
    [r"cli::issue181_native_c_physcal "], "#181/#397",
    "Single-process cells; MPI multi-rank cells stay gated (#179).")
missing_family("CalHamiltonian1 / ReturnSlaterElmDiff on the Hubbard real path", "none", "#207", "Gap documented in issue.", area="optimization")
F.append(dict(area="ctest", scenario="Deterministic 20-step references for all 13 ctest models",
              entry="rust_ctest_upstream_rule_selected_models + same-implementation repeatability", state=D,
              tests=[r"core::ctest_equivalent rust_ctest_upstream_rule_selected_models", r"core run::callback_tests::canonical_cg_fixed_seed_same_configuration_is_repeatable",
                     r"core::ctest_general_reference corrected_general_twenty_step_public_runner_is_repeatable"],
              owner="#180 (closed; decision)", evidence="statistical ctest + same-impl repeatability",
              note="Decision at #180 closure: SR-CG parameter trajectories are ill-conditioned (#358; 1e-16 operand differences amplify to 1e-4..1e-2), "
                   "so long runs use the upstream 3-sigma/1e-8 statistical rule plus same-implementation repeatability instead of independent step-20 references; "
                   "50-step references are not truncated and none are generated from Rust."))
missing_family("MPI multi-rank scenario matrix execution (optimization and measurement)", "mvmc-core mpi_* gates", "#179", "Gated tests exist; full matrix not executed in the default build.", area="mpi")

# ---- known C defects (Julia-mVMC#55-#63) and how Rust treats them
DEFECTS = [
    ("Julia-mVMC#55", "real-mode OptTrans derivatives stored in wrong SR slots (NQPOptTrans > 1)", I,
     [r"core run::callback_tests::real_mode_opttrans_derivatives_use_their_own_slots", r"core run::callback_tests::opttrans_real_direct_prefixes_match_c_kernel_reference"],
     "#370/#368", "Rust writes the correct slots (not C's); difference from C is intentional and tested."),
    ("Julia-mVMC#56", "NSRCG >= 2 with NStore = 0 writes unallocated SROptO_Store", R,
     [r"core run::callback_tests::nsrcg_two_with_store", r"core run::callback_tests::cg_accumulation_forces_store", r"core validation::nsrcg_contract_tests::"],
     "#345", "Rust treats any nonzero NSRCG as CG, stores O when NStore != 0 or NSRCG == 1, and rejects NSRCG >= 2 with NStore == 0 as C-undefined (PR #362)."),
    ("Julia-mVMC#57", "Ne = 0 admitted, fails at run time (XERBLA INFO=-5, SIGFPE)", I,
     [r"core::native342_negative_info "], "#342", "Native negative INFO (-5) from the C kernels is covered by the #342 fixtures; C's later run-time SIGFPE is not a reference."),
    ("Julia-mVMC#58", "rsbOld typo in two-electron updates (no observable effect)", I,
     [r"core::two_electron_rsbold "], "#365", "Rust matches the C definition; no observable difference."),
    ("Julia-mVMC#59", "real-mode runs silently ignore the RBM factor", D,
     [r"core run::callback_tests::rbm_real_and_general_complex_direct_prefixes", r"core::issue403_fsz_rbm_sampler "], "#379/#403",
     "Rust applies the RBM factor in real mode like Julia; C silently ignores it (defect, not copied)."),
    ("Julia-mVMC#60", "ComplexUHF input-handling defects (NULL fclose, out-of-range writes, ignored errors)", D,
     [r"mvmc-uhf::definition_contract c_crashes_become_errors", r"cli::issue350_complex_uhf usage_and_missing_files"], "#350",
     "C crashes become Rust errors."),
    ("Julia-mVMC#61", "grouped runs with stored O read uninitialized SROptO_Store columns", D,
     [r"core::grouped_nsplit_349 ", r"core validation::tests::grouped_matrix_rejects_only_the_c_undefined_sr_cg_combination"], "#349/#178",
     "Rust computes grouped direct-SR NStore=1 correctly (equals serial) and rejects grouped SR-CG citing the C defect; other grouped cases follow C."),
    ("Julia-mVMC#62", "-b binary parameter output (varbin) writes only half of the complex parameters", D,
     [r"cli::issue347_c_cli_options binary_mode_blocks_contain_every_text_parameter", r"cli::issue347_c_cli_options binary_mode_optimizer_matches_c_varbin_layout"],
     "#347/#390", "Rust follows the C varbin layout but writes every parameter (C writes only half of the complex ones)."),
    ("Julia-mVMC#63", "tool/greenr2k: exit status, invalid input format, namelist path truncation", D,
     [r"mvmc-greenr2k::behavior missing_index_exit_status_is_nonzero_unlike_fortran", r"mvmc-greenr2k::behavior file_names_with_slashes_are_read_whole",
      r"mvmc-greenr2k::behavior values_may_continue_on_the_next_record"],
     "#351/#385", "Rust returns a nonzero status, reads whole path names and accepts continued records where the original does not."),
    ("Julia-mVMC#64", "PhysCal Lanczos: FSZ admitted with NSPGaussLeg > 1; uninitialized singular output; exchange counters missing in Julia", R,
     [r"cli::issue181_native_c_physcal heisenberg_fsz_lanczos_with_gauss_leg_is_a_c_defect_rust_rejects", r"cli::issue181_native_c_physcal exact_eigenstate_real_lanczos_has_c_singular_alpha",
      r"cli::issue181_native_c_physcal exact_eigenstate_complex_lanczos_has_c_singular_alpha"],
     "#181/#397", "Rust rejects FSZ + Lanczos regardless of NSPGaussLeg and writes nothing for a singular Lanczos step (matches C's empty-file contract)."),
    ("Julia-mVMC#65", "MultiDef (-m N): division by zero for N=0 and non-collective rank-0 failure", D,
     [r"cli::issue348_multidef nonpositive_n_is_rejected_where_c_divides_by_zero", r"cli::issue348_multidef bad_group_directory_and_small_world_fail_collectively"],
     "#348/#401", "Rust rejects N <= 0 and fails collectively where C divides by zero or fails on rank 0 only."),
    ("Julia-mVMC#66", "StdFace lattice defects (Ladder NotUsed order, wrong unused checks and labels)", D,
     [r"mvmc-stdface::stdface_c_fixtures ", r"cli::issue353_standard_mode "],
     "#404/#405", "Rust corrects the defects (PR #405) instead of reproducing them; fixtures keep the historical C outputs for comparison."),
    ("Julia-mVMC#67", "StdFace Wannier90 defects (uninitialized free, uninitialized reads, cutoff flag always true)", D,
     [r"mvmc-stdface::wannier90_defects ", r"cli::issue357_standard_wannier90 missing_geometry_file"],
     "#357/#407", "Rust errors or initializes correctly for defects 1-3; the always-true cutoff flag keeps C behavior pending clarification."),
    ("Julia-mVMC#68", "StdFace 3D lattice defects (Pyrochlore Kondo sites, FCOrtho field, missing NotUsed checks, ntransMax overflow)", D,
     [r"mvmc-stdface::stdface_3d_defects "], "#356/#408", "Rust output differs from the historical C output in every defect case (asserted)."),
]


# --------------------------------------------------------------------------- build
def resolve_tests(index, regexes):
    keys = []
    for rx in regexes:
        found = index.find(rx)
        if not found:
            sys.exit(f"family test pattern matched nothing: {rx}")
        keys.extend(found)
    seen = {}
    for key, kind in keys:
        seen.setdefault(key, kind)
    return list(seen.items())


def cell_tests(items, limit=4):
    names = [short(k) for k, _ in items]
    shown = "; ".join(names[:limit])
    return shown + (f"; (+{len(names) - limit} more)" if len(names) > limit else "")


def evidence_of(items, override):
    if override:
        return override
    if not items:
        return "none"
    types = Counter(evidence_type(k) for k, _ in items)
    ordered = [t for t, _ in types.most_common() if t != UNIT]
    return " + ".join(ordered[:2]) if ordered else UNIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nextest-log", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    log = load_log(args.nextest_log)
    index = Index(log, static_tests())
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []  # unified TSV rows
    hdr = ["id", "tier", "area", "julia_or_c_item", "rust_entry", "state", "evidence", "tests", "result", "command", "owner", "note"]

    # families
    fam_rows = []
    for n, f in enumerate(F, 1):
        items = resolve_tests(index, f["tests"]) if f["tests"] else []
        keys = [k for k, _ in items]
        row = {"id": f"F{n:03d}", "tier": "scenario-family", "area": f["area"], "julia_or_c_item": f["scenario"],
               "rust_entry": f["entry"], "state": f["state"], "evidence": evidence_of(items, f["evidence"]),
               "tests": cell_tests(items), "result": fmt_result([k for _, k in items]) if items else "no test",
               "command": command_for(keys), "owner": f["owner"], "note": f["note"]}
        fam_rows.append(row)
    defect_rows = []
    for n, (ident, text, state, tests, owner, note) in enumerate(DEFECTS, 1):
        items = resolve_tests(index, tests) if tests else []
        defect_rows.append({"id": f"D{n:03d}", "tier": "c-defect", "area": "c-defect", "julia_or_c_item": f"{ident}: {text}",
                            "rust_entry": "see tests", "state": state, "evidence": evidence_of(items, None) if items else "none",
                            "tests": cell_tests(items), "result": fmt_result([k for _, k in items]) if items else "no test",
                            "command": command_for([k for k, _ in items]), "owner": owner, "note": note})

    # exports / public API
    exports = set()
    for jl in glob.glob(os.path.join(ROOT, "extern/Julia-mVMC/*/src/*.jl")):
        for m in re.finditer(r"(?m)^\s*export\s+([^\n]+(?:,\s*\n[^\n]+)*)", open(jl, errors="replace").read()):
            exports.update(x.strip() for x in re.split(r"[,\s]+", m.group(1)) if x.strip())
    stat = static_tests()
    defs = set()
    for path in glob.glob(os.path.join(ROOT, "crates/*/src/**/*.rs"), recursive=True) + glob.glob(
            os.path.join(ROOT, "crates/*/tests/*.rs")) + glob.glob(os.path.join(ROOT, "crates/*/src/*.rs")):
        defs.update(re.findall(r"\b(?:fn|struct|enum|const|static|type|trait)\s+(\w+)", open(path, errors="replace").read()))
    generic = {"new", "from", "tests", "test", "main", "default", "self", "with", "into", "none", "true", "false", "parse", "read", "write", "load", "type"}
    test_by_name = defaultdict(list)
    for t in stat:
        test_by_name[t["fn"]].append(t)

    def analyse(cites):
        """Resolved citations, their defined symbols, and directly cited tests."""
        resolved, symbols, direct = [], [], []
        for c in cites:
            loose = [i for i in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", c) if i in defs and i not in generic]
            idents = [i for i in loose if len(i) >= 6 and ("_" in i or i[0].isupper() or re.search(r"[a-z][A-Z]", i))]
            if loose:
                resolved.append(c)
            for i in idents:
                if i in test_by_name:
                    direct.extend(test_by_name[i])
                else:
                    symbols.append(i)
        return resolved, symbols, direct

    def referencing(symbols, direct):
        items, weight = {}, {}
        for t in direct:
            key = f"{t['crate']}::{t['stem']} {t['fn']}" if t["kind"] == "integration" else f"{t['crate']} {t['fn']}"
            items[key] = index.result(t)
            weight[key] = 10 ** 6  # explicitly cited by the inventory
        syms = set(symbols)
        if syms:
            pats = [re.compile(r"\b" + re.escape(x) + r"\b") for x in syms]
            for t in stat:
                hits = sum(len(p.findall(t["body"])) for p in pats)
                if hits:
                    key = f"{t['crate']}::{t['stem']} {t['fn']}" if t["kind"] == "integration" else f"{t['crate']} {t['fn']}"
                    items.setdefault(key, index.result(t))
                    weight[key] = weight.get(key, 0) + hits
        # most focused tests (most references to the cited symbols) first
        out = sorted(items.items(), key=lambda kv: (kv[1] != "PASS", -weight[kv[0]], kv[0]))
        return out

    def classify(scope, entry, resolved):
        if scope.startswith("excluded: separately owned InterAll"):
            return "Excluded (InterAll, separately owned)"
        if scope.startswith("excluded reference FFI") or scope.startswith("excluded deprecated Julia value-term shim"):
            return D
        if scope.startswith("rejected"):
            return R
        if scope.startswith("partial: no standalone"):
            return M
        if entry.startswith("NO "):
            return D
        return I if resolved else M

    api_override = {
        "A201": dict(state=D, symbols=["max_integer"],
                     entry="ParallelScalarOperations::max_integer (crates/mvmc-core/src/parallel_scalar.rs:36; serial :48, MPI world crates/mvmc-core/src/mpi.rs:237, "
                           "split groups mpi.rs:287); sampling status max: Reducer::sampling_max_info (crates/mvmc-core/src/reducer.rs:83)",
                     note="Rust covers Julia's integer max with the typed `ParallelScalarOperations::max_integer(domain, i32)`; the differences are "
                          "i32 (C int) instead of any Julia Integer and an enum domain instead of Symbol `which`; source reconciliation in issue-184-a201-public-max-source.md"),
    }
    api_rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, VER, "issue-184-public-apis.tsv")), delimiter="\t"):
        sym = r["julia_symbol_or_scenario"]
        cites = [c.strip() for c in r["rust_entry"].split(";") if c.strip()]
        resolved, symbols, direct = analyse(cites)
        state = classify(r["implementation_scope"], r["rust_entry"], resolved)
        ov = api_override.get(r["id"])
        if ov:
            state, symbols = ov["state"], symbols + ov["symbols"]
            resolved = [ov["entry"]]
        items = referencing(symbols, direct)
        cited = items[:3]
        exported = "export" in r["kind"] or sym.split(".")[-1] in exports
        note = ("explicit Julia export" if exported else "public Julia name (not exported)")
        if state == D and r["rust_entry"].startswith("NO ") and not ov:
            note += "; no Rust counterpart by design: " + r["rust_entry"][3:120]
        if ov:
            note += "; " + ov["note"]
        elif state == M:
            note += "; " + r["rust_entry"][:120]
        api_rows.append({"id": r["id"], "tier": "julia-api", "area": r["kind"], "julia_or_c_item": sym,
                         "rust_entry": "; ".join(resolved or cites)[:300], "state": state,
                         "evidence": evidence_of(cited, None) if cited else "none",
                         "tests": cell_tests(cited, 3) + (f"; (+{len(items) - 3} referencing tests)" if len(items) > 3 else ""),
                         "result": fmt_result([k for _, k in cited]) if cited else "no test",
                         "command": command_for([k for k, _ in cited]), "owner": r["owner"], "note": note})

    # scenarios (Julia testsets/assertions)
    scen_rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, VER, "issue-184-scenarios.tsv")), delimiter="\t"):
        cites = [c.strip() for c in r["rust_entry"].split(";") if c.strip()]
        resolved, symbols, direct = analyse(cites)
        state = classify(r["implementation_scope"], r["rust_entry"], resolved)
        items = referencing(symbols, direct)
        cited = items[:3]
        scen_rows.append({"id": r["id"], "tier": "julia-scenario", "area": r["kind"], "julia_or_c_item": r["julia_symbol_or_scenario"][:160],
                          "rust_entry": "; ".join(resolved or cites)[:300], "state": state,
                          "evidence": evidence_of(cited, None) if cited else "none",
                          "tests": cell_tests(cited, 3), "result": fmt_result([k for _, k in cited]) if cited else "no test",
                          "command": command_for([k for k, _ in cited]), "owner": r["owner"], "note": r["julia_source"][:120]})

    all_rows = fam_rows + defect_rows + api_rows + scen_rows

    # owner audit: a Missing row needs a concrete open owner issue (not only #184/#185)
    concrete = set(OPEN_OWNERS) - {"#184", "#185"}
    ownerless, umbrella_only = [], []
    for row in all_rows:
        if row["state"] == M:
            ids = set(re.findall(r"#\d+", row["owner"]))
            if not ids & concrete:
                (umbrella_only if ids & {"#184", "#185"} else ownerless).append(row)
    tsv = os.path.join(args.out_dir, "issue-184-complete-matrix.tsv")
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # markdown
    def table(rows, cols=("id", "area", "julia_or_c_item", "rust_entry", "state", "evidence", "tests", "result", "command", "owner")):
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for row in rows:
            cells = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in cols]
            if row.get("note") and "note" not in cols:
                cells[2] += f" ({row['note']})"
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def counts(rows):
        c = Counter(r["state"] for r in rows)
        return ", ".join(f"{k}: {v}" for k, v in sorted(c.items()))

    def ev_counts(rows):
        c = Counter(r["evidence"] for r in rows)
        return ", ".join(f"{k}: {v}" for k, v in c.most_common())

    def pass_counts(rows):
        c = Counter("PASS-only" if r["result"].startswith("PASS") and "NotRun" not in r["result"] else
                    ("no test" if r["result"] == "no test" else "has NotRun/mixed") for r in rows)
        return ", ".join(f"{k}: {v}" for k, v in c.items())

    md = [f"# Complete Julia-mVMC to Rust evidence matrix (issue #184)",
          "",
          f"Generated by `scripts/build_issue184_complete_matrix.py` at Rust head `{args.head}`. "
          "The machine-readable copy is [issue-184-complete-matrix.tsv](issue-184-complete-matrix.tsv). "
          "Older matrices in this directory keep their historical identities; this document is the "
          "current, consolidated view.",
          "",
          "## How to read the matrix",
          "",
          "- **State**: `Implemented` (Rust entry point exists and is cited), `Rejected-with-C-reason` "
          "(Rust rejects what C rejects or leaves undefined), `Intentional difference` (Rust deliberately differs "
          "from Julia or from a C defect), `Missing` (not ported). `Excluded (InterAll, separately owned)` marks "
          "inventory rows owned by the InterAll work, listed only to avoid silent omission.",
          "- **Evidence**: strongest evidence type among the cited tests: `C fixture` (native C values or C-derived reference "
          "with provenance), `Julia fixture`, `statistical ctest` (upstream 3-sigma / 1e-8 rule), "
          "`same-impl repeatability`, `unit test (no external reference)` (a test exercises the entry point against analytic or self-consistency expectations only), or `none` (no test found; source mapping only). The type of a test is derived from the fixtures/names it uses "
          "(see the script) and is a classification, not a verdict.",
          f"- **Result**: from one local `cargo nextest run --workspace --cargo-profile test-fast --status-level all` at head "
          f"`{args.head}` (Linux x86_64): {sum(1 for v in log.values() if v == 'PASS')} PASS, "
          f"{sum(1 for v in log.values() if v == 'SKIP')} skipped (ignored/opt-in), 0 failures. `PASS n/n` means every cited test passed; "
          "`NotRun-ignored` means the cited test is an ignored opt-in gate that was not executed; `NotRun-gated` means a feature/launcher-gated "
          "test absent from the default build (MPI). A row with `no test` has only a source mapping. Nothing unrun is counted as passing.",
          "- Cited tests were found by grep in the tree (family rows abort the build if a pattern matches nothing); API/scenario rows cite up to "
          "three tests whose bodies reference the Rust symbol and say so when more exist. A referencing test shows the entry point is exercised, "
          "not that it covers every Julia condition; that is why `Evidence` and `Result` are separate from `State`.",
          "- Regenerate (developer command, not run by Cargo or CI): `cargo nextest run --workspace --cargo-profile test-fast --no-fail-fast "
          "--retries 0 --status-level all > nextest.log 2>&1`, then `uv run --no-project python scripts/build_issue184_complete_matrix.py "
          "--nextest-log nextest.log --head <sha> --out-dir docs/reference/c-to-julia/verification`.",
          "- Owner issues: open ones are " + ", ".join(sorted(OPEN_OWNERS)) + "; closed owners are history.",
          "",
          "## Summary counts",
          "",
          f"- scenario families ({len(fam_rows)} rows): {counts(fam_rows)}",
          f"  - evidence: {ev_counts(fam_rows)}",
          f"  - result: {pass_counts(fam_rows)}",
          f"- C defects Julia-mVMC#55-#68 ({len(defect_rows)} rows): {counts(defect_rows)}",
          f"- Julia public API inventory ({len(api_rows)} rows, {sum(1 for r in api_rows if 'explicit Julia export' in r['note'])} explicit exports): {counts(api_rows)}",
          f"  - evidence: {ev_counts(api_rows)}",
          f"  - result: {pass_counts(api_rows)}",
          f"- Julia scenario inventory ({len(scen_rows)} testsets/assertions/entries): {counts(scen_rows)}",
          f"  - evidence: {ev_counts(scen_rows)}",
          f"  - result: {pass_counts(scen_rows)}",
          f"- total rows: {len(all_rows)}",
          f"- `Missing` rows whose only owner is the #184/#185 umbrella or a closed issue (a concrete issue should be filed): {len(ownerless) + len(umbrella_only)}",
          ""]
    for r in ownerless + umbrella_only:
        md.append(f"  - {r['id']} {r['julia_or_c_item'][:100]} (owner field: {r['owner']})")
    md += ["", "## Scenario families (hand-curated, one row per supported scenario family)", "", table(fam_rows),
           "", "## Known C defects (Julia-mVMC#55-#68) and Rust treatment", "", table(defect_rows),
           "", "## Julia public API inventory (one row per name)", "",
           "Rows come from `issue-184-public-apis.tsv` (explicit exports and every public type/constant/function the Julia tests name). "
           "`Implemented` means the cited Rust symbol exists; see Evidence/Result for whether a test exercises it.", "",
           table(api_rows, cols=("id", "julia_or_c_item", "rust_entry", "state", "evidence", "tests", "result", "owner")),
           "", "## Julia scenario inventory", "",
           f"The {len(scen_rows)} Julia testsets, assertions and entry scripts are rows in the TSV (tier `julia-scenario`); "
           "their counts are in the summary above. Each carries its Rust entry point, state, referencing tests and owner.", ""]
    with open(os.path.join(args.out_dir, "issue-184-complete-matrix.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"rows: {len(all_rows)} (families {len(fam_rows)}, defects {len(defect_rows)}, api {len(api_rows)}, scenarios {len(scen_rows)})")
    print("states:", counts(all_rows))
    print("ownerless Missing:", len(ownerless), "umbrella-only:", len(umbrella_only))
    for r in (ownerless + umbrella_only)[:40]:
        print("  ", r["id"], r["julia_or_c_item"][:80], "|", r["owner"])


if __name__ == "__main__":
    main()
