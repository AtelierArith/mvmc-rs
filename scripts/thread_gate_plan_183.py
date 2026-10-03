#!/usr/bin/env python3
"""Offline, bounded thread selection/report infrastructure (#182/#183).

Never builds or launches gates. Case evidence is a structurally validated report
of retained assertions, not a new numerical oracle or a workflow dispatch proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

SCHEMA = 1
TEST_SOURCE = "crates/mvmc-core/tests/threaded_issue182.rs"
AUDITED_TEST_SHA = "41452d5fe1c6deb42c8868bda4b5df1f06c5570f7a15de2668100f269ee20024"
IDENTITIES = {
    "long20": "runner_twenty_step_workers_repeat_supported_outcomes_and_physcal",
    "failure": "direct_sr_non_spd_boundary_repeats_workers_one_two_four",
    "prefix": "independent_runner_prefixes_match_full_normalized_pre_sr_arrays",
    "physcal": "independent_physcal_workers_match_saved_rng_and_ordered_outputs",
    "transfer": "transfer_site_executes_actual_jobs_at_threshold",
    "cg20": "reviewed_cg_twenty_step_workers_match_and_repeat",
    "real-fsz": "independent_real_fsz_workers_match_public_pre_sr_and_rng",
}
MODELS = ("heisenberg_chain_real", "heisenberg_chain_cmp", "heisenberg_chain_fsz",
          "real-fsz", "hubbard_chain_real")
PREFIX_MODELS = MODELS[:3] + ("hubbard_chain_real", "general_rbm_cmp_cg")
REJECTIONS = {(32, 0): (10, 2), (32, 1): (10, 4), (33, 1): (16, 2)}
STATUSES = {"Pass", "ExplicitSkip", "NotRun", "MissingFixture", "Unsupported", "Failure"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), f"regular artifact required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(value, name):
    require(isinstance(value, dict) and value, f"empty {name} binding")
    for key, digest in value.items():
        require(isinstance(key, str) and key and not Path(key).is_absolute()
                and ".." not in Path(key).parts, f"unsafe {name} path")
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), f"invalid {name} digest")


def cases():
    result = []

    def add(gate, name, kind, checks, refs=(), repeats=1, rejection=None):
        result.append({"id": f"{gate}/{name}", "gate": gate, "planned_kind": kind,
                       "checks": list(checks), "reference_roots": list(refs),
                       "workers": [1, 2, 4], "repeats": repeats,
                       "rejection": rejection})

    invariant = ("exact_rng_configurations", "numerical_worker_comparison")
    for model in MODELS:
        for size in (31, 32, 33):
            for store, cg in ((0, 0), (1, 0), (0, 1)):
                rejection = REJECTIONS.get((size, store)) if model == "hubbard_chain_real" and cg == 0 else None
                checks = invariant + (("original_factor_info", "failed_update_unchanged") if rejection else ())
                add("long20", f"{model}/qp{size}/store{store}/cg{cg}",
                    "expected_rejection" if rejection else "invariance", checks,
                    repeats=2, rejection=list(rejection) if rejection else None)
            add("long20", f"{model}/qp{size}/physcal", "invariance", invariant, repeats=2)
    # runner_matrix(false) also invokes five independent prefixes, but with
    # verify_stage=false: do not count that path as pre-SR OO/HO proof.
    for model in PREFIX_MODELS:
        refs = (f"tests/fixtures/ctest_model_prefixes/{model}/step-1",)
        if model == "general_rbm_cmp_cg":
            refs += ("tests/fixtures/reviewed_cg_62b/canonical_general_rbm",)
        add("long20", f"prefix/{model}", "independent_comparison",
            ("fixture_final_parameters_energy_rng_configurations",) + invariant, refs, repeats=2)
        add("prefix", model, "independent_comparison",
            ("fixture_pre_sr_oo_ho", "fixture_final_parameters_energy_rng_configurations") + invariant, refs)
    add("failure", "hubbard_chain_real/qp32/store0/cg0", "expected_rejection",
        invariant + ("original_factor_info", "failed_update_unchanged"), repeats=2, rejection=[10, 2])
    for model in MODELS[:3] + ("hubbard_chain_real",):
        add("physcal", model, "independent_comparison",
            ("fixture_seeded_sample_rng_configurations", "fixture_two_frames_ordered_outputs") + invariant,
            (f"tests/fixtures/physcal_181/two-samples/{model}",), repeats=2)
    for size in (31, 32, 33):
        add("transfer", f"hubbard_chain_real/terms{size}", "activation",
            invariant + ("actual_term_items", "serial_parallel_worker_ids"))
    add("cg20", "general_rbm_cmp/steps20/window20/store0/cg1", "independent_comparison",
        ("fixture_final_parameters_energy_rng_configurations", "fixture_full_declared_final_pack",
         "actual_entry_jobs") + invariant,
        ("tests/fixtures/reviewed_cg_62b/canonical_general_rbm",), repeats=2)
    for stage in ("seeded", "initialized", "pre-sr", "final"):
        checks = ("fixture_rng_parameters",)
        if stage in ("pre-sr", "final"):
            checks += ("fixture_pre_sr_oo_ho", "fixture_saved_configurations", "actual_serial_qp_items")
        if stage != "seeded":
            checks += ("fixture_defined_flags_written_mask",)
        add("real-fsz", stage, "independent_comparison", checks + invariant,
            ("tests/fixtures/threaded_182_real_fsz",), repeats=2)
    return result


def required_fixture_files(selected):
    """Exact audited Rust preflight inventory, not ANY file under a root."""
    required = {"extern/Julia-mVMC/Manifest-v1.13.toml"}
    prefix_root = "tests/fixtures/ctest_model_prefixes"
    reviewed = "tests/fixtures/reviewed_cg_62b/canonical_general_rbm"
    if set(selected) & {"long20", "failure", "prefix", "transfer", "cg20"}:
        required.add(prefix_root + "/provenance.txt")
        for model in PREFIX_MODELS:
            for name in ("status.txt", "configs.txt", "rng.txt", "parameters.txt", "energy.txt",
                         "sr_oo.txt", "sr_ho.txt", "model-settings.txt", "inputs.sha256"):
                required.add(f"{prefix_root}/{model}/step-1/{name}")
            input_model = "general_rbm_cmp" if model == "general_rbm_cmp_cg" else model
            original = f"extern/Julia-mVMC/test/integration/reference/{input_model}"
            required.add(original + "/inputs/namelist.def")
            if model != "general_rbm_cmp_cg":
                required.update((original + "/physcal_ref/inputs/namelist.def", original + "/physcal_ref/zqp_opt.dat"))
        for step in ([1, 20] if "cg20" in selected else [1]):
            required.update(f"{reviewed}/step-{step}/{name}" for name in ("provenance.txt", "c-window-input.txt"))
            required.update(f"{reviewed}/step-{step}-{name}.txt" for name in ("status", "configs", "rng", "energy", "parameters"))
    if "physcal" in selected:
        names = ["inputs/namelist.def", "zqp_opt.dat", "provenance.txt", "fixed-parameters.txt"]
        for stage in ("seeded", "initialized", "sample-1"):
            names.extend(f"{stage}/{name}.txt" for name in ("next624", "draw-count"))
        names.extend(f"sample-1/{name}.txt" for name in ("ele_idx", "ele_cfg", "ele_num", "ele_proj_cnt", "ele_spn", "counter"))
        for index in ("007", "008"):
            names.extend(f"expected/zvo_{name}_{index}.dat" for name in ("out", "var", "cisajs", "cisajscktalt", "cisajscktaltex"))
        for model in MODELS[:3] + ("hubbard_chain_real",):
            required.update(f"tests/fixtures/physcal_181/two-samples/{model}/{name}" for name in names)
    if "real-fsz" in selected:
        root = "tests/fixtures/threaded_182_real_fsz"
        names = ["inputs/namelist.def", "provenance.txt", "status.txt", "optimization-flags.txt",
                 "initialized/qp_weights.txt", "pre-sr/sr_oo.txt", "pre-sr/sr_ho.txt"]
        for stage in ("seeded", "initialized", "pre-sr", "final"):
            names.extend(f"{stage}/{name}.txt" for name in ("draw-count", "next624", "parameters"))
            if stage != "seeded":
                names.extend(f"{stage}/{name}.txt" for name in ("julia-raw-flags", "c-written-mask", "defined-flags"))
        for stage in ("pre-sr", "final"):
            names.extend(f"{stage}/{name}.txt" for name in ("ele_idx", "ele_cfg", "ele_num", "ele_proj_cnt", "ele_spn", "burn_ele_idx", "counter"))
        required.update(f"{root}/{name}" for name in names)
    return sorted(required)


def select(values):
    require(isinstance(values, list) and values, "explicit nonempty selections required")
    require(len(values) == len(set(values)), "duplicate selection")
    require(all(value in IDENTITIES for value in values), "unknown/unsupported selection")
    return values


def build_plan(selected, head, source, fixtures, input_closure=None, workspace_julia_version="1.13.1"):
    select(selected)
    require(isinstance(head, str) and len(head) == 40
            and all(c in "0123456789abcdef" for c in head), "invalid checkout HEAD")
    manifest(source, "source")
    manifest(fixtures, "fixture")
    require(source.get(TEST_SOURCE) == AUDITED_TEST_SHA, "test source needs new case audit")
    require(isinstance(workspace_julia_version, str)
            and re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", workspace_julia_version),
            "actual nonempty workspace Julia version string required")
    require(set(required_fixture_files(selected)) <= fixtures.keys(), "MissingFixture: full audited inventory required")
    require(isinstance(input_closure, dict), "namelist definition closure required")
    namelists = [path for path in required_fixture_files(selected) if path.endswith("/namelist.def")]
    require(set(namelists) == input_closure.keys(), "missing/unexpected namelist closure")
    for namelist, definitions in input_closure.items():
        manifest(definitions, "input definition")
        require(namelist in definitions, "namelist itself must be hash-bound")
        require(all(fixtures.get(path) == digest for path, digest in definitions.items()),
                "input closure definitions absent/changed in fixture inventory")
    return {"schema": SCHEMA, "head": head, "source": source, "fixtures": fixtures,
            "input_closure": input_closure,
            "reference_workspace": {"manifest_path": "extern/Julia-mVMC/Manifest-v1.13.toml",
                                    "manifest_sha256": fixtures["extern/Julia-mVMC/Manifest-v1.13.toml"],
                                    "julia_version": workspace_julia_version,
                                    "role": "current-workspace-only", "oracle_execution": "none"},
            "selected": selected, "identities": IDENTITIES, "cases": cases(),
            "required_fixture_files": required_fixture_files(selected),
            "selection_commands": [["cargo", "nextest", "list", "--locked", "-p", "mvmc-core",
                                    "--cargo-profile", "test-fast", "--test", "threaded_issue182",
                                    "--run-ignored", "only", "-E", f"test(={IDENTITIES[key]})",
                                    "--message-format", "json"] for key in selected],
            "settings": {"workers": [1, 2, 4], "threshold": 32, "ranks": 1,
                         "long_steps": 20, "long_window": 20, "profile": "test-fast",
                         "features": "default", "oracle_execution": "none"},
            "scope": "seven opt-in identities; existing primary smoke unchanged"}


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA, "plan schema mismatch")
    expected = build_plan(plan["selected"], plan["head"], plan["source"], plan["fixtures"],
                          plan.get("input_closure"), plan.get("reference_workspace", {}).get("julia_version"))
    require(plan == expected, "plan case/settings/identity tampering")


def checked_artifact(root, item):
    manifest({item["path"]: item["sha256"]}, "artifact")
    path = root / item["path"]
    require(path.resolve().is_relative_to(root.resolve()), "artifact escapes evidence root")
    require(sha(path) == item["sha256"], "artifact changed")
    return json.loads(path.read_text())


def report(plan, evidence, root):
    validate_plan(plan)
    require(evidence.get("schema") == SCHEMA, "evidence schema mismatch")
    for key in ("head", "source", "fixtures"):
        require(evidence.get(key) == plan[key], f"{key} execution binding mismatch")
    manifest(evidence.get("binary"), "executed binary")
    for key in ("source", "fixtures", "binary"):
        require(evidence.get(key + "_after") == evidence[key], f"{key} changed during execution")
    require(evidence.get("selection") == [IDENTITIES[key] for key in plan["selected"]],
            "listed selection/count does not match explicit plan")
    runtime = evidence.get("runtime", {})
    require(isinstance(runtime, dict) and runtime.get("profile") == "test-fast"
            and runtime.get("features") == "default" and runtime.get("ranks") == 1,
            "runtime profile/features/ranks metadata missing or unsupported")
    require(isinstance(runtime.get("rust_version"), str) and runtime["rust_version"],
            "actual compiler version required")
    backend = runtime.get("blas", {})
    require(isinstance(backend, dict) and isinstance(backend.get("version"), str)
            and backend["version"] and type(backend.get("threads")) is int and backend["threads"] == 1
            and backend.get("observation") == "runtime-query",
            "actual single-thread BLAS runtime metadata required; linkage alone insufficient")
    manifest(backend.get("libraries"), "BLAS library")
    gates = evidence.get("gates", [])
    require(isinstance(gates, list), "gate records must be a list")
    identities = [gate["identity"] for gate in gates]
    require(len(identities) == len(set(identities)), "duplicate gate identity")
    allowed = {IDENTITIES[key] for key in plan["selected"]}
    require(set(identities) <= allowed, "unselected/unsupported test identity")
    gate_map = {}
    for gate in gates:
        require(gate["status"] in STATUSES, "unknown gate status")
        require(type(gate["invocations"]) is int and gate["invocations"] >= 0, "invalid invocation count")
        require(gate["invocations"] <= 1, "one outer invocation maximum; child runs not counted")
        if gate["status"] == "Pass":
            require(gate["invocations"] == 1 and type(gate.get("exit_status")) is int
                    and gate["exit_status"] == 0,
                    "Pass requires one completed successful gate")
        elif gate["status"] in ("NotRun", "ExplicitSkip"):
            require(gate["invocations"] == 0 and gate.get("exit_status") is None,
                    "unattempted gate cannot have execution result")
        elif gate["invocations"]:
            require(type(gate.get("exit_status")) is int and gate["exit_status"] != 0,
                    "attempted failure requires nonzero terminal")
        else:
            require(gate.get("exit_status") is None, "preflight failure is not an executed gate")
        gate_map[gate["identity"]] = gate
    rows = evidence.get("cases", [])
    require(isinstance(rows, list), "case records must be a list")
    case_map = {}
    expected = {case["id"]: case for case in plan["cases"] if case["gate"] in plan["selected"]}
    for row in rows:
        require(row["id"] in expected and row["id"] not in case_map, "unexpected/duplicate case")
        case = expected[row["id"]]
        gate = gate_map.get(IDENTITIES[case["gate"]])
        require(gate is not None and gate["status"] == "Pass", "case without successful selected gate")
        body = checked_artifact(root, row["artifact"])
        require(body.get("case") == case["id"] and body.get("identity") == IDENTITIES[case["gate"]],
                "artifact case/selection identity mismatch")
        for key in ("head", "source", "fixtures", "binary"):
            require(body.get(key) == evidence[key], f"case {key} binding mismatch")
        require(body.get("workers") == case["workers"]
                and all(type(worker) is int for worker in body["workers"])
                and type(body.get("repeats")) is int and body["repeats"] == case["repeats"],
                "missing worker/repeat coverage")
        require(body.get("checks") == {check: True for check in case["checks"]}
                and all(type(value) is bool for value in body["checks"].values()),
                "missing/unexpected assertion checks")
        require(body.get("rejection") == case["rejection"], "wrong expected rejection step/INFO")
        for reference_root in case["reference_roots"]:
            require(any(path.startswith(reference_root + "/") for path in plan["fixtures"]),
                    "independent comparison reference binding absent")
        metadata = body.get("metadata", {})
        require(isinstance(metadata, dict) and isinstance(metadata.get("model"), str) and metadata["model"]
                and type(metadata.get("seed")) is int and type(metadata.get("steps")) is int
                and metadata["steps"] > 0 and type(metadata.get("ranks")) is int and metadata["ranks"] == 1
                and type(metadata.get("groups")) is int and metadata["groups"] == 1
                and metadata.get("threads") == [1, 2, 4]
                and all(type(worker) is int for worker in metadata["threads"]),
                "case model/seed/steps/ranks/groups/threads metadata required")
        require(metadata.get("oracle_execution") == "none", "historical references are not oracle execution")
        reference = metadata.get("reference", {})
        if case["planned_kind"] != "independent_comparison":
            require(reference == {"status": "NotApplicable"}, "invariance/activation/rejection has no independent fixture oracle")
        else:
            require(isinstance(reference, dict) and reference.get("label") == "historical-fixture"
                    and (reference.get("julia_version") is None
                         or isinstance(reference["julia_version"], str) and reference["julia_version"]),
                    "historical reference Julia version must be labelled, possibly unknown")
            state = reference.get("manifest_status")
            if state in ("Unavailable", "MissingEvidence"):
                require(reference.get("manifest_sha256") is None,
                        "missing historical Manifest cannot borrow current hash")
            else:
                require(state == "Available" and reference.get("manifest_role") == "historical-generation",
                        "historical Manifest needs historical generation identity")
                manifest({"historical/Manifest.toml": reference.get("manifest_sha256")}, "historical Manifest")
                if reference["manifest_sha256"] == plan["reference_workspace"]["manifest_sha256"]:
                    require(reference.get("julia_version") == plan["reference_workspace"]["julia_version"],
                            "current Manifest cannot identify a different historical Julia version")
        require(body.get("reference_workspace") == plan["reference_workspace"],
                "current workspace metadata binding differs; not historical provenance")
        if case["gate"] in ("long20", "cg20") and "/prefix/" not in case["id"]:
            expected_steps = 2 if case["id"].endswith("/physcal") else 20
            require(metadata["steps"] == expected_steps, "wrong long20/PhysCal effective steps")
        case_map[case["id"]] = {"artifact": row["artifact"], "reported_metadata": metadata}
    results = []
    for case in plan["cases"]:
        gate = gate_map.get(IDENTITIES[case["gate"]])
        if case["gate"] not in plan["selected"] or gate is None:
            status = "NotRun"
        elif gate["status"] != "Pass":
            status = gate["status"]
        else:
            status = "AttestedOnly" if case["id"] in case_map else "Incomplete"
        results.append({"id": case["id"], "status": status,
                        "planned_kind": case["planned_kind"],
                        "evidence_kind": "reported_assertion" if status == "AttestedOnly" else "none",
                        "verification": "NotVerified", "settings_verification": "NotVerified",
                        "reported_metadata": case_map.get(case["id"], {}).get("reported_metadata"),
                        "reported_metadata_artifact": case_map.get(case["id"], {}).get("artifact")})
    selected_results = [row for row in results if row["id"].split("/")[0] in plan["selected"]]
    # No approved source-bound case emitter exists. Checks:true, even in a
    # hashed artifact, cannot establish actual execution or independent parity.
    status = "Incomplete"
    for category in ("Failure", "Unsupported", "MissingFixture"):
        if any(row["status"] == category for row in selected_results):
            status = category
            break
    gate_results = []
    for name, identity in IDENTITIES.items():
        record = gate_map.get(identity)
        gate_status = record["status"] if record else "NotRun"
        if gate_status == "Pass":
            gate_status = "Incomplete"
        gate_results.append({"identity": identity, "status": gate_status,
                             "outer_invocations": record["invocations"] if record else 0})
    return {"schema": SCHEMA, "status": status, "gates": gate_results,
            "case_evidence": "attested_only; no approved case emitter/log execution binding",
            "execution_verification": "NotVerified",
            "approved_emitter": None,
            "reference_workspace": plan["reference_workspace"],
            "required_future_bindings": ["reviewed_emitter_source_sha256", "run_uuid",
                                         "actual_nextest_selection_and_terminal_logs",
                                         "gate_binary_and_per_case_log_artifacts"],
            "selected_test_identities": len(plan["selected"]),
            "observed_gate_identities": len(gates),
            "outer_driver_invocations": sum(gate["invocations"] for gate in gates),
            "reported_attested_cases": len(case_map),
            "verified_independent_comparisons": 0,
            "reported_counts_by_kind": {kind: sum(row["status"] == "AttestedOnly" and row["planned_kind"] == kind
                                         for row in results)
                               for kind in ("independent_comparison", "invariance", "expected_rejection", "activation")},
            "cases": results}


def capture(root, selected):
    source = {}
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                                     "crates", "third_party", ".cargo", "Cargo.toml", "Cargo.lock",
                                     "rustfmt.toml", "scripts/thread_gate_plan_183.py"], cwd=root, text=True).splitlines()
    for name in sorted(set(paths)):
        path = root / name
        if path.is_file():
            source[name] = sha(path)
    fixture_roots = set()
    if set(selected) & {"long20", "failure", "prefix", "transfer", "cg20"}:
        # These gates all call require_runner_fixtures(), even the focused
        # failure/transfer gates: preserve the actual current preflight scope.
        fixture_roots.update(("extern/Julia-mVMC/test/integration/reference",
                              "tests/fixtures/ctest_model_prefixes",
                              "tests/fixtures/reviewed_cg_62b/canonical_general_rbm"))
    if "physcal" in selected:
        fixture_roots.add("tests/fixtures/physcal_181/two-samples")
    if "real-fsz" in selected:
        fixture_roots.add("tests/fixtures/threaded_182_real_fsz")
    fixtures = {}
    for name in sorted(fixture_roots):
        path = root / name
        require(path.is_dir() and not path.is_symlink(), f"MissingFixture: {name}")
        for file in sorted(path.rglob("*")):
            if file.is_file():
                fixtures[str(file.relative_to(root))] = sha(file)
    manifest_path = "extern/Julia-mVMC/Manifest-v1.13.toml"
    fixtures[manifest_path] = sha(root / manifest_path)
    input_closure = {}
    for namelist in required_fixture_files(selected):
        if namelist.endswith("/namelist.def"):
            input_closure[namelist] = definition_closure(root, namelist, fixtures)
    if set(selected) & {"long20", "failure", "prefix", "transfer", "cg20"}:
        for model in PREFIX_MODELS:
            input_model = "general_rbm_cmp" if model == "general_rbm_cmp_cg" else model
            namelist = f"extern/Julia-mVMC/test/integration/reference/{input_model}/inputs/namelist.def"
            hash_manifest = f"tests/fixtures/ctest_model_prefixes/{model}/step-1/inputs.sha256"
            bind_declared_input_hashes(root, hash_manifest, namelist, fixtures, input_closure[namelist])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    workspace_version = tomllib.loads((root / manifest_path).read_text())["julia_version"]
    return build_plan(selected, head, source, fixtures, input_closure, workspace_version)


def definition_closure(root, namelist, fixtures):
    """Bind every referenced definition, plus present implicit initial.def.

    This is the flat Expert namelist artifact scanner, not a substitute for
    production parsing/validation. Unsupported path/record forms fail closed.
    """
    path = root / namelist
    require(namelist in fixtures and sha(path) == fixtures[namelist], "MissingFixture: bound namelist required")
    names = {namelist}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        require(len(fields) == 2, "unsupported namelist closure record")
        reference = Path(fields[1])
        require(not reference.is_absolute() and ".." not in reference.parts,
                "unsupported escaping namelist reference")
        names.add(str(path.parent.relative_to(root) / reference))
    initial = path.parent / "initial.def"
    if initial.exists():
        names.add(str(initial.relative_to(root)))
    result = {}
    for name in sorted(names):
        require((root / name).resolve().is_relative_to(root.resolve()), "definition resolves outside checkout")
        require(name in fixtures and sha(root / name) == fixtures[name],
                f"MissingFixture: referenced definition absent/changed: {name}")
        result[name] = fixtures[name]
    return result


def bind_declared_input_hashes(root, hash_manifest, namelist, fixtures, closure):
    """Require the oracle input SHA list to cover actual loaded definitions."""
    path = root / hash_manifest
    require(hash_manifest in fixtures and sha(path) == fixtures[hash_manifest], "bound input SHA manifest required")
    declared = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        require(len(fields) == 2, "malformed oracle input SHA record")
        digest, filename = fields
        relative = Path(filename)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe oracle input SHA filename")
        name = str(Path(namelist).parent / relative)
        require(name not in declared, "duplicate oracle input SHA filename")
        manifest({name: digest}, "oracle input")
        require(fixtures.get(name) == digest and sha(root / name) == digest,
                "oracle input SHA differs from actual source definition")
        declared[name] = digest
    require(closure.items() <= declared.items(), "oracle input SHA omits loaded definition/implicit initial")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--root", type=Path, required=True)
    plan_parser.add_argument("--select", nargs="+", required=True, choices=list(IDENTITIES))
    plan_parser.add_argument("--output", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--plan", type=Path, required=True)
    report_parser.add_argument("--evidence", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = capture(args.root, args.select)
    else:
        result = report(json.loads(args.plan.read_text()), json.loads(args.evidence.read_text()), args.evidence.parent)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if args.command == "plan" or result["status"] == "Pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
