"""External source-bound execution adapter; no execution on import.

Consumes existing recorder artifacts, never Rust-generated expected values.
Execution binding is not numerical certification. Reviewed inputs are supplied
separately by the reviewer, not trusted from a producer envelope.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import uuid
from exact_record_schema import validate as validate_numeric_schema
import exact_record_schema
import initial_stage_schema

BASE = "2a8e6cd33e7b2840a4935ca40bb7792621f9080a"
GATES = {
    "long20": ("runner_twenty_step_workers_repeat_supported_outcomes_and_physcal", "runners-long20-observed", 6, [(0, 1), (2, 3), (0, 2), (4, 5), (0, 4)]),
    "failure": ("direct_sr_non_spd_boundary_repeats_workers_one_two_four", "direct-sr-failure-boundary", 6, [(0, 1), (2, 3), (0, 2), (4, 5), (0, 4)]),
    "prefix": ("independent_runner_prefixes_match_full_normalized_pre_sr_arrays", "independent-prefixes", 3, [(0, 1), (0, 2)]),
    "physcal": ("independent_physcal_workers_match_saved_rng_and_ordered_outputs", "independent-physcal", 6, [(0, 0), (0, 1), (0, 2), (2, 3), (0, 4), (4, 5)]),
    "transfer": ("transfer_site_executes_actual_jobs_at_threshold", "transfer-site", 3, [(0, 1), (0, 2)]),
    "cg20": ("reviewed_cg_twenty_step_workers_match_and_repeat", "reviewed-cg-long20", 6, [(0, 0), (0, 1), (0, 2), (2, 3), (0, 4), (4, 5)]),
    "real-fsz": ("independent_real_fsz_workers_match_public_pre_sr_and_rng", "independent-real-fsz", 6, [(0, 0), (0, 1), (0, 2), (2, 3), (0, 4), (4, 5)]),
}


def case_ids(gate):
    models = ["heisenberg_chain_real", "heisenberg_chain_cmp", "heisenberg_chain_fsz", "real-fsz", "hubbard_chain_real"]
    prefixes = models[:3] + ["hubbard_chain_real", "general_rbm_cmp_cg"]
    if gate == "long20":
        return {f"long20/{model}/qp{size}/store{store}/cg{cg}" for model in models
                for size in (31, 32, 33) for store, cg in ((0, 0), (1, 0), (0, 1))} | {
                f"long20/{model}/qp{size}/physcal" for model in models for size in (31, 32, 33)} | {
                f"long20/prefix/{model}" for model in prefixes}
    return {
        "failure": {"failure/hubbard_chain_real/qp32/store0/cg0"},
        "prefix": {f"prefix/{m}" for m in prefixes},
        "physcal": {f"physcal/{m}" for m in models[:3] + ["hubbard_chain_real"]},
        "transfer": {f"transfer/hubbard_chain_real/terms{n}" for n in (31, 32, 33)},
        "cg20": {"cg20/general_rbm_cmp/steps20/window20/store0/cg1"},
        "real-fsz": {f"real-fsz/{stage}" for stage in ("seeded", "initialized", "pre-sr", "final")},
    }[gate]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(root, name):
    path = Path(name)
    require(isinstance(name, str) and not path.is_absolute() and path.parts
            and all(p not in (".", "..") for p in path.parts), "unsafe artifact path")
    target = Path(root) / path
    require(target.is_file() and target.resolve().is_relative_to(Path(root).resolve()), "missing artifact")
    require(all(not p.is_symlink() for p in [target, *target.parents]), "symlink artifact")
    return target.read_bytes()


def checked(root, hashes, name):
    require(name in hashes, f"unbound artifact {name}")
    data = read(root, name)
    require(digest(data) == hashes[name], f"tampered artifact {name}")
    return data


def load_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def records(data, empty_labels=()):
    # Exactly the original Rust child parser's D/N prefix rule. No liberal
    # prefix stripping or skipped duplicate records; unsupported encodings fail.
    require(data.endswith(b"\n"), "truncated child stdout")
    result = {}
    for line in data.decode().splitlines():
        if line.startswith(("D|", "N|")):
            fields = line.split("|", 2)
            require(len(fields) == 3 and fields[1] and fields[1] not in result, "duplicate/malformed child record")
            if fields[0] == "N":
                require((fields[2] == "" and fields[1] in empty_labels)
                        or (fields[2].split() and all(math.isfinite(float(v)) for v in fields[2].split())),
                        "empty/nonfinite worker operand; exact zero-size schema required")
            result[fields[1]] = (fields[0], fields[2])
    require(result, "missing child operands")
    return result


def operand(data):
    # Rust BTreeMap<String,(char,String)> Debug syntax for these ASCII records.
    # Inspect the AST BEFORE evaluation: literal_eval(dict) otherwise discards
    # duplicate keys silently. Rust-specific unsupported escapes fail closed.
    require(data.endswith(b"\n"), "truncated comparison operand")
    tree = ast.parse(data.decode().strip(), mode="eval").body
    require(isinstance(tree, ast.Dict), "comparison map required")
    keys = [ast.literal_eval(key) for key in tree.keys]
    require(all(isinstance(k, str) for k in keys) and len(keys) == len(set(keys)), "duplicate comparison operand")
    value = ast.literal_eval(tree)
    require(value and all(isinstance(v, tuple) and len(v) == 2 and v[0] in ("D", "N")
                         and isinstance(v[1], str) for v in value.values()), "comparison schema")
    return value


def metadata(run, index, job, worker, threshold, repeat):
    return (f"run={run}\ninvocation={index}\njob={job}\nworkers={worker}\n"
            f"threshold={threshold}\nrepeat={repeat}\n").encode()


def review_contract(reviewed):
    require(type(reviewed.get("schema")) is int and reviewed["schema"] == 1
            and type(reviewed.get("head")) is str and reviewed["head"] == BASE, "wrong reviewed schema/base")
    run = reviewed["run_uuid"]
    require(type(run) is str, "UUID string required")
    require(str(uuid.UUID(run)) == run, "canonical reviewed UUID")
    require(type(reviewed["gate"]) is str and reviewed["gate"] in GATES, "unsupported selection")
    require(type(reviewed.get("suite")) is str and reviewed["suite"] == "mvmc-core::threaded_issue182", "wrong suite")
    settings = reviewed.get("settings")
    require(type(settings) is dict and type(settings.get("threshold")) is int
            and settings["threshold"] == 32 and type(settings.get("workers")) is list
            and all(type(w) is int for w in settings["workers"])
            and settings["workers"] == [1, 2, 4], "strict threshold/worker plan")
    for field in ("seed", "steps", "window", "ranks", "groups", "store", "cg", "samples", "warmup"):
        if field in settings:
            require(type(settings[field]) is int, f"strict integer setting {field}")
    require(reviewed.get("source") and reviewed.get("fixtures"), "absent source/fixture closure")
    require(reviewed["binary"].get("source_manifest_sha256") == digest(json.dumps(reviewed["source"], sort_keys=True).encode()), "binary/source association")
    require(reviewed.get("adapter_sha256") == digest(Path(__file__).read_bytes()), "unreviewed adapter source")
    require(reviewed.get("schema_adapter_sha256") == digest(Path(exact_record_schema.__file__).read_bytes()),
            "unreviewed exact-schema adapter source")
    require(reviewed.get("emitter_sources") and all(reviewed["source"].get(p) == h
            for p, h in reviewed["emitter_sources"].items()), "missing emitter binding")
    require(reviewed.get("case_map") and reviewed.get("selection_artifact"), "absent case map/selection binding")
    cases = reviewed["case_map"]
    require(type(cases) is list and all(type(c) is dict
            and all(type(c.get(field)) is str and c[field] for field in ("id", "gate", "boundary"))
            and type(c.get("required_record_labels")) is list and c["required_record_labels"]
            and all(type(label) is str and label for label in c["required_record_labels"])
            and type(c.get("required_discrete_values")) is dict
            and all(type(label) is str and label and type(value) is str
                    for label, value in c["required_discrete_values"].items())
            for c in cases), "strict case identity/operand schema")
    require(len(cases) == len(case_ids(reviewed["gate"])) and {c["id"] for c in cases} == case_ids(reviewed["gate"]), "missing/duplicate reviewed case scope")
    require(all(c["gate"] == reviewed["gate"] for c in cases), "wrong reviewed case gate")
    require(type(reviewed.get("timeout_seconds")) is int and 0 < reviewed["timeout_seconds"] <= 2400, "bounded timeout required")
    require(isinstance(reviewed.get("launch_environment"), dict) and all(
        key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "OPENBLAS_CORETYPE", "JULIA_MVMC_ROOT", "MVMC_RS_THREADED_FIXTURE_ROOT",
                "MVMC_RS_THREADED_REAL_FSZ_ROOT") and isinstance(value, str) and value
        for key, value in reviewed["launch_environment"].items()), "unsupported launch environment override")
    require(reviewed.get("reference_metadata") and reviewed.get("backend")
            and reviewed.get("toolchain"), "missing provenance roles/backend/toolchain")
    return GATES[reviewed["gate"]]


def verify_inputs(workspace, reviewed):
    review_contract(reviewed)
    for closure in ("source", "fixtures"):
        for path, expected in reviewed[closure].items():
            require(digest(read(workspace, path)) == expected, f"changed/missing {closure}: {path}")
    binary = Path(reviewed["binary"]["path"])
    require(binary.is_absolute() and binary.is_file() and not binary.is_symlink(), "missing reviewed binary")
    require(digest(binary.read_bytes()) == reviewed["binary"]["sha256"], "wrong binary")
    expected = set(reviewed["source"]) | set(reviewed["fixtures"])
    workspace = Path(workspace)
    if binary.resolve().is_relative_to(workspace.resolve()):
        expected.add(str(binary.relative_to(workspace)))
    actual = set()
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if ".git" in relative.parts:
            continue  # git administration is not compiled input
        require(not path.is_symlink(), "symlink in source closure")
        if path.is_file():
            actual.add(str(relative))
    require(actual == expected, "missing/unexpected full workspace closure")


def verify_selection(data, reviewed):
    identity = GATES[reviewed["gate"]][0]
    require(digest(data) == reviewed["selection_artifact"], "wrong selection artifact")
    selection = load_json(data)
    suites = selection["rust-suites"]
    selected = [(suite, name, test) for suite, values in suites.items()
                for name, test in values["testcases"].items() if test["filter-match"]["status"] == "matches"]
    require(len(selected) == 1 and selected[0][:2] == (reviewed["suite"], identity)
            and selected[0][2]["ignored"] is True, "wrong selected identity/count/ignored status")
    require(suites[reviewed["suite"]]["binary-path"] == reviewed["binary"]["path"], "selection binary mismatch")


def validate(root, workspace, envelope, reviewed):
    identity, job, count, comparisons = review_contract(reviewed)
    verify_inputs(workspace, reviewed)
    require(envelope.get("reviewed_sha256") == digest(json.dumps(reviewed, sort_keys=True).encode()), "wrong reviewed contract")
    require(envelope.get("run_uuid") == reviewed["run_uuid"] and envelope.get("head") == reviewed["head"], "wrong run/head")
    require(envelope.get("parent_exit_code") == 0 and type(envelope["parent_exit_code"]) is int, "parent failed/unfinished")
    require("checks" not in envelope, "forged self-checks")
    hashes = envelope["artifacts"]
    require(load_json(read(root, "envelope.json")) == envelope, "envelope artifact differs from supplied envelope")
    require(hashes and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()), "artifact digest schema")
    actual_files = {str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file()}
    require(actual_files == set(hashes) | {"envelope.json"}, "missing/unexpected package artifact")
    for name in hashes:
        checked(root, hashes, name)
    verify_selection(checked(root, hashes, "selection.json"), reviewed)
    expected_argv = [reviewed["binary"]["path"], "--ignored", "--exact", identity, "--nocapture", "--test-threads=1"]
    parent = load_json(checked(root, hashes, "parent.json"))
    require(type(parent.get("exit_code")) is int and parent["exit_code"] == 0, "strict parent exit status")
    require(parent == {"argv": expected_argv, "run_uuid": reviewed["run_uuid"], "head": reviewed["head"],
                       "binary_sha256": reviewed["binary"]["sha256"], "exit_code": 0,
                       "cleanup": {"timed_out": False, "status": "NotRequired", "errors": [], "races": []},
                       "post_inputs": True}, "wrong parent execution binding")
    require(parent["cleanup"]["timed_out"] is False and parent["post_inputs"] is True,
            "strict completion flags")
    stdout = checked(root, hashes, "parent.stdout").decode()
    require(len(re.findall(r"^test " + re.escape(identity) + r" \.\.\. ok$", stdout, re.M)) == 1,
            "missing exact parent identity PASS")
    require(len(re.findall(r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .*?$", stdout, re.M)) == 1,
            "unfinished/wrong parent summary")
    require("FAILED" not in stdout, "parent failure output")
    cases = reviewed["case_map"]
    require(all(c["gate"] == reviewed["gate"] for c in cases) and len({c["id"] for c in cases}) == len(cases), "duplicate/wrong case inventory")
    outputs = []
    require(type(reviewed.get("numeric_schema")) is dict
            and type(reviewed.get("layout_bindings")) is dict,
            "complete independently reviewed numeric schema/layout bindings required")
    numeric_schema = reviewed["numeric_schema"]
    initial = reviewed.get("initial_stage_bindings", {})
    require(type(initial) is dict, "initial stage bindings")
    initial_ids = set(initial)
    permitted_initial = {"real-fsz/seeded", "real-fsz/initialized"}
    require(not initial_ids or (reviewed["gate"] == "real-fsz" and initial_ids == permitted_initial),
            "exact initial stage cases required")
    if initial:
        require(reviewed.get("initial_stage_adapter_sha256") ==
                digest(Path(initial_stage_schema.__file__).read_bytes()), "unreviewed initial-stage adapter")
        for case_id, binding in initial.items():
            require(type(binding) is dict and set(binding) == {"label", "contract", "source", "inputs"},
                    "initial stage binding fields")
            require(binding["source"] == reviewed["source"] and binding["inputs"] == reviewed["fixtures"],
                    "initial stage source/input closure")
            require(binding["label"] == "real-fsz-" + case_id.split("/")[1]
                    and binding["contract"].get("stage") == case_id.split("/")[1], "exact initial stage identity")
    require(set(reviewed["layout_bindings"]) == {c["id"] for c in cases}, "exact case/layout binding")
    layout_artifacts = set()
    for index in range(count):
        worker = [1, 2, 4][index if count == 3 else index // 2]
        repeat = 1 if count == 3 else index % 2 + 1
        prefix = f"records/child-{index}"
        request = metadata(reviewed["run_uuid"], index, job, worker, 32, repeat)
        require(checked(root, hashes, prefix + ".requested") == request
                and checked(root, hashes, prefix + ".actual-settings") == request, "wrong child settings/repeat/outcome")
        require(checked(root, hashes, prefix + ".terminal") == request + b"status=Some(0)\n", "child failed/unfinished")
        data = checked(root, hashes, prefix + ".events")
        require(data.endswith(b"\n"), "truncated events")
        rows = data.decode().splitlines()
        require(len(rows) == 2 * len(cases), "missing/duplicate case completion")
        seen = set()
        for offset in range(0, len(rows), 2):
            start, complete = rows[offset].split("|"), rows[offset + 1].split("|")
            require(len(start) == 7 and len(complete) == 7, "event schema")
            case = next((c for c in cases if c["id"] == start[3]), None)
            require(case is not None and start[3] not in seen, "unknown/duplicate case")
            seen.add(start[3])
            tail = [case["id"], case["boundary"], str(worker), str(repeat)]
            require(start == [str(offset), reviewed["run_uuid"], "START", *tail]
                    and complete == [str(offset + 1), reviewed["run_uuid"], "COMPLETE", *tail], "wrong event/run/assertion boundary")
        actual_layouts = {}
        stage_observations = []
        for case in cases:
            label = reviewed["layout_bindings"][case["id"]]
            require(type(label) is str and re.fullmatch(r"[A-Za-z0-9_-]+", label), "exact layout label")
            if case["id"] in initial:
                require(label == initial[case["id"]]["label"], "initial layout identity")
                name = prefix + f".initial-stage-{label}.json"
                layout_artifacts.add(name)
                stage_observations.append(load_json(checked(root, hashes, name)))
                continue
            name = prefix + f".layout-{label}.json"
            layout_artifacts.add(name)
            layout = load_json(checked(root, hashes, name))
            require(set(layout) == {"run_uuid", "invocation", "label", "settings"}
                    and type(layout["invocation"]) is int and layout["invocation"] == index
                    and layout["run_uuid"] == reviewed["run_uuid"] and layout["label"] == label,
                    "layout run/invocation/label binding")
            actual_layouts[case["id"]] = layout["settings"]
        empty_labels = {record["label"] for case in numeric_schema.get("cases", [])
                        for record in case.get("records", []) if type(record.get("dimension")) is int
                        and record["dimension"] == 0}
        value = records(checked(root, hashes, prefix + ".stdout"), empty_labels)
        stage_numeric_labels = {binding["label"] + suffix for binding in initial.values()
                                if binding["contract"]["stage"] == "initialized"
                                for suffix in ("-parameters", "-qpweights")}
        if initial:
            stage_discrete_labels = {binding["label"] + suffix for binding in initial.values()
                                     for suffix in ("-raw624", "-rng-index", "-draw-count", "-rng")}
            require(stage_discrete_labels <= value.keys() and stage_numeric_labels <= value.keys(),
                    "missing initial stage operands")
            require(all(value[label][0] == "D" for label in stage_discrete_labels)
                    and all(value[label][0] == "N" for label in stage_numeric_labels), "initial operand kind")
            initial_stage_schema.validate(
                stage_observations, {binding["label"]: binding["contract"] for binding in initial.values()},
                reviewed["run_uuid"], index,
                {label: load_json(value[label][1]) for label in stage_discrete_labels},
                {label: [float(v) for v in value[label][1].split()] for label in stage_numeric_labels})
        validate_numeric_schema(numeric_schema, actual_layouts,
                                {label: payload for label, (kind, payload) in value.items()
                                 if kind == "N" and label not in stage_numeric_labels},
                                reviewed["source"], reviewed["fixtures"], {c["id"] for c in cases} - initial_ids)
        checked(root, hashes, prefix + ".stderr")
        for case in cases:
            require(case["required_record_labels"] and set(case["required_record_labels"]) <= value.keys(), "missing per-case operands")
            require(isinstance(case.get("required_discrete_values"), dict), "reviewed discrete outcome contract required")
            if case["boundary"] == "runtime-outcome":
                require(case["required_discrete_values"], "runtime outcome must have reviewed discrete value")
            for label, expected_value in case["required_discrete_values"].items():
                require(value.get(label) == ("D", expected_value), "wrong case outcome/discrete contract")
        outputs.append(value)
    for index, (left, right) in enumerate(comparisons):
        prefix = f"records/comparison-{index}"
        require(checked(root, hashes, prefix + ".terminal") ==
                f"run={reviewed['run_uuid']}\ncomparison={index}\nboundary=original-worker-compare-returned\n".encode(), "wrong comparison terminal")
        expected = operand(checked(root, hashes, prefix + ".expected"))
        actual = operand(checked(root, hashes, prefix + ".actual"))
        require(expected == outputs[left] and actual == outputs[right], "wrong child/parent operand binding")
        require(expected.keys() == actual.keys(), "worker operand shape")
        for key in expected:
            require(expected[key][0] == actual[key][0], "worker operand type")
            if expected[key][0] == "D":
                require(expected[key] == actual[key], "discrete worker drift")
    expected_names = {"selection.json", "parent.json", "parent.stdout", "parent.stderr"}
    expected_names.update(layout_artifacts)
    for index in range(count):
        expected_names.update(f"records/child-{index}.{suffix}" for suffix in
                              ("requested", "actual-settings", "terminal", "events", "stdout", "stderr"))
    for index in range(len(comparisons)):
        expected_names.update(f"records/comparison-{index}.{suffix}" for suffix in ("terminal", "expected", "actual"))
    require(set(hashes) == expected_names, "unexpected/duplicate invocation or comparison artifacts")
    return {"status": "ExecutionBound", "numerical_verification": "NotVerified",
            "observed_groups": len(cases), "child_invocations": count,
            "parent_comparisons": len(comparisons), "verified_independent_comparisons": 0,
            "unselected_groups": sorted(set().union(*(case_ids(g) for g in GATES)) - {c["id"] for c in cases}),
            "repeat_evidence": "parent-artifact-bound, not independent repetition observation"}
