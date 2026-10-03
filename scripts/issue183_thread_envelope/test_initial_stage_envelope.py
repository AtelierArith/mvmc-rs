"""Synthetic complete-package stage integration; not model observations."""
import json
from pathlib import Path
import unittest

import initial_stage_schema
from test_thread_envelope import Controls
from thread_envelope import BASE, GATES, digest, metadata, validate


class InitialStageEnvelopeControls(unittest.TestCase):
    def setUp(self):
        fixture = Controls("test_valid_execution_bound_not_numeric")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        review = fixture.reviewed
        identity, job, _, pairs = GATES["real-fsz"]
        review["gate"] = "real-fsz"
        review["initial_stage_adapter_sha256"] = digest(Path(initial_stage_schema.__file__).read_bytes())
        review["initial_stage_bindings"] = {}
        review["case_map"] = []
        review["layout_bindings"] = {}
        review["numeric_schema"]["cases"] = []
        stages = ("seeded", "initialized", "pre-sr", "final")
        values = {}
        for stage in stages:
            case = "real-fsz/" + stage
            label = "real-fsz-" + stage
            boundary = "independent-seeded-rng" if stage == "seeded" else "independent-assertions"
            labels = []
            review["layout_bindings"][case] = label
            if stage in ("seeded", "initialized"):
                contract = dict(stage=stage, launch_seed=1, npara=2 if stage == "initialized" else 0,
                                nqp=1 if stage == "initialized" else 0)
                review["initial_stage_bindings"][case] = dict(label=label, contract=contract,
                    source=dict(review["source"]), inputs=dict(review["fixtures"]))
                for suffix, value in (("-raw624", [0]*624), ("-rng", [1]*624),
                                      ("-rng-index", 624), ("-draw-count", 0)):
                    labels.append(label + suffix)
                    values[label + suffix] = ("D", json.dumps(value))
                if stage == "initialized":
                    for suffix, payload in (("-parameters", "0 0 0 0"), ("-qpweights", "1 0")):
                        labels.append(label + suffix)
                        values[label + suffix] = ("N", payload)
            else:
                labels.append(label + "-pf")
                values[label + "-pf"] = ("N", "1 0")
                review["numeric_schema"]["cases"].append(dict(id=case,
                    settings=fixture.layout_settings.copy(), source=dict(review["source"]),
                    inputs=dict(review["fixtures"]), records=[dict(label=label+"-pf", kind="pf",
                    dimension=2, zero_reason=None)]))
            review["case_map"].append(dict(id=case, gate="real-fsz", boundary=boundary,
                required_record_labels=labels, required_discrete_values={}))
        selection = json.loads((fixture.package / "selection.json").read_bytes())
        suite = selection["rust-suites"][review["suite"]]
        suite["testcases"] = {identity: dict(ignored=True, **{"filter-match": {"status": "matches"}})}
        fixture.put("selection.json", json.dumps(selection).encode())
        review["selection_artifact"] = digest((fixture.package / "selection.json").read_bytes())
        parent = json.loads((fixture.package / "parent.json").read_bytes())
        parent["argv"][3] = identity
        fixture.put("parent.json", json.dumps(parent).encode())
        fixture.put("parent.stdout", (f"test {identity} ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\n").encode())
        for index in range(6):
            worker, repeat = (1, 2, 4)[index//2], index%2+1
            prefix = f"records/child-{index}"
            request = metadata(fixture.uuid, index, job, worker, 32, repeat)
            for suffix in ("requested", "actual-settings"):
                fixture.put(prefix+"."+suffix, request)
            fixture.put(prefix+".terminal", request+b"status=Some(0)\n")
            rows = []
            for offset, case in enumerate(review["case_map"]):
                for kind, number in (("START", 2*offset), ("COMPLETE", 2*offset+1)):
                    rows.append(f"{number}|{fixture.uuid}|{kind}|{case['id']}|{case['boundary']}|{worker}|{repeat}\n")
            fixture.put(prefix+".events", "".join(rows).encode())
            (fixture.package / (prefix+".layout-synthetic-exact.json")).unlink()
            for stage in stages:
                label = "real-fsz-"+stage
                if stage in ("seeded", "initialized"):
                    contract = review["initial_stage_bindings"]["real-fsz/"+stage]["contract"]
                    artifact = dict(run_uuid=fixture.uuid, invocation=index, label=label,
                        raw624=[0]*624, next624=[1]*624, index=624, draw_count=0, **contract)
                    fixture.put(prefix+f".initial-stage-{label}.json", json.dumps(artifact).encode())
                else:
                    fixture.put(prefix+f".layout-{label}.json", json.dumps(dict(run_uuid=fixture.uuid,
                        invocation=index, label=label, settings=fixture.layout_settings)).encode())
            fixture.put(prefix+".stdout", "".join(f"{kind}|{label}|{payload}\n"
                for label, (kind, payload) in values.items()).encode())
        for index in range(len(pairs)):
            fixture.put(f"records/comparison-{index}.terminal",
                f"run={fixture.uuid}\ncomparison={index}\nboundary=original-worker-compare-returned\n".encode())
            for suffix in ("expected", "actual"):
                fixture.put(f"records/comparison-{index}.{suffix}", (repr(values)+"\n").encode())
        fixture.refresh()

    def test_complete_binding_remains_not_verified(self):
        f = self.fixture
        result = validate(f.package, f.workspace, f.envelope, f.reviewed)
        self.assertEqual(result["verified_independent_comparisons"], 0)

    def test_missing_stage(self):
        f = self.fixture
        f.reject(lambda: (f.package / "records/child-0.initial-stage-real-fsz-seeded.json").unlink())

    def test_wrong_stage(self):
        f = self.fixture
        f.reject(lambda: f.change("records/child-0.initial-stage-real-fsz-seeded.json",
                                 b'"stage": "seeded"', b'"stage": "initialized"'))

    def test_unknown_extra_stage_artifact(self):
        f = self.fixture
        f.reject(lambda: f.put("records/child-0.initial-stage-extra.json", b"{}\n"))

    def test_unreviewed_stage_module(self):
        f = self.fixture
        f.reject(lambda: f.reviewed.update(initial_stage_adapter_sha256="0"*64))


if __name__ == "__main__":
    unittest.main()
