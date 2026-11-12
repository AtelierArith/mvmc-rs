"""Verify that sharded/retried logs cover every test in the standard CUDA gate."""
import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "profile"
    report = {}
    for expected_file in sorted(root.glob("expected-*.txt")):
        target = expected_file.stem.removeprefix("expected-")
        expected = {line[:-6] for line in expected_file.read_text().splitlines() if line.endswith(": test")}
        passed = set()
        logs = sorted(set(root.glob(f"final-*-{target}.log")) | set(root.glob(f"retry*-{target}.log")))
        for path in logs:
            passed.update(re.findall(r"^test (\S+)(?: - should panic)? \.\.\. ok$", path.read_text(), re.M))
        if target.endswith("-lib"):
            debug = (root / "partial-debug.log").read_text()
            if "test result: ok. 1 passed; 0 failed" in debug:
                passed.add("opt::complex_tests::resident_opt_complex_partial_warp_groups_match_cpu")
                logs.append(root / "partial-debug.log")
        report[target] = {"expected": len(expected), "passed": len(expected & passed),
                          "missing": sorted(expected - passed), "unexpected": sorted(passed - expected),
                          "logs": [p.name for p in logs]}
    assert len(report) == 5, report
    result = {"targets": report, "total": sum(v["expected"] for v in report.values()),
              "all_passed": all(not v["missing"] and not v["unexpected"] for v in report.values())}
    (root / "final-gate-coverage.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    assert result["all_passed"], "failed or unfinished tests remain"


if __name__ == "__main__":
    main()
