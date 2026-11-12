"""Run disjoint portions of the unchanged CUDA gate on two physical devices."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shard", choices=("A", "B"))
    p.add_argument("--image", default="mvmc731-cuda-toolchain:local")
    p.add_argument("--target-volume", default="mvmc-731-cuda-target")
    p.add_argument("--cache-volume", default="mvmc-733-driver-cache")
    p.add_argument("--remaining-only", action="store_true", help="resume failed/incomplete cases; keep original logs")
    p.add_argument("--device", help="physical GPU override")
    p.add_argument("--tag", help="unique output tag when resuming")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = Path(__file__).resolve().parent / "profile"
    artifacts = {}
    for line in (out / "final-test-artifacts.json").read_text().splitlines():
        d = json.loads(line)
        if (d.get("reason") == "compiler-artifact" and d.get("executable")
                and d.get("profile", {}).get("test")
                and d["target"]["src_path"].startswith("/work/gpu/mvmc-gpu-cuda/")):
            artifacts[d["executable"]] = d["target"]
    assert artifacts
    normal = ("opt::complex_tests::", "opt::physcal::complex_tests::")
    for executable, target in artifacts.items():
        is_lib = "lib" in target["kind"]
        if a.shard == "B" and not is_lib:
            continue
        name = target["name"] + ("-lib" if is_lib else "")
        docker = ["docker", "run", "--rm", "--user", "1000:1000", "--gpus",
                  "device=" + (a.device or ("0" if a.shard == "A" else "1")),
                  "--tmpfs", "/tmp:rw,mode=1777,size=2g",
                  "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
                  "-e", "CUDA_CACHE_PATH=/driver-cache",
                  "-v", f"{a.cache_volume}:/driver-cache",
                  "-v", f"{a.target_volume}:/cuda-target:ro", "-v", f"{repo}:/work:ro",
                  "-w", "/work/gpu/mvmc-gpu-cuda", a.image,
                  "taskset", "-c", "2-4,18-20" if a.shard == "A" else "5-7,21-23", executable]
        if a.shard == "A" or a.remaining_only:
            with (out / f"expected-{name}.txt").open("w") as f:
                subprocess.run(docker + ["--list", "--include-ignored"], stdout=f,
                               stderr=subprocess.STDOUT, check=True)
        flags = ["--include-ignored", "--test-threads=1"]
        if a.remaining_only:
            expected = {line[:-6] for line in (out / f"expected-{name}.txt").read_text().splitlines()
                        if line.endswith(": test")}
            passed = set()
            for pattern in (f"final-*-{name}.log", f"retry*-{name}.log"):
                for log in out.glob(pattern):
                    passed.update(re.findall(r"^test (\S+)(?: - should panic)? \.\.\. ok$",
                                             log.read_text(), re.M))
            if is_lib:
                debug = (out / "partial-debug.log").read_text()
                if "test result: ok. 1 passed; 0 failed" in debug:
                    passed.add("opt::complex_tests::resident_opt_complex_partial_warp_groups_match_cpu")
            missing = sorted(expected - passed - {"device_rng_rejects_a_different_device_context"})
            if is_lib:
                missing = [n for n in missing if
                           ("opt::physcal::fsz_complex_tests::" in n) == (a.shard == "B")]
            if not missing:
                continue
            flags += ["--exact", *missing]
        elif a.shard == "A":
            flags += ["--skip", "device_rng_rejects_a_different_device_context"]
            if is_lib:
                for pattern in normal:
                    flags += ["--skip", pattern]
        else:
            flags += normal
        print(f"{a.shard}: {name}", flush=True)
        tag = a.tag or ("retry" if a.remaining_only else "final") + "-" + a.shard
        destination = out / f"{tag}-{name}.log"
        assert not destination.exists(), "preserve previous validation logs"
        with destination.open("w") as f:
            subprocess.run(docker + flags, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"Shard {a.shard} passed", flush=True)


if __name__ == "__main__":
    main()
