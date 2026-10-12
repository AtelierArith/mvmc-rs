"""Verify the independent Rust/Julia E2E receipts from the comparison worker.

Run with uv run --no-project python verify_receipts.py /absolute/results.
Julia 1.13.1 is used only to decode its own Serialization receipts, not to
regenerate expectations from Rust. No CUDA context is needed for verification.
"""
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
subprocess.run(
    ["julia", "+1.13.1", "--startup-file=no", "-", str(root)],
    input=r'''
using Serialization
root=ARGS[1]
for rep in 0:5
    r=deserialize(joinpath(root,"julia-output","rep-$rep.bin"))
    open(joinpath(root,"julia-output","rep-$rep.txt"),"w") do io
        for (key,x) in [("energies",r.energies),("parameters",r.parameters),
                        ("cfg",r.cfg),("accepted",r.accepted)]
            println(io,key," ",length(x));foreach(v->println(io,v),vec(x))
        end
    end
end
''',
    text=True,
    check=True,
)


def read(path):
    tokens = iter(path.read_text().split())
    result = {}
    for name in tokens:
        if name == "seconds":
            next(tokens)
            continue
        result[name] = [float(next(tokens)) for _ in range(int(next(tokens)))]
    return result


baseline = {name: read(root / f"{name}-output/rep-0.txt") for name in ("rust", "julia")}
maximum = {"energies_abs": 0.0, "parameters_abs": 0.0,
           "energies_scaled": 0.0, "parameters_scaled": 0.0,
           "rust_repeat_abs": 0.0, "julia_repeat_abs": 0.0}
for rep in range(6):
    a = read(root / f"rust-output/rep-{rep}.txt")
    b = read(root / f"julia-output/rep-{rep}.txt")
    for key, size in (("energies", 20), ("parameters", 772)):
        assert len(a[key]) == len(b[key]) == size
        for x, y in zip(a[key], b[key]):
            assert abs(x - y) <= 1e-10 + 1e-10 * abs(y), (rep, key, x, y)
            maximum[key + "_abs"] = max(maximum[key + "_abs"], abs(x - y))
            maximum[key + "_scaled"] = max(maximum[key + "_scaled"], abs(x - y) / (1 + abs(y)))
        for name, values in (("rust", a), ("julia", b)):
            reference = baseline[name][key]
            assert len(values[key]) == len(reference)
            for x, y in zip(values[key], reference):
                assert abs(x - y) <= 1e-12 + 1e-12 * abs(y), (rep, name, key)
                maximum[name + "_repeat_abs"] = max(maximum[name + "_repeat_abs"], abs(x - y))
    assert b["cfg"] == baseline["julia"]["cfg"]
    assert b["accepted"] == baseline["julia"]["accepted"]

(root / "numerical-verification.json").write_text(json.dumps(maximum, indent=2) + "\n")
(root / "numerical-receipt.json").write_text(json.dumps(
    {name: {key: values[key] for key in ("energies", "parameters")}
     for name, values in baseline.items()}, indent=2) + "\n")
print(json.dumps(maximum, indent=2))
