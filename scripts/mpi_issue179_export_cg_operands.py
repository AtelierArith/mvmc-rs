"""Export ONLY independent published Julia operands for an optional C MPI replay."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_mpi_issue179 import read as read_state
from mpi_issue179_cg_diagnostics import read as read_cg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(reference, inputs, output, mode):
    prefix, dimension, imaginary = (3, 10, 0) if mode == "real" else (2, 20, 1)
    settings = {}
    for line in (inputs / "modpara.def").read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            settings[fields[0]] = fields[1]
    required = {"NSplitSize": 1, "NSRCG": 1, "NStore": 0, "NVMCSample": 3,
                "RndSeed": 1, "DSROptStaDel": 1e-5, "DSROptStepDt": .01}
    if any(float(settings.get(k, "nan")) != v for k, v in required.items()):
        raise ValueError("not the declared fixed-operand cell/settings")
    if float(settings.get("DSROptCGTol", "1e-10")) != 1e-10 or int(settings.get("NSROptCGMaxIter", "0")) != 0:
        raise ValueError("unexpected actual CG tolerance/iteration setting")
    operands = []
    sources = {}
    for rank in range(4):
        cg_file, state_file = reference / f"cg-rank-{rank}.txt", reference / f"rank-{rank}.txt"
        provenance = reference / f"reference-provenance-{rank}.txt"
        if not any(line.startswith("b11d75d9b2baaef31abc59c110c09fedc86a7705b1a3c2ce2bb6c75b8b8a17b3  ")
                   and line.endswith("/stochastic_opt.jl") for line in provenance.read_text().splitlines()):
            raise ValueError("missing independently loaded published62b CG source provenance")
        cg, state = read_cg(cg_file), read_state(state_file)
        if state.get("d:group") != [str(rank), "0", "1"] or state.get("d:steps") != [str(prefix)] or state.get("d:status") != ["0"]:
            raise ValueError("incorrect rank/width/prefix/actual status")
        events = [i for i in range(cg["d:events"][0]) if cg[f"d:event-{i:06d}-kind"] == [0]]
        if len(events) != prefix:
            raise ValueError("missing actual prepared solve operands")
        for path in (cg_file, state_file, provenance):
            sources[str(path)] = digest(path)
        operands.append((cg, state, events))
    output.mkdir(parents=True, exist_ok=False)
    exported = {}
    for step in range(prefix):
        directory = output / f"step-{step}"
        directory.mkdir()
        common = None
        for rank, (cg, state, events) in enumerate(operands):
            stem = f"event-{events[step]:06d}-"
            mapping = cg["d:" + stem + "mapping"]
            arrays = [cg["n:" + stem + key] for key in ("mean", "diagonal", "gradient")]
            if len(mapping) != dimension or any(len(a) != dimension for a in arrays):
                raise ValueError("unexpected active dimension")
            metadata = (mapping, arrays)
            if common is not None and metadata != common:
                raise ValueError("inconsistent actual global prepared operands")
            common = metadata
            weight_key = "n:reduced-energy" if step == 0 else f"n:step-{step}-reduced-energy"
            if float(state[weight_key][0]) != 12:
                raise ValueError("unexpected actual global weight")
            real = cg["n:" + stem + "real-samples"]
            imag = cg.get("n:" + stem + "imag-samples", [])
            if len(real) != dimension * 3 or (imaginary and len(imag) != len(real)) or (not imaginary and imag):
                raise ValueError("incomplete or wrong declared sample planes")
            # Preserve acquired Float64 precision; these are operands, not
            # expected results or bitwise numerical acceptance assertions.
            lines = [f"{dimension} 3 {imaginary} {rank} 4 1 0 {float(12).hex()} {float(1e-5).hex()} {float(1e-10).hex()}"]
            for values in (arrays[0], arrays[1], real, imag, arrays[2]):
                if values:
                    lines.append(" ".join(value.hex() for value in values))
            path = directory / f"operand-rank-{rank}.txt"
            path.write_text("\n".join(lines) + "\n")
            exported[str(path)] = digest(path)
    sources[str(inputs / "modpara.def")] = digest(inputs / "modpara.def")
    (output / "provenance.json").write_text(json.dumps({
        "scope": "INDEPENDENT_JULIA_OPERANDS_ONLY_NOT_FULL_C_SAMPLING",
        "published_head": "62b0f97f076fb55c71c3ab0caa041a9adff94e04",
        "source_hashes": sources, "exported_operand_hashes": exported,
        "exporter_sha256": digest(Path(__file__)),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("real", "cmp"), required=True)
    args = parser.parse_args()
    export(args.reference, args.inputs, args.output, args.mode)
