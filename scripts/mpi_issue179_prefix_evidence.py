"""Bounded same-implementation prefix/group evidence; not reference accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_mpi_issue179 import read, validate_trace
from mpi_issue179_repeat_evidence import validate as validate_repeat


def record_backend(binary, root):
    # Reuse the frozen existing observer; do not copy a dirty owner183 module.
    from run_optional_gates_183 import backend
    root = Path(root)
    observed = backend(Path(binary), root / "runtime-backend-ldd.txt")
    observed["scope"] = "separate metadata process loading the binary-linked library; not in-rank observation"
    (root / "runtime-backend.json").write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")


def record_inputs(inventory, output):
    from run_optional_gates_183 import validate_namelist
    inventory = Path(inventory).resolve()
    files = {path.resolve() for path in inventory.rglob("*") if path.is_file()}
    namelists = list(inventory.glob("*/inputs/namelist.def"))
    if len(namelists) != 12:
        raise ValueError("input closure requires exactly twelve namelists")
    for namelist in namelists:
        files.update(path.resolve() for path in validate_namelist(namelist))
    lines = []
    for path in sorted(files):
        if any(character in str(path) for character in "\n\r\\"):
            raise ValueError("unsupported input manifest filename")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
    Path(output).write_text("".join(lines))


def validate_build(proof, binary, manifest, commit, snapshot):
    proof = json.loads(Path(proof).read_text())
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if proof.get("source_commit") != commit or proof.get("build_exit") != 0:
        raise ValueError("wrong committed build association")
    if proof.get("binary_sha256") != digest(binary) or proof.get("source_manifest_sha256") != digest(manifest):
        raise ValueError("build binary/source manifest association mismatch")
    scripts = ("mpi_issue179_prefix_evidence.py", "test_mpi_issue179_prefix_evidence.py",
               "verify_mpi_issue179_prefixes.sh")
    expected = {"scripts/" + name: digest(Path(snapshot) / "scripts" / name) for name in scripts}
    if proof.get("owner_overlay_sha256") != expected:
        raise ValueError("build owner overlay differs from approved three scripts")
    required = ("build-command.txt", "build.log", "build.exit", "build.json",
                "compiler.txt", "git-rev-parse.txt", "baseline.sha256")
    records = proof.get("build_records_sha256", {})
    if set(records) != set(required):
        raise ValueError("incomplete build provenance records")
    for name in required:
        if records[name] != digest(Path(snapshot) / name):
            raise ValueError(f"build record changed: {name}")
    if (Path(snapshot) / "git-rev-parse.txt").read_text().strip() != commit:
        raise ValueError("captured actual git revision mismatch")
    if (Path(snapshot) / "build.exit").read_text().strip() != "0":
        raise ValueError("actual build exit is nonzero")


def scalar(rows, key):
    values = rows.get("d:" + key, [])
    if len(values) != 1:
        raise ValueError(f"missing/malformed scalar {key}")
    return int(values[0])


def words(rows, key):
    values = [int(x) for x in rows.get("d:" + key, [])]
    if len(values) != 624 or any(not 0 <= x < 2**32 for x in values):
        raise ValueError(f"missing/malformed 624 words {key}")
    return values


def trace(rows):
    return [rows[f"d:trace-{i:06}"] for i in range(scalar(rows, "trace-events"))]


def validate_rank(rows, rank, ranks, width, steps):
    if scalar(rows, "steps") != steps or scalar(rows, "status") != 0:
        raise ValueError("unsuccessful/wrong prefix")
    if scalar(rows, "requested-width") != width:
        raise ValueError("wrong requested width")
    group = rank // width
    expected = [group, rank % width, min(width, ranks - group * width)]
    if width == 1:
        expected = [rank, 0, 1]
    if [int(x) for x in rows.get("d:group", [])] != expected:
        raise ValueError("incorrect group membership")
    seed = scalar(rows, "seed")
    if not 0 <= seed < 2**32:
        raise ValueError("invalid primitive seed")
    for raw, cursor in (("before-init-raw-rng", "before-init-rng-cursor"),
                        ("initial-raw-rng", "initial-rng-cursor"),
                        ("raw-rng", "rng-cursor")):
        words(rows, raw)
        if not 0 <= scalar(rows, cursor) <= 624:
            raise ValueError("invalid RNG cursor")
    for key in ("initial-rng", "rng"):
        words(rows, key)
    initial, total, sampled = (scalar(rows, key) for key in
                               ("initial-draw-count", "total-draw-count", "sampling-draw-count"))
    if initial < 0 or sampled <= 0 or total != initial + sampled:
        raise ValueError("inconsistent primitive draw counts")
    for count, cursor in ((initial, "initial-rng-cursor"), (total, "rng-cursor")):
        if scalar(rows, cursor) != ((count - 1) % 624 + 1 if count else 624):
            raise ValueError("cursor/draw count mismatch")
    if scalar(rows, "before-init-rng-cursor") != 624:
        raise ValueError("invalid seeded cursor")
    if scalar(rows, "configured-samples") <= 0 or scalar(rows, "chain-samples") != scalar(rows, "configured-samples"):
        raise ValueError("invalid full-chain sample count")
    for key in ("ele_idx", "ele_cfg", "ele_num", "counter"):
        if not rows.get("d:" + key):
            raise ValueError(f"missing configuration {key}")
    if len(rows["d:counter"]) != 10 or len(rows.get("d:acceptance-events", [])) != 2:
        raise ValueError("missing counter/acceptance metadata")
    validate_trace(rows, rank)
    events = trace(rows)
    if not any(1 <= int(event[0]) <= 6 for event in events):
        raise ValueError("missing actual proposals")
    if not any(event[0] == "7" for event in events):
        raise ValueError("missing actual decisions")
    for event in events:
        if 1 <= int(event[0]) <= 6 and int(event[-1]) not in (0, 1):
            raise ValueError("invalid candidate reject flag")
    return events


def validate(cell, ranks, width):
    if ranks not in (2, 4) or width not in (1, 2):
        raise ValueError("unsupported bounded axes")
    cell = Path(cell)
    histories = {}
    initial_states = {}
    for steps in (1, 2, 3):
        root = cell / f"prefix{steps}" / "w1"
        exits = [int((root / f"launch{repeat}.exit").read_text()) for repeat in (1, 2)]
        validate_repeat(root / "repeat1", root / "repeat2", ranks, 0, *exits)
        for repeat in (1, 2):
            groups = {}
            bases = set()
            for rank in range(ranks):
                rows = read(root / f"repeat{repeat}" / f"rank-{rank}.txt")
                events = validate_rank(rows, rank, ranks, width, steps)
                initial = {key: rows[key] for key in (
                    "d:seed", "d:before-init-raw-rng", "d:before-init-rng-cursor",
                    "d:initial-raw-rng", "d:initial-rng-cursor", "d:initial-draw-count", "d:initial-rng")}
                if (repeat, rank) in initial_states and initial_states[repeat, rank] != initial:
                    raise ValueError(f"initial prefix state mismatch prefix={steps} rank={rank}")
                initial_states[repeat, rank] = initial
                bases.add(scalar(rows, "seed") - rank // width)
                group = rank // width
                # Measurement-local arrays/counters/scratch are deliberately excluded.
                signature = (events, {key: rows[key] for key in (
                    "d:initial-raw-rng", "d:initial-rng-cursor", "d:initial-draw-count",
                    "d:initial-rng", "d:raw-rng", "d:rng-cursor", "d:total-draw-count", "d:rng")})
                if group in groups and groups[group] != signature:
                    raise ValueError(f"group sampling mismatch prefix={steps} rank={rank}")
                groups[group] = signature
                previous = histories.get((repeat, rank))
                if previous is not None and events[:len(previous)] != previous:
                    raise ValueError(f"common prefix trace mismatch prefix={steps} rank={rank}")
                histories[repeat, rank] = events
            if len(bases) != 1:
                raise ValueError("base seed/offset mismatch")
    return f"PREFIX_GROUP_EVIDENCE ranks={ranks} width={width} prefixes=1,2,3 repeats=2 workers=1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--backend-binary", type=Path)
    parser.add_argument("--build-proof", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--input-closure", type=Path)
    args = parser.parse_args()
    try:
        if args.input_closure is not None:
            record_inputs(args.cell, args.input_closure)
        elif args.build_proof is not None:
            validate_build(args.build_proof, args.binary, args.manifest, args.commit, args.cell)
        elif args.backend_binary is not None:
            record_backend(args.backend_binary, args.cell)
        else:
            print(validate(args.cell, args.ranks, args.width))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"PREFIX_FAILURE: {error}\n")
