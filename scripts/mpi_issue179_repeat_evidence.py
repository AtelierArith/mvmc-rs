"""Same-configuration fresh-process repeat validation, not reference accuracy."""
import argparse
from pathlib import Path


def validate(first, second, ranks, cg, first_exit, second_exit):
    if first_exit != 0 or second_exit != 0:
        raise ValueError("nonzero launcher exit cannot pass repeatability")
    if ranks not in (2, 4) or cg not in (0, 1):
        raise ValueError("unsupported repeat configuration")
    names = [f"rank-{rank}.txt" for rank in range(ranks)]
    if cg:
        names += [f"cg-rank-{rank}.txt" for rank in range(ranks)]
    names += ["zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"]
    if cg:
        names.append("zvo_SRinfo.dat")
    for name in names:
        a, b = (Path(root) / name for root in (first, second))
        if not a.is_file() or not b.is_file() or not a.stat().st_size or not b.stat().st_size:
            raise ValueError(f"missing/empty repeat evidence: {name}")
        if a.read_bytes() != b.read_bytes():
            raise ValueError(f"same-configuration repeat mismatch: {name}")
    return f"REPEAT_EVIDENCE ranks={ranks} cg={cg} files={len(names)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--cg", type=int, required=True)
    parser.add_argument("--first-exit", type=int, required=True)
    parser.add_argument("--second-exit", type=int, required=True)
    args = parser.parse_args()
    try:
        print(validate(args.first, args.second, args.ranks, args.cg, args.first_exit, args.second_exit))
    except (OSError, ValueError) as error:
        parser.exit(1, f"REPEAT_FAILURE: {error}\n")
