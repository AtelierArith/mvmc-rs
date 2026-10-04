"""Diagnostic C MPI replay comparison; no solver/trajectory acceptance budget."""
import argparse
import json
import math
from pathlib import Path

from compare_mpi_issue179 import read as read_text
from mpi_issue179_cg_diagnostics import read as read_cg


def read_c(path):
    raw = read_text(path)
    rows = {key: [int(v) if key.startswith("d:") else float(v) for v in values]
            for key, values in raw.items()}
    if any(not math.isfinite(v) for values in rows.values() for v in values):
        raise ValueError("nonfinite actual C observation")
    required = {"d:rank", "d:world", "d:width", "d:dimension", "d:samples", "d:imag",
                "d:iterations", "d:operator-count", "d:check-count", "n:weight", "n:shift",
                "n:tolerance", "n:mean", "n:diagonal", "n:gradient", "n:real-samples",
                "n:solution", "n:residual", "n:direction"}
    if not required <= rows.keys() or any(len(rows[key]) != 1 for key in required if key.startswith("d:")):
        raise ValueError("missing actual C protocol metadata")
    n, samples, imaginary = (rows[key][0] for key in ("d:dimension", "d:samples", "d:imag"))
    if rows["d:world"] != [4] or rows["d:width"] != [1] or samples != 3 or imaginary not in (0, 1) or n != (20 if imaginary else 10):
        raise ValueError("out-of-scope actual C rank/domain/dimension")
    if not 0 <= rows["d:rank"][0] < 4 or not 0 <= rows["d:iterations"][0] <= n:
        raise ValueError("malformed actual C rank/iteration count")
    for key, expected in (("n:weight", 12), ("n:shift", 1e-5), ("n:tolerance", 1e-10)):
        if rows[key] != [expected]:
            raise ValueError("unexpected actual C settings")
    allowed = set(required)
    if imaginary:
        allowed.add("n:imag-samples")
    for field in ("mean", "diagonal", "gradient", "solution", "residual", "direction"):
        if len(rows["n:" + field]) != n:
            raise ValueError("invalid actual C vector shape")
    for field in ("real-samples", "imag-samples")[:1 + imaginary]:
        if len(rows.get("n:" + field, [])) != n * samples:
            raise ValueError("missing actual C sample plane")
    for kind, fields in (("operator", ("root-search", "local-product", "global-product", "corrected-product")),
                         ("check", ("delta", "threshold", "solution", "residual", "direction"))):
        count = rows[f"d:{kind}-count"][0]
        if not 0 <= count <= 2 * n + 1:
            raise ValueError("invalid actual C boundary count")
        for index in range(count):
            for field in fields:
                key = f"n:{kind}-{index:06d}-{field}"
                allowed.add(key)
                length = 1 if field in ("delta", "threshold") else n
                if len(rows.get(key, [])) != length:
                    raise ValueError(f"missing/malformed actual C boundary {key}")
    if rows.keys() != allowed:
        raise ValueError("unknown/missing actual C fields")
    return rows


def solve_events(rows, step):
    prepared = [i for i in range(rows["d:events"][0]) if rows[f"d:event-{i:06d}-kind"] == [0]]
    if not 0 <= step < len(prepared):
        raise ValueError("missing actual prepared solve")
    start = prepared[step]
    end = prepared[step + 1] if step + 1 < len(prepared) else rows["d:events"][0]
    finished = [i for i in range(start, end) if rows[f"d:event-{i:06d}-kind"] == [3]]
    if len(finished) != 1:
        raise ValueError("missing actual finish boundary")
    products = [i for i in range(start, end) if rows[f"d:event-{i:06d}-kind"] == [1]]
    if len(products) % 4 or any(rows[f"d:event-{event:06d}-phase"] != [index % 4] for index, event in enumerate(products)):
        raise ValueError("malformed actual operator phase order")
    return start, finished[0], products


def compare(c, other, step):
    events = [solve_events(rows, step) for rows in other]
    input_delta = 0.0
    for rank in range(4):
        start = events[rank][0]
        for field in ("mean", "diagonal", "gradient", "real-samples", "imag-samples"):
            left = c[rank].get("n:" + field, [])
            right = other[rank].get(f"n:event-{start:06d}-{field}", [])
            if len(left) != len(right):
                raise ValueError("C/reference actual operand shape mismatch")
            input_delta = max(input_delta, max((abs(x-y) for x, y in zip(left, right)), default=0))
    if input_delta:
        return dict(scope="NONCOMMON_PREPARED_OPERANDS_NO_SOLVER_COMPARISON", input_max_delta=input_delta)
    first = None
    count = min(c[0]["d:operator-count"][0], len(events[0][2]) // 4)
    for op in range(count):
        for phase, field in enumerate(("root-search", "local-product", "global-product", "corrected-product")):
            for rank in range(4):
                event = events[rank][2][op * 4 + phase]
                left = c[rank][f"n:operator-{op:06d}-{field}"]
                right = other[rank][f"n:event-{event:06d}-" + ("search" if phase == 0 else "product")]
                for index, (x, y) in enumerate(zip(left, right)):
                    if x != y:
                        first = dict(operator=op, phase=field, rank=rank, index=index,
                                     c=x, other=y, absolute_difference=abs(x-y),
                                     scale=max(map(abs, left + right)))
                        if phase == 2:
                            locals_c = [row[f"n:operator-{op:06d}-local-product"][index] for row in c]
                            locals_o = [other[r][f"n:event-{events[r][2][op*4+1]:06d}-product"][index] for r in range(4)]
                            if locals_c == locals_o:
                                u = 2.0**-53
                                bound = 2 * (3 * u / (1 - 3 * u)) * sum(map(abs, locals_c))
                                first["global_sum_boundary_model"] = dict(
                                    assumptions="four identical finite normal local inputs; three rounded additions per reduction; no overflow/underflow",
                                    formula="2*gamma_3*sum(abs(local_products))", absolute_bound=bound,
                                    within_boundary_model=abs(x-y) <= bound,
                                    scope="ONLY_GLOBAL_SUM_NOT_CG_SOLUTION_OR_TRAJECTORY")
                        break
                if first:
                    break
            if first:
                break
        if first:
            break
    finish = events[0][1]
    results = {field: max(abs(x-y) for x, y in zip(c[0]["n:" + field], other[0][f"n:event-{finish:06d}-{field}"]))
               for field in ("solution", "residual", "direction")}
    return dict(scope="COMMON_OPERAND_DIAGNOSTIC_ONLY_NO_NUMERICAL_ACCEPTANCE",
                first_primitive_difference=first, c_iterations=c[0]["d:iterations"][0],
                other_iterations=other[0][f"d:event-{finish:06d}-iterations"][0],
                final_max_deltas=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c_directory", type=Path)
    parser.add_argument("julia", type=Path)
    parser.add_argument("rust", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    c = [read_c(args.c_directory / f"c-rank-{rank}.txt") for rank in range(4)]
    if any(row["d:rank"] != [rank] for rank, row in enumerate(c)):
        raise ValueError("actual C rank identities mismatch")
    print(json.dumps({label: compare(c, [read_cg(directory / f"cg-rank-{rank}.txt") for rank in range(4)], args.step)
                      for label, directory in (("Julia", args.julia), ("Rust", args.rust))}, sort_keys=True))
