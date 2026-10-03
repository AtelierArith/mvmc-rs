"""Early-stage shape/identity binding, not independent numerical verification.

Reviewed dimensions must originate from input/source review, never these records.
Seeded has no VMC arrays; initialized owns no VMC state and borrows two arrays.
"""

FIELDS = {"run_uuid", "invocation", "label", "stage", "launch_seed", "raw624",
          "index", "draw_count", "next624", "npara", "nqp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def validate(observations, reviewed, run_uuid, invocation, discrete, numeric):
    """Require exact stage labels/settings and matching stdout operands.

    RNG matches artifact to stdout only. Independent raw-state expectations must
    be supplied and checked by a future reviewed producer; this is ShapeBound.
    """
    require(type(reviewed) is dict and bool(reviewed), "missing reviewed stages")
    require(type(observations) is list, "observation list")
    require(integer(invocation, 0, 2**64 - 1), "invocation integer")
    require(type(run_uuid) is str and bool(run_uuid), "run UUID")
    seen = set()
    expected_discrete = {}
    expected_numeric = {}
    for value in observations:
        require(type(value) is dict and set(value) == FIELDS, "exact stage fields")
        label = value["label"]
        require(type(label) is str and label in reviewed and label not in seen,
                "missing/duplicate/unknown stage label")
        seen.add(label)
        rule = reviewed[label]
        require(type(rule) is dict and set(rule) == {"stage", "launch_seed", "npara", "nqp"},
                "reviewed stage contract")
        require(rule["stage"] in ("seeded", "initialized"), "reviewed stage")
        require(value["stage"] == rule["stage"], "wrong stage")
        require(value["run_uuid"] == run_uuid, "wrong run")
        require(type(value["invocation"]) is int and value["invocation"] == invocation,
                "wrong invocation")
        for key in ("launch_seed", "npara", "nqp"):
            require(integer(rule[key], 0, 2**32 - 1), "reviewed integer")
            require(type(value[key]) is int and value[key] == rule[key], "wrong dimension/seed")
        if value["stage"] == "seeded":
            require(value["npara"] == value["nqp"] == 0, "seeded has no arrays")
        else:
            require(value["npara"] > 0 and value["nqp"] > 0, "zero active array")
            expected_numeric[label + "-parameters"] = 2 * value["npara"]
            expected_numeric[label + "-qpweights"] = 2 * value["nqp"]
        require(integer(value["index"], 0, 624), "SFMT index")
        require(integer(value["draw_count"], 0, 2**128 - 1), "primitive count")
        for key in ("raw624", "next624"):
            words = value[key]
            require(type(words) is list and len(words) == 624
                    and all(integer(word, 0, 2**32 - 1) for word in words), "primitive words")
        expected_discrete.update({label + "-raw624": value["raw624"],
                                  label + "-rng-index": value["index"],
                                  label + "-draw-count": value["draw_count"],
                                  label + "-rng": value["next624"]})
    require(seen == set(reviewed), "missing stage observation")
    require(set(discrete) == set(expected_discrete), "exact discrete labels")
    for key, expected in expected_discrete.items():
        actual = discrete[key]
        require(type(actual) is type(expected), "discrete type")
        if type(expected) is list:
            require(all(type(word) is int for word in actual), "discrete word type")
        require(actual == expected, "discrete artifact/stdout mismatch")
    require(set(numeric) == set(expected_numeric), "exact numeric labels; no fabricated state")
    import math
    for key, count in expected_numeric.items():
        values = numeric[key]
        require(type(values) is list and len(values) == count
                and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                "active numeric shape/finite operands")
    return {"status": "ShapeBound", "independent_comparisons": 0}
