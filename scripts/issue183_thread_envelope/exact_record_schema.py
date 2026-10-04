"""Source-only exact label/dimension validator. No execution on import.

Caller must hash-check actual layout artifacts and pass the independently
reviewed source/input closures; producer self-claims are never schema authority.
"""
import math
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


INTEGER_SETTINGS = (
    "npara", "nelec", "nqp", "samples", "onebody", "twobody", "store", "cg",
    "input_seed", "launch_seed", "steps", "window", "groups", "ranks",
)
SUFFIXES = (
    "parameters", "energy", "pf", "inverse", "inverse-real", "pf-real",
    "oo", "oo-real", "ho", "ho-real", "store", "store-real", "onebody", "twobody",
    "iteration-energy", "retained-matrix", "retained-rhs",
)


def settings(value):
    require(type(value) is dict and set(value) in (
        set(INTEGER_SETTINGS) | {"all_complex", "physical"},
        set(INTEGER_SETTINGS) | {"all_complex", "physical", "retained_dimension"}),
            "complete resolved settings required")
    require(all(type(value[k]) is int for k in INTEGER_SETTINGS), "strict integer settings")
    require(type(value["all_complex"]) is bool and type(value["physical"]) is bool,
            "strict resolved mode flags")
    require(all(value[k] > 0 for k in ("nelec", "nqp", "samples", "groups", "ranks")),
            "positive active input dimensions")
    require(all(value[k] >= 0 for k in ("npara", "onebody", "twobody", "input_seed", "launch_seed", "steps", "window")),
            "nonnegative resolved dimensions/settings")
    require(value["launch_seed"] <= 0xffffffff, "actual SFMT seed width")
    if "retained_dimension" in value:
        require(type(value["retained_dimension"]) is int and value["retained_dimension"] > 0,
                "positive strict original retained system dimension")
    require(value["store"] in (0, 1) and value["cg"] in (0, 1), "supported store/CG settings")
    return value


def dimension(kind, layout):
    s = 1 + layout["npara"]
    q = layout["nqp"]
    real = not layout["all_complex"]
    count = {
        "parameters": 2 * layout["npara"], "energy": 10,
        "pf": 2 * q, "inverse": 2 * q * ((2 * layout["nelec"]) ** 2 + 1),
        "inverse-real": q * ((2 * layout["nelec"]) ** 2 + 1) if real else 0,
        "pf-real": q if real else 0,
        "oo": 4 * s * (2 * s + 2), "oo-real": s * (s + 2) if real else 0,
        "ho": 4 * s, "ho-real": s if real else 0,
        "store": 4 * s * layout["samples"],
        "store-real": s * layout["samples"] if real else 0,
        "onebody": 2 * layout["onebody"], "twobody": 2 * layout["twobody"],
        "iteration-energy": 2,
    }
    if kind in ("retained-matrix", "retained-rhs"):
        require("retained_dimension" in layout, "missing original retained SR shape")
        count[kind] = layout["retained_dimension"] ** (2 if kind == "retained-matrix" else 1)
    require(kind in count, "unsupported dimension kind")
    if kind in ("onebody", "twobody"):
        require(layout["physical"], "Green record without PhysicalQuantities")
    return count[kind]


def bound_hashes(expected, closure):
    require(type(expected) is dict and expected, "missing source/input authority")
    require(all(type(p) is str and p and type(h) is str
                and re.fullmatch(r"[0-9a-f]{64}", h) and closure.get(p) == h
                for p, h in expected.items()), "changed/missing reviewed authority")


def validate(reviewed, actual_layouts, records, source, inputs, expected_cases):
    """Validate exact inventory and finite scalar counts, not numerical values.

    actual_layouts must originate from separately hash-checked child artifacts.
    Expected cases/source/inputs must originate from the external review contract.
    No glob, wildcard, inferred zero or producer checks:true field is supported.
    """
    require(type(reviewed) is dict and set(reviewed) == {"schema", "cases"}, "exact schema fields")
    require(type(reviewed["schema"]) is int and reviewed["schema"] == 1, "strict schema version")
    cases = reviewed["cases"]
    require(type(cases) is list and cases, "nonempty exact case schema")
    require(all(type(c) is dict and type(c.get("id")) is str for c in cases), "case identity type")
    require(len(cases) == len(expected_cases) and {c["id"] for c in cases} == set(expected_cases),
            "missing/duplicate/wrong reviewed case")
    require(type(actual_layouts) is dict and set(actual_layouts) == set(expected_cases),
            "missing/extra actual resolved layout")
    labels = {}
    for case in cases:
        require(set(case) == {"id", "settings", "source", "inputs", "records"}, "case schema fields")
        layout = settings(case["settings"])
        observed = settings(actual_layouts[case["id"]])
        require(observed == layout, "actual input/mode/layout/settings changed")
        bound_hashes(case["source"], source)
        bound_hashes(case["inputs"], inputs)
        require(type(case["records"]) is list, "exact record inventory required")
        for record in case["records"]:
            require(type(record) is dict and set(record) == {"label", "kind", "dimension", "zero_reason"},
                    "record schema fields")
            label = record["label"]
            require(type(label) is str and label and not any(c in label for c in "*?[]|\n\r"),
                    "exact safe label required")
            require(label not in labels, "duplicate/shared labels require separate reviewed alias schema")
            require(type(record["dimension"]) is int and record["dimension"] >= 0,
                    "strict scalar dimension")
            require(type(record["kind"]) is str and record["kind"] in SUFFIXES, "known dimension kind")
            count = dimension(record["kind"], layout)
            require(record["dimension"] == count, "dimension inconsistent with reviewed input layout")
            reason = record["zero_reason"]
            if count == 0:
                inactive = layout["all_complex"] and record["kind"] in (
                    "inverse-real", "pf-real", "oo-real", "ho-real", "store-real")
                green = layout["physical"] and record["kind"] in ("onebody", "twobody")
                require((inactive and reason == "inactive-real-buffer")
                        or (green and reason == "zero-green-definition-count"),
                        "zero lacks independently reviewed structural reason")
            else:
                require(reason is None, "active record cannot carry empty permission")
            labels[label] = count
    require(type(records) is dict and set(records) == set(labels), "missing/extra exact numeric record")
    for label, count in labels.items():
        payload = records[label]
        require(type(payload) is str, "numeric payload type")
        if count == 0:
            require(payload == "", "zero-size encoding must be exactly empty, not whitespace")
        else:
            values = payload.split()
            require(len(values) == count and all(math.isfinite(float(v)) for v in values),
                    "active array empty/wrong-size/nonfinite")
    return {"status": "ShapeBound", "numeric_comparisons": 0,
            "zero_size_structural_records": sum(count == 0 for count in labels.values())}
