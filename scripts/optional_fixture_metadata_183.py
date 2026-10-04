"""Reviewed fixture reporting contract, NOT a general C/Rust Modpara parser."""
import hashlib
import re

MODELS = ("hubbard_chain_real", "hubbard_chain_lanczos", "spin_chain_lanczos")
FIELDS = ("NVMCSample", "NDataQtySmp", "NVMCWarmUp", "NVMCInterval",
          "RndSeed", "NVMCCalMode", "NLanczosMode")
SAMPLES = (100, 1000, 5000)


def expected(model):
    if model not in MODELS:
        raise ValueError("unknown fixture report model")
    return dict(zip(FIELDS, (SAMPLES[MODELS.index(model)], 1, 10, 1, 1, 1, 2)))


def declared(path):
    raw = path.read_bytes()
    values = {}
    for line in raw.decode("utf-8").splitlines():
        tokens = line.split("#", 1)[0].split()
        canonical = {key.lower(): key for key in FIELDS}
        if not tokens or tokens[0].lower() not in canonical:
            continue
        key = canonical[tokens[0].lower()]
        if key in values or len(tokens) != 2 or not re.fullmatch(r"[+-]?[0-9]+", tokens[1]):
            raise ValueError(f"ambiguous fixture report field: {key}")
        values[key] = int(tokens[1])
    if set(values) != set(FIELDS):
        raise ValueError("missing required fixture report fields")
    return values, hashlib.sha256(raw).hexdigest()


def capture(root):
    rows = []
    for model in MODELS:
        path = root / f"extern/Julia-mVMC/test/integration/reference/{model}/physcal_ref/inputs/modpara.def"
        values, sha = declared(path)
        rows.append({"model": model, "path": str(path), "sha256": sha,
                     "declared": values})
    result = {"schema": 1, "authority": "static fixture declaration; NOT runtime observation",
              "fixtures": rows, "explicit_gate_overrides": {"seed": 1, "modes": ["real", "cmp"]}}
    validate(result, {row["path"]: row["sha256"] for row in rows})
    return result


def validate(value, fixture_hashes):
    if not isinstance(value, dict) or set(value) != {"schema", "authority", "fixtures", "explicit_gate_overrides"}:
        raise ValueError("invalid fixture report schema")
    if type(value["schema"]) is not int or value["schema"] != 1 or value["authority"] != "static fixture declaration; NOT runtime observation":
        raise ValueError("invalid fixture report authority")
    overrides = value["explicit_gate_overrides"]
    if not isinstance(overrides, dict) or overrides != {"seed": 1, "modes": ["real", "cmp"]} or type(overrides.get("seed")) is not int:
        raise ValueError("wrong explicit gate overrides")
    rows = value["fixtures"]
    if not isinstance(rows, list) or len(rows) != len(MODELS):
        raise ValueError("missing/duplicate fixture report models")
    for model, row in zip(MODELS, rows):
        if not isinstance(row, dict) or set(row) != {"model", "path", "sha256", "declared"} or row["model"] != model:
            raise ValueError("wrong fixture report identity")
        if not isinstance(row["path"], str) or not row["path"].endswith(f"/{model}/physcal_ref/inputs/modpara.def"):
            raise ValueError("wrong fixture report path")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or fixture_hashes.get(row["path"]) != row["sha256"]:
            raise ValueError("fixture report hash not bound")
        fields = row["declared"]
        if not isinstance(fields, dict) or set(fields) != set(FIELDS) or any(type(v) is not int for v in fields.values()) or fields != expected(model):
            raise ValueError("fixture declarations mismatch reviewed gate report")
