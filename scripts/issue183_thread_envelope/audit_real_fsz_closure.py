"""Read-only historical42/input inventory; never promotes fixture authority.

Developer command only. No Julia, C compiler, Cargo or model execution.
"""
import argparse
import hashlib
import json
from pathlib import Path


def required_files():
    result = ["inputs/namelist.def", "provenance.txt", "status.txt",
              "optimization-flags.txt", "initialized/qp_weights.txt",
              "pre-sr/sr_oo.txt", "pre-sr/sr_ho.txt"]
    for stage in ("seeded", "initialized", "pre-sr", "final"):
        result.extend(stage + "/" + name for name in
                      ("draw-count.txt", "next624.txt", "parameters.txt"))
        if stage != "seeded":
            result.extend(stage + "/" + name for name in
                          ("julia-raw-flags.txt", "c-written-mask.txt", "defined-flags.txt"))
    for stage in ("pre-sr", "final"):
        result.extend(stage + "/" + name for name in
                      ("ele_idx.txt", "ele_cfg.txt", "ele_num.txt", "ele_proj_cnt.txt",
                       "ele_spn.txt", "burn_ele_idx.txt", "counter.txt"))
    assert len(result) == len(set(result)) == 42
    return result


def audit(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("regular fixture directory required")
    inventory = {}

    def read(name):
        relative = Path(name)
        if relative.is_absolute() or any(atom in ("..", ".") for atom in relative.parts):
            raise ValueError("unsafe fixture path")
        current = root
        for atom in relative.parts:
            current = current / atom
            if current.is_symlink():
                raise ValueError("symlink fixture")
        if not current.is_file():
            raise ValueError("missing regular fixture: " + name)
        content = current.read_bytes()
        inventory[name] = hashlib.sha256(content).hexdigest()
        return content

    for name in required_files():
        read(name)
    if read("status.txt").decode().strip() != "0" or (root / "UNVERIFIED.txt").exists():
        raise ValueError("historical producer not successful")
    keys = set()
    for line in read("inputs/namelist.def").decode().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "=")):
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] in keys:
            raise ValueError("malformed/duplicate namelist dependency")
        keys.add(fields[0])
        read("inputs/" + fields[1])
    if "ModPara" not in keys or "LocSpin" not in keys:
        raise ValueError("required input metadata absent")
    if (root / "inputs/initial.def").exists():
        raise ValueError("historical explicit absent initial policy violated")
    missing_raw = [stage + "/" + name
                   for stage in ("seeded", "initialized", "pre-sr", "final")
                   for name in ("raw624.txt", "index.txt")
                   if not (root / stage / name).is_file()]
    return {"status": "HistoricalInventoryOnly", "required_consumer_files": 42,
            "input_initial_policy": "Absent", "files_sha256": inventory,
            "missing_raw_evidence": missing_raw,
            "producer_full_loaded_source_closure": "NotVerified",
            "independent_numerical_comparisons": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True, indent=2))
