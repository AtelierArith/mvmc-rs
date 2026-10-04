"""Offline acquisition declarations, never evidence of a current Julia execution."""
import hashlib
import re
import tomllib

MANIFEST = "extern/Julia-mVMC/Manifest-v1.13.toml"
AUTHORITY = "hash-bound offline declarations; NOT current runtime verification"
NAMES = {"metadata.txt", "provenance.txt", "metadata-provenance.txt",
         "source-provenance.txt", "generation-provenance.txt",
         "c-window-provenance.txt", "model-settings.txt",
         "native-reference-environment.txt"}


def selected(path):
    return path.endswith("/" + MANIFEST) or path.rsplit("/", 1)[-1] in NAMES


def row(path, text):
    if not isinstance(text, str):
        raise ValueError("reference declaration text must be UTF-8")
    result = {"path": path, "sha256": hashlib.sha256(text.encode()).hexdigest(),
              "text": text}
    if path.endswith("/" + MANIFEST):
        data = tomllib.loads(text)
        version = data.get("julia_version")
        if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            raise ValueError("Manifest Julia version missing/invalid")
        if version != "1.13.1":
            raise ValueError("checkout Manifest must declare reviewed Julia 1.13.1")
        result.update(kind="checkout_manifest", julia_version=version)
    else:
        # Only explicit machine-style declarations are interpreted. Preserve all
        # raw C/mixed prose; absence is not inferred from the checkout Manifest.
        versions = re.findall(r"^(?:julia|Julia)[=:]\s*([0-9]+\.[0-9]+\.[0-9]+)(?=\s|$)",
                              text, re.MULTILINE)
        result.update(kind="historical_acquisition_declaration",
                      julia_versions=sorted(set(versions)) if versions else "NotRecorded")
    return result


def report(rows):
    return {"schema": 1, "authority": AUTHORITY,
            "julia_runtime": {"status": "NotRun", "version": None, "blas": None},
            "declarations": rows}


def capture(paths):
    return report([row(str(path), path.read_bytes().decode("utf-8"))
                   for path in sorted(set(paths)) if selected(str(path))])


def validate(value, fixture_hashes):
    if not isinstance(value, dict) or set(value) != {"schema", "authority", "julia_runtime", "declarations"}:
        raise ValueError("invalid offline reference report schema")
    if type(value["schema"]) is not int or value["schema"] != 1 or value["authority"] != AUTHORITY:
        raise ValueError("invalid offline reference authority")
    if value["julia_runtime"] != {"status": "NotRun", "version": None, "blas": None}:
        raise ValueError("offline gate cannot claim Julia runtime execution")
    rows = value["declarations"]
    expected_paths = sorted(path for path in fixture_hashes if selected(path))
    if (not isinstance(rows, list) or
            any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in rows) or
            [item["path"] for item in rows] != expected_paths or
            sum(path.endswith("/" + MANIFEST) for path in expected_paths) != 1):
        raise ValueError("missing/duplicate/off-closure reference declarations")
    for item in rows:
        canonical = row(item["path"], item.get("text"))
        if item != canonical or fixture_hashes[item["path"]] != canonical["sha256"]:
            raise ValueError("reference declaration semantics/hash mismatch")
