"""Package independently generated ARM references; never derive values from Rust.

Run with uv run --no-project python scripts/package_macos_arm_references.py.
Verify every generated logical SHA before reusing archived bytes or compressing.
"""

from pathlib import Path
import gzip
import hashlib


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUTPUT = FIXTURES / "macos_arm_julia"


def read(path: Path) -> bytes | None:
    if path.is_file():
        return path.read_bytes()
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return gzip.decompress(compressed.read_bytes())
    return None


def archived(relative: Path) -> bytes | None:
    result = read(FIXTURES / relative)
    if result is not None:
        return result
    native = FIXTURES / "c_kernel_order" / "native_fsz"
    if relative.is_relative_to(Path("c_kernel_order/native_fsz")):
        key = str(relative.relative_to("c_kernel_order/native_fsz"))
        for line in (native / "inheritance.tsv").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, target, digest = line.split("\t")
            if name == key:
                payload = read(FIXTURES / target)
                assert payload is not None
                assert hashlib.sha256(payload).hexdigest() == digest
                return payload
    return None


def unused_auxiliary(relative: Path) -> bool:
    if relative.parts[0] == "sr_direct_fixed" and relative.name == "fixed-input.txt":
        return True  # Replay stores only the independently refactored solution.
    if relative.name not in ("fixed-input.txt", "gram.txt") or "sr_direct" not in relative.parts:
        return False
    case = relative.parts[relative.parts.index("sr_direct") + 1]
    # Runner tests inspect sampled matrices/stores only for RBM and OptTrans.
    # All fixed-input solve tests retain their archived inputs separately.
    return not case.startswith(("rbm_", "opt_"))


for core in sorted(p for p in OUTPUT.iterdir() if p.is_dir()):
    manifest = core / "SHA256-all.tsv"
    assert manifest.is_file(), f"Complete --job=all before packaging {core.name}"
    entries = []
    omitted = []
    original = stored = reused = 0
    supplemental = core / "SHA256-fixed-direct.tsv"
    combined = manifest.read_text() + (supplemental.read_text() if supplemental.exists() else "")
    logical = {}
    for line in combined.splitlines():
        digest, name = line.split("\t")
        assert name not in logical or logical[name] == digest, name
        logical[name] = digest
    combined = "".join(f"{digest}\t{name}\n" for name, digest in sorted(logical.items()))
    for line in combined.splitlines():
        digest, name = line.split("\t")
        relative = Path(name)
        assert all(part not in ("..", ".") for part in relative.parts)
        path = core / relative
        payload = read(path)
        # An already packaged, reused file is resolved through the archive.
        if payload is None:
            payload = archived(relative)
        assert payload is not None, name
        assert hashlib.sha256(payload).hexdigest() == digest, name
        original += len(payload)
        if unused_auxiliary(relative):
            path.unlink(missing_ok=True)
            Path(str(path) + ".gz").unlink(missing_ok=True)
            omitted.append(f"{digest}\t{name}\n")
            continue
        if payload == archived(relative):
            path.unlink(missing_ok=True)
            Path(str(path) + ".gz").unlink(missing_ok=True)
            reused += 1
            location = "archive"
        elif len(payload) >= 4096:
            compressed = gzip.compress(payload, compresslevel=9, mtime=0)
            Path(str(path) + ".gz").write_bytes(compressed)
            path.unlink(missing_ok=True)
            stored += len(compressed)
            location = "gzip"
        else:
            path.write_bytes(payload)
            stored += len(payload)
            location = "plain"
        entries.append(f"{digest}\t{name}\t{location}\n")
    (core / "storage.tsv").write_text("".join(entries))
    manifest.write_text("".join("\t".join(row.rstrip().split("\t")[:2]) + "\n" for row in entries))
    omitted_path = core / "omitted-auxiliary.tsv"
    if omitted:
        omitted_path.write_text((omitted_path.read_text() if omitted_path.exists() else "") + "".join(omitted))
    # The complete manifest supersedes exploratory single-job manifests.
    for path in core.glob("SHA256-*.tsv"):
        if path != manifest:
            path.unlink()
    print(f"{core.name}: {original:,} oracle bytes, {stored:,} stored bytes, {reused} archived files reused")
