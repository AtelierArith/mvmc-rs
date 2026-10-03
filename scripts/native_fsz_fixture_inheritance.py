#!/usr/bin/env -S uv run --no-project
"""Explicitly verify/record exact historical fixture reuse; optional developer tool."""
import argparse
import hashlib
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mappings(native, manifest):
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        local, historical, expected = line.split()
        for name in (local, historical):
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
        yield local, historical, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Record byte-identical historical duplicates")
    parser.add_argument("--prune", action="store_true", help="Remove only manifest-listed hash-verified duplicates")
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent.parent / "tests/fixtures"
    native = fixtures / "c_kernel_order/native_fsz"
    manifest = native / "inheritance.tsv"
    if args.write:
        entries = {local: (historical, expected) for local, historical, expected
                   in mappings(native, manifest)} if manifest.exists() else {}
        for path in sorted(native.rglob("*")):
            if not path.is_file() or path == manifest:
                continue
            local = path.relative_to(native).as_posix()
            # Mac originals have matching solver-relative paths. Other
            # platforms provide explicit verified mappings independently.
            if not local.startswith(("sr_direct/", "sr_cg/")):
                continue
            old = fixtures / local
            if old.is_file() and path.read_bytes() == old.read_bytes():
                entries[local] = (local, digest(old))
            else:
                entries.pop(local, None)
        text = "# native-relative\thistorical-relative\tsha256\n"
        text += "".join(f"{local}\t{old}\t{sha}\n" for local, (old, sha) in sorted(entries.items()))
        manifest.write_text(text)
    count, size = 0, 0
    for local, old, expected in mappings(native, manifest):
        historical = fixtures / old
        assert digest(historical) == expected, f"Historical fixture changed: {old}"
        path = native / local
        if path.exists():
            assert digest(path) == expected, f"Native expectation differs: {local}"
            size += path.stat().st_size
            if args.prune:
                path.unlink()
        count += 1
    print(f"Verified {count} explicit historical SHA256 mappings; {'pruned' if args.prune else 'prunable'} bytes={size}")


if __name__ == "__main__":
    main()
