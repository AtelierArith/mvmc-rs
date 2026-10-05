#!/usr/bin/env python3
"""Check the file:line citations in docs/manual/*.md.

Every citation of the form ``path:LINE`` or ``path:LINE-LINE`` inside backticks
is checked for (a) an existing file and (b) a line number inside the file.
For implementation bullets of the form

    > - C: `symbol` — `path:LINE`
    > - Rust: `symbol` — `path:LINE`

the first identifier of ``symbol`` must additionally occur on the cited line.

This is an optional developer tool; it is not run by the Rust build or tests.

Usage (from the repository root):

    uv run --no-project scripts/check_manual_citations.py \
        [--c-root extern/mVMC-1.3.0]

``extern/mVMC-1.3.0`` is a git submodule. When it is not checked out, pass its
location with ``--c-root`` or the citations into it are skipped with a warning.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BULLET = re.compile(r"^>\s*-\s*(?:C|Rust):\s*`([^`]+)`[^`]*`([^`:]+):(\d+)(?:-(\d+))?`")
CITE = re.compile(r"`((?:[A-Za-z0-9_./-]+/)?[A-Za-z0-9_.-]+\.(?:rs|c|h|rst|toml|md)):(\d+)(?:-(\d+))?`")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def resolve(repo: Path, c_root: Path | None, path: str, index: dict[str, list[Path]]):
    """Return (resolved_path | None, note)."""
    if path.startswith("extern/mVMC-1.3.0/"):
        if c_root is None or not c_root.exists():
            return None, "skipped (C reference not available)"
        return c_root / path[len("extern/mVMC-1.3.0/"):], ""
    candidate = repo / path
    if candidate.exists():
        return candidate, ""
    if "/" not in path:
        hits = index.get(path, [])
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            return None, f"ambiguous basename ({len(hits)} files)"
    return candidate, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--c-root", default=None)
    args = ap.parse_args()
    repo = Path(args.repo)
    c_root = Path(args.c_root) if args.c_root else repo / "extern" / "mVMC-1.3.0"

    index: dict[str, list[Path]] = {}
    for root in (repo / "crates", repo / "docs", repo / "xtask"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".rs", ".md", ".toml"}:
                index.setdefault(p.name, []).append(p)
    if c_root.exists():
        for p in (c_root / "src" / "mVMC").rglob("*"):
            if p.is_file() and p.suffix in {".c", ".h"}:
                index.setdefault(p.name, []).append(p)
        for p in (c_root / "doc" / "en").rglob("*.rst"):
            index.setdefault(p.name, []).append(p)

    cache: dict[Path, list[str]] = {}

    def lines_of(p: Path) -> list[str]:
        if p not in cache:
            cache[p] = p.read_text(errors="replace").splitlines()
        return cache[p]

    errors: list[str] = []
    checked = skipped = 0
    for md in sorted((repo / "docs" / "manual").glob("*.md")):
        for lineno, text in enumerate(md.read_text().splitlines(), 1):
            seen: set[tuple[str, int]] = set()
            m = BULLET.match(text)
            if m:
                symbol, path, start = m.group(1), m.group(2), int(m.group(3))
                resolved, note = resolve(repo, c_root, path, index)
                if resolved is None:
                    skipped += 1
                else:
                    checked += 1
                    if not resolved.exists():
                        errors.append(f"{md.name}:{lineno}: missing file {path}")
                    else:
                        body = lines_of(resolved)
                        if start < 1 or start > len(body):
                            errors.append(f"{md.name}:{lineno}: {path}:{start} beyond {len(body)} lines")
                        else:
                            ident = IDENT.findall(symbol.split("::")[-1] if "::" in symbol.split("(")[0] else symbol)
                            ident = ident[0] if ident else symbol
                            if ident not in body[start - 1]:
                                errors.append(
                                    f"{md.name}:{lineno}: `{ident}` not on {path}:{start}: "
                                    f"{body[start - 1].strip()[:80]!r}"
                                )
                seen.add((path, start))
            for m in CITE.finditer(text):
                path, start = m.group(1), int(m.group(2))
                if (path, start) in seen:
                    continue
                resolved, note = resolve(repo, c_root, path, index)
                if resolved is None:
                    skipped += 1
                    if "ambiguous" in note:
                        errors.append(f"{md.name}:{lineno}: {path}: {note}")
                    continue
                checked += 1
                if not resolved.exists():
                    errors.append(f"{md.name}:{lineno}: missing file {path}")
                    continue
                n = len(lines_of(resolved))
                end = int(m.group(3)) if m.group(3) else start
                if start < 1 or end > n or end < start:
                    errors.append(f"{md.name}:{lineno}: {path}:{start}-{end} outside 1..{n}")
    print(f"checked {checked} citations, skipped {skipped}, {len(errors)} problem(s)")
    for e in errors:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
