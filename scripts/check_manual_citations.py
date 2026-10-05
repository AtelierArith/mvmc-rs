#!/usr/bin/env python3
"""Check the file:line citations and relative links in docs/manual/*.md and
docs/manual/ja/*.md.

Every citation of the form ``path:LINE`` or ``path:LINE-LINE`` inside backticks
is checked for (a) an existing file and (b) a line number inside the file.
For implementation bullets of the form

    > - C: `symbol` — `path:LINE`
    > - Rust: `symbol` — `path:LINE`

the first identifier of ``symbol`` must additionally occur on the cited line.

Relative Markdown links (including ``#anchor`` fragments, using GitHub heading
slugs) are also checked for the English and Japanese manuals, and the Japanese
manual must mirror every English file name.

This is an optional developer tool; it is not run by the Rust build or tests.

Usage (from the repository root):

    python3 scripts/check_manual_citations.py \
        [--c-root extern/mVMC-1.3.0] [--fix]

The script uses only the Python standard library (``uv run --no-project`` also
works). CI runs it with ``--tolerant``: a bullet citation whose symbol moved but is
still found within ``--window`` lines only warns; a missing file, symbol, link or
anchor, or a symbol outside the window, fails. Strict mode (no flag) fails on any
line drift and is for maintenance. With
``--fix``, a bullet citation whose symbol is no longer on the cited line is
rewritten to the nearest line within ``--window`` lines (default 400) that
defines or contains the symbol; the corrected lines are listed and the check is
re-run. Citations whose symbol is not found nearby remain errors.

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


LINK = re.compile(r"(?<!\!)\[(?:[^\]]|\[[^\]]*\])*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def slugify(title: str) -> str:
    """GitHub heading slug: lowercase, drop punctuation, spaces to hyphens."""
    title = re.sub(r"[*_`]", lambda m: m.group(0) if m.group(0) == "_" else "", title)
    out = []
    for ch in title.lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch == " ":
            out.append("-")
    return "".join(out)


_anchor_cache: dict[Path, set[str]] = {}


def anchors_of(p: Path) -> set[str]:
    if p not in _anchor_cache:
        seen: dict[str, int] = {}
        result: set[str] = set()
        fence = False
        for line in p.read_text().splitlines():
            if line.startswith("```"):
                fence = not fence
            if fence:
                continue
            m = HEADING.match(line)
            if m:
                slug = slugify(m.group(2))
                n = seen.get(slug, 0)
                seen[slug] = n + 1
                result.add(slug if n == 0 else f"{slug}-{n}")
        _anchor_cache[p] = result
    return _anchor_cache[p]


def check_links(md: Path, md_name: str, errors: list[str]) -> int:
    count = 0
    fence = False
    for lineno, text in enumerate(md.read_text().splitlines(), 1):
        if text.startswith("```"):
            fence = not fence
        if fence:
            continue
        for m in LINK.finditer(re.sub(r"`[^`]*`", "", text)):
            target = m.group(1)
            if re.match(r"[a-z]+:", target):
                continue
            count += 1
            path, _, frag = target.partition("#")
            dest = (md.parent / path).resolve() if path else md
            if not dest.exists():
                errors.append(f"{md_name}:{lineno}: broken link {target}")
            elif frag and dest.suffix == ".md" and frag not in anchors_of(dest):
                errors.append(f"{md_name}:{lineno}: missing anchor {target}")
    return count


DEFINITION = re.compile(r"\b(?:fn|struct|enum|trait|type|const|static|mod)\s+{ident}\b")


def nearest_symbol_line(body: list[str], ident: str, start: int, window: int) -> int | None:
    """Nearest 1-based line within ``window`` of ``start`` defining (else containing) ``ident``."""
    definition = re.compile(DEFINITION.pattern.format(ident=re.escape(ident)))
    for matcher in (lambda l: definition.search(l), lambda l: ident in l):
        hits = [i + 1 for i, l in enumerate(body) if matcher(l) and abs(i + 1 - start) <= window]
        if hits:
            return min(hits, key=lambda h: abs(h - start))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--c-root", default=None)
    ap.add_argument("--fix", action="store_true",
                    help="rewrite stale bullet-citation line numbers when the symbol is found nearby")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--tolerant", action="store_true",
                    help="CI mode: a moved symbol found within --window lines is a warning, not an error")
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
    fixes: list[tuple[Path, int, str, int, int]] = []
    warnings: list[str] = []
    checked = skipped = 0
    manual = repo / "docs" / "manual"
    en_files = sorted(manual.glob("*.md"))
    ja_files = sorted((manual / "ja").glob("*.md"))
    for f in en_files:
        if not (manual / "ja" / f.name).exists():
            errors.append(f"ja/{f.name}: missing Japanese translation")
    for f in ja_files:
        if not (manual / f.name).exists():
            errors.append(f"ja/{f.name}: no English counterpart")
    links = 0
    for md in en_files + ja_files:
        md_name = str(md.relative_to(manual))
        links += check_links(md, md_name, errors)
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
                        errors.append(f"{md_name}:{lineno}: missing file {path}")
                    else:
                        body = lines_of(resolved)
                        if start < 1 or start > len(body):
                            errors.append(f"{md_name}:{lineno}: {path}:{start} beyond {len(body)} lines")
                        else:
                            ident = IDENT.findall(symbol.split("::")[-1] if "::" in symbol.split("(")[0] else symbol)
                            ident = ident[0] if ident else symbol
                            if ident not in body[start - 1]:
                                near = nearest_symbol_line(body, ident, start, args.window)
                                if args.fix and near is not None:
                                    fixes.append((md, lineno, path, start, near))
                                    continue
                                if args.tolerant and near is not None:
                                    warnings.append(
                                        f"{md_name}:{lineno}: `{ident}` is at {path}:{near}, "
                                        f"cited {start} (run --fix)"
                                    )
                                    continue
                                errors.append(
                                    f"{md_name}:{lineno}: `{ident}` not on {path}:{start}: "
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
                        errors.append(f"{md_name}:{lineno}: {path}: {note}")
                    continue
                checked += 1
                if not resolved.exists():
                    errors.append(f"{md_name}:{lineno}: missing file {path}")
                    continue
                n = len(lines_of(resolved))
                end = int(m.group(3)) if m.group(3) else start
                if start < 1 or end > n or end < start:
                    errors.append(f"{md_name}:{lineno}: {path}:{start}-{end} outside 1..{n}")
    for md, lineno, path, old_line, new_line in fixes:
        lines = md.read_text().split("\n")
        lines[lineno - 1] = re.sub(
            rf"{re.escape(path)}:{old_line}(?:-(\d+))?`",
            lambda m: f"{path}:{new_line}"
            + (f"-{int(m.group(1)) + new_line - old_line}" if m.group(1) else "")
            + "`",
            lines[lineno - 1],
        )
        md.write_text("\n".join(lines))
        print(f"fixed {md.relative_to(manual)}:{lineno}: {path}:{old_line} -> {new_line}")
    if fixes:
        print(f"rewrote {len(fixes)} citation(s); re-run to verify")
    print(f"checked {checked} citations, {links} relative links, skipped {skipped}, "
          f"{len(errors)} problem(s)")
    for w in warnings:
        print("  warning: " + w)
    for e in errors:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
