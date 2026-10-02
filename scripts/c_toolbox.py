"""Maintain optional C reference snippets; Rust tests consume fixtures only."""
import hashlib


def materialize(root, name, body, sources, write):
    provenance = "; ".join(
        f"{source.name} sha256={hashlib.sha256(source.read_bytes()).hexdigest()}"
        for source in sources
    )
    notice = sources[0].read_text().split("#include", 1)[0]
    result = (f"/* Generated verbatim from authoritative mVMC-1.3.0: {provenance}.\n"
              " * Regenerate with the corresponding scripts/check_*_c_parity.py --write.\n"
              " * This optional C oracle is not a Rust test/build dependency. */\n"
              + notice + body.rstrip() + "\n")
    target = root / "c_toolbox" / name
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != result:
            target.write_text(result)
    else:
        assert target.read_text() == result, f"C toolbox source changed: {name}"
