"""Maintain optional C reference snippets; Rust tests consume fixtures only."""
import hashlib


def materialize(root, name, body, sources, write):
    provenance = "; ".join(
        f"{source.name} sha256={hashlib.sha256(source.read_bytes()).hexdigest()}"
        for source in sources
    )
    upstream = sources[0].read_text()
    notice = (upstream.split("#include", 1)[0] if "#include" in upstream
              else upstream.split("*/", 1)[0] + "*/\n")
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


def add_native_platform_argument(parser):
    parser.add_argument("--platform", choices=("auto", "apple", "linux-gnu"), default="auto",
                        help="Native fixture platform; auto selects macOS or Linux x86_64 glibc")


def native_platform(selection):
    import platform
    import sys
    detected = ("apple" if sys.platform == "darwin" else
                "linux-gnu" if sys.platform == "linux" and platform.machine() == "x86_64"
                and platform.libc_ver()[0] == "glibc" else None)
    if detected is None or (selection != "auto" and selection != detected):
        raise RuntimeError(f"Unsupported native C fixture platform: {sys.platform} "
                           f"{platform.machine()} {platform.libc_ver()}, requested {selection}")
    return detected


def native_compiler():
    import os
    import shlex
    # The staging shell's cc shim records actual commands using REFERENCE_CC.
    return ["cc"] if os.environ.get("REFERENCE_CC") else shlex.split(os.environ.get("CC", "cc"))


def native_provenance(kind):
    import platform
    import subprocess
    if kind == "apple":
        return None  # Preserve existing Apple fixture headers/bytes exactly.
    compiler = native_compiler()
    version = subprocess.check_output(compiler + ["--version"], text=True).splitlines()[0]
    archive = subprocess.check_output(compiler + ["-print-libgcc-file-name"], text=True).strip()
    from pathlib import Path
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise RuntimeError(f"Native GNU compiler runtime archive missing: {archive}")
    return (f"Linux x86_64 GNU; compiler={version}; libc={platform.libc_ver()}; "
            f"libgcc_archive={archive_path}; "
            f"libgcc_archive_sha256={hashlib.sha256(archive_path.read_bytes()).hexdigest()}; "
            "-O0 -ffp-contract=off")


def native_target(root, stem, kind):
    suffix = "_linux_gnu" if kind == "linux-gnu" else ""
    return root / "tests/fixtures/interall" / (stem + suffix + ".txt")
