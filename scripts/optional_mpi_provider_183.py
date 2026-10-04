"""Matched MPI dependency receipts; no install, model or oracle execution here."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


MANIFESTS = ("source", "tools", "compiler-providers", "provider", "startup", "startup-providers")
READY = "MPI_PROVIDER_READY version=4.2.0 pmi=pmi1 worlds=2,4\n"


def digest(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/nonregular MPI receipt: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_state(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("missing/unsafe matched MPI receipt root")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("unsafe MPI receipt member")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def validate_startup(root, prefix):
    if not prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("invalid MPI prefix")
    if (root / "terminal.txt").read_text() != "prior=0 post=0\n" or (root / "ready.txt").read_text() != READY:
        raise ValueError("matched MPI installer/startup not completed")
    version = (root / "mpichversion.txt").read_text()
    if not re.search(r"^MPICH Version:\s+4\.2\.0$", version, re.M) or not all(
            word in version for word in (f"--prefix={prefix}", "--with-pm=hydra", "--with-pmi=pmi1", "--without-pmix")):
        raise ValueError("unmatched MPI version/configuration")
    for name in MANIFESTS:
        if not (root / f"{name}.sha256").read_text() or (root / f"{name}.post.log").read_bytes():
            raise ValueError("absent/failed MPI installer post check")
    for world in (2, 4):
        if (root / f"world-{world}.native.status").read_text() != "0\n":
            raise ValueError("MPI startup native failure")
        ranks = []
        for line in (root / f"world-{world}.stdout").read_text().splitlines():
            match = re.fullmatch(r"MPI_STARTUP rank=([0-3]) size=([24]) required=([0-3]) provided=([0-3]) rank_sum=([0-6]) ok=1", line)
            if match is None:
                raise ValueError("malformed MPI startup record")
            rank, size, required, provided, total = map(int, match.groups())
            if size != world or required != 1 or provided < required or total != world * (world - 1) // 2:
                raise ValueError("wrong MPI startup world/thread/sum")
            ranks.append(rank)
        if sorted(ranks) != list(range(world)):
            raise ValueError("missing/duplicate MPI startup ranks")


def runtime_state(receipt):
    """Verify live installer-bound inputs, not just copies of successful logs."""
    result = {}
    for name in MANIFESTS:
        for line in (receipt / f"{name}.sha256").read_text().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (/[^\r\n]+)", line)
            if match is None:
                raise ValueError("malformed MPI live hash manifest")
            expected, filename = match.groups()
            # Upstream sha256sum follows linker SONAME symlinks. Keep the
            # original path key, but hash the resolved regular live provider.
            actual = digest(Path(filename).resolve(strict=True))
            if actual != expected or (filename in result and result[filename] != actual):
                raise ValueError("MPI installer-bound input changed")
            result[filename] = actual
    if not result:
        raise ValueError("empty MPI runtime closure")
    return result


def capture(output, environment):
    prefix = Path(environment.get("MVMC_ISSUE234_MPI_PREFIX", ""))
    receipt = Path(environment.get("MVMC_ISSUE234_MPI_RECEIPT", ""))
    validate_startup(receipt, prefix)
    tools = {}
    for name in ("mpicc", "mpiexec"):
        value = shutil.which(name)
        if value is None:
            raise ValueError("matched MPI tool unavailable")
        resolved = Path(value).resolve(strict=True)
        if not resolved.is_relative_to(prefix / "bin"):
            raise ValueError("MPI tool resolves outside matched prefix")
        tools[name] = {"path": str(resolved), "sha256": digest(resolved)}
    if Path(environment.get("MPICC", "")).resolve() != Path(tools["mpicc"]["path"]):
        raise ValueError("Cargo MPICC differs from matched MPI tool")
    state = runtime_state(receipt)
    if str(prefix / "include/mpi.h") not in state:
        raise ValueError("MPI header not bound")
    raw_hashes = receipt_state(receipt)
    shutil.copytree(receipt, output / "mpi-provider-receipt")
    if receipt_state(output / "mpi-provider-receipt") != raw_hashes:
        raise ValueError("MPI receipt copy changed")
    binding = {"schema": 1, "prefix": str(prefix), "receipt_root": str(receipt),
               "receipt_hashes": raw_hashes, "tools": tools}
    (output / "mpi-provider.before.json").write_text(json.dumps(state, sort_keys=True) + "\n")
    return binding


def finish(output, binding, binary):
    receipt = Path(binding["receipt_root"])
    state = runtime_state(receipt)
    before = json.loads((output / "mpi-provider.before.json").read_text())
    if state != before or receipt_state(receipt) != binding["receipt_hashes"]:
        raise ValueError("MPI provider/receipt changed during selected run")
    for name, tool in binding["tools"].items():
        if Path(shutil.which(name) or "").resolve() != Path(tool["path"]) or digest(Path(tool["path"])) != tool["sha256"]:
            raise ValueError("MPI launcher/compiler changed")
    linkage = subprocess.check_output(["ldd", str(binary)], text=True)
    (output / "mpi-linkage.txt").write_text(linkage)
    prefix = Path(binding["prefix"])
    matches = re.findall(r"^\s*libmpi\.so\S*\s+=>\s+(\S+)", linkage, re.M)
    if len(matches) != 1 or "not found" in linkage or "libpmix" in linkage:
        raise ValueError("selected ELF MPI linkage not matched")
    library = Path(matches[0]).resolve(strict=True)
    if not library.is_relative_to(prefix / "lib") or str(library) not in state:
        raise ValueError("selected ELF MPI library outside bound prefix")
    binding.update(binary_sha256=digest(binary), library={"path": matches[0],
                   "resolved": str(library), "sha256": digest(library)})
    (output / "mpi-provider.after.json").write_text(json.dumps(state, sort_keys=True) + "\n")
    (output / "mpi-provider.json").write_text(json.dumps(binding, sort_keys=True) + "\n")


def validate_package(evidence, read_json):
    """Offline aggregate validation: does not touch original runner paths."""
    binding = read_json(evidence / "mpi-provider.json")
    if type(binding.get("schema")) is not int or binding["schema"] != 1:
        raise ValueError("wrong MPI binding schema")
    prefix = Path(binding["prefix"])
    receipt = evidence / "mpi-provider-receipt"
    if receipt_state(receipt) != binding["receipt_hashes"]:
        raise ValueError("changed/unlisted MPI receipt member")
    validate_startup(receipt, prefix)
    before = read_json(evidence / "mpi-provider.before.json")
    if not before or before != read_json(evidence / "mpi-provider.after.json"):
        raise ValueError("empty/changed MPI runtime closure")
    if any(not Path(path).is_absolute() or ".." in Path(path).parts or
           not re.fullmatch(r"[0-9a-f]{64}", value) for path, value in before.items()):
        raise ValueError("invalid MPI runtime hash")
    library = binding["library"]
    if any(not Path(library[key]).is_absolute() or ".." in Path(library[key]).parts or
           not Path(library[key]).is_relative_to(prefix / "lib") for key in ("path", "resolved")) or before.get(library["resolved"]) != library["sha256"]:
        raise ValueError("unbound selected MPI library")
    linkage = (evidence / "mpi-linkage.txt").read_text()
    if re.findall(r"^\s*libmpi\.so\S*\s+=>\s+(\S+)", linkage, re.M) != [library["path"]] or "not found" in linkage or "libpmix" in linkage:
        raise ValueError("changed selected MPI linkage")
    binary = read_json(evidence / "binary.json")
    if list(binary.values()) != [binding["binary_sha256"]]:
        raise ValueError("MPI binding belongs to another selected ELF")
    if set(binding["tools"]) != {"mpicc", "mpiexec"} or str(prefix / "include/mpi.h") not in before:
        raise ValueError("MPI compiler/launcher/header binding absent")
    for tool in binding["tools"].values():
        if not Path(tool["path"]).is_absolute() or ".." in Path(tool["path"]).parts or not Path(tool["path"]).is_relative_to(prefix / "bin") or before.get(tool["path"]) != tool["sha256"]:
            raise ValueError("MPI tool not bound in runtime closure")
