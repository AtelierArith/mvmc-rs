"""Derive the complex PhysCal input without duplicating the checked-in defs."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = repo / "gpu/mvmc-gpu-cuda/tests/fixtures/complex/hubbard_complex_L128"
    shutil.copytree(source, args.destination)
    path = args.destination / "modpara.def"
    lines = path.read_text().splitlines()
    assert sum(line.split()[:1] == ["NVMCCalMode"] for line in lines) == 1
    path.write_text("\n".join("NVMCCalMode 1" if line.split()[:1] == ["NVMCCalMode"]
                              else line for line in lines) + "\n")


if __name__ == "__main__":
    main()
