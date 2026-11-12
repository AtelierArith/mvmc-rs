"""Assemble INTERVAL_MODULE from any recorded checkout for cuda_resources.cpp."""
import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    src = args.checkout / "gpu/mvmc-gpu-cuda/src"
    text = (src / "interval.rs").read_text().split("pub const INTERVAL_MODULE:", 1)[1]
    text = text.split(");", 1)[0]
    files = re.findall(r'include_str!\("([^"]+)"\)', text)
    assert files
    with args.output.open("x") as output:
        output.write("\n".join((src / name).read_text() for name in files))


if __name__ == "__main__":
    main()
