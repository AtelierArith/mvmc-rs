"""Create bounded L128 benchmark inputs; leave repository fixtures unchanged."""
import argparse
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    for dtype, source in (
        ("real", repo / "benchmark/cpu_scaling_l128/inputs/opt"),
        ("complex", repo / "gpu/mvmc-gpu-cuda/tests/fixtures/complex/hubbard_complex_L128"),
    ):
        for mode in ("opt", "phys"):
            dest = args.output / f"cpu-{dtype}-{mode}"
            shutil.copytree(source, dest)  # Refuse to overwrite existing inputs.
            path = dest / "modpara.def"
            text, n_sample = re.subn(r"(?m)^(NVMCSample)\s+\d+", r"\1 32", path.read_text())
            text, n_mode = re.subn(r"(?m)^(NVMCCalMode)\s+\d+",
                                  lambda m: f"{m[1]} {int(mode == 'phys')}", text)
            assert n_sample == n_mode == 1
            path.write_text(text)


if __name__ == "__main__":
    main()
