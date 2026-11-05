#!/usr/bin/env python3
"""Optional #682 oracle: zero `zvo_CalcTimer.dat` reports of C mVMC 1.4.0.

Extracts `OutputTimerParaOpt` and `OutputTimerPhysCal` verbatim from
`extern/mVMC-1.4.0/src/mVMC/vmcclock.c` into `c_toolbox/c_timer_report_upstream.inc`,
compiles `c_toolbox/c_timer_report.c` and runs both reports with every timer at
zero. With `--write` it stores them as `tests/fixtures/timers/c_*_zero.dat`;
otherwise it verifies the checked-in files. Rust tests read only the fixtures.

    uv run --no-project python scripts/check_c_timer_c_parity.py [--write]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from c_toolbox import materialize
from check_projection_count_c_parity import function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.4.0") / "src/mVMC"
    clock = (src / "vmcclock.c").read_text()
    body = "\n".join(function(clock, signature)
                     for signature in ("void OutputTimerParaOpt()", "void OutputTimerPhysCal()"))
    materialize(root, "c_timer_report_upstream.inc", body, [src / "vmcclock.c"], args.write)
    fixtures = root / "tests/fixtures/timers"
    with tempfile.TemporaryDirectory(prefix="mvmc-c-timer-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", "-I", str(root / "c_toolbox"),
                        str(root / "c_toolbox/c_timer_report.c"), "-o", str(tmp / "probe")],
                       check=True)
        for mode, name in (("paraopt", "c_para_opt_zero.dat"), ("physcal", "c_phys_cal_zero.dat")):
            subprocess.run([str(tmp / "probe"), mode, mode], cwd=tmp, check=True)
            output = (tmp / f"{mode}_CalcTimer.dat").read_text()
            target = fixtures / name
            if args.write:
                target.write_text(output)
            else:
                assert target.read_text() == output, f"C timer report changed: {name}"
    print("C ParaOpt and PhysCal zero timer reports checked")


if __name__ == "__main__":
    main()
