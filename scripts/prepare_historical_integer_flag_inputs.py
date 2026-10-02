#!/usr/bin/env python3
"""Explicit legacy Julia SR inputs; not full C family acceptance fixtures."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    target = root / "tests/fixtures/c_orbital_inputs"
    source = root / "extern/Julia-mVMC/examples/inputs/hubbard_chain_real/orbitalidx.def"
    assert source.read_text().count("ComplexType          0") == 1
    outputs = {target / "ap_hubbard_six_complex.def":
               source.read_text().replace("ComplexType          0", "ComplexType          1")}
    # A complete real AP control for actual CLI flag-2 behavior (no legacy mask).
    lines = source.read_text().splitlines()
    assert len(lines) == 5 + 36 + 12
    assert all(row.split()[1] == "1" for row in lines[-12:])
    lines[-12:] = [f"{index} 2" for index in range(12)]
    outputs[target / "ap_hubbard_six_flag2.def"] = "\n".join(lines)+"\n"
    families = [("dh2_cmp", "dh2/production_cmp/namelist.def"),
                ("dh4_cmp", "dh4/production_dh4_cmp/namelist.def"),
                ("dh24_cmp", "dh4/production_dh24_cmp/namelist.def"),
                ("opt_dh24_rbm_cmp", "opttrans/run_opt_dh24_rbm_cmp/namelist.def")]
    for case in ("rbm_real", "rbm_cmp", "rbm_general_cmp", "rbm_dh24_cmp", "rbm_fsz"):
        families.append((case, f"rbm/run_{case}/namelist.def"))
    for name, relative in families:
        original = root / "tests/fixtures" / relative
        namelist = ""
        for line in original.read_text().splitlines():
            kind, filename = line.split()
            source = Path(os.path.normpath(original.parent / filename))
            if kind == "Orbital" and name in ("dh2_cmp", "dh4_cmp", "dh24_cmp", "rbm_dh24_cmp", "opt_dh24_rbm_cmp"):
                filename = "ap_hubbard_six_complex.def"
            elif kind.startswith(("ChargeRBM_", "SpinRBM_", "GeneralRBM_")):
                # Julia's bool flag assembly made raw flag 2 equivalent to 1.
                # C does not. These explicit binary inputs preserve the OLD
                # Julia numerical regression while raw 2 keeps its C semantics
                # in production and the separate native eligibility tests.
                lines = source.read_text().splitlines()
                assert [row.split() for row in lines[-3:]] == [["0","1"],["1","2"],["2","0"]]
                lines[-2] = "1 1"
                destination = target / "historical_binary_rbm" / source.name
                contents = "\n".join(lines)+"\n"
                if destination in outputs: assert outputs[destination] == contents
                outputs[destination] = contents
                filename = os.path.relpath(destination, target)
            else:
                filename = os.path.relpath(source, target)
            namelist += f"{kind} {filename}\n"
        outputs[target / f"namelist_{name}.def"] = namelist
    for path, text in outputs.items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_text()!=text: path.write_text(text)
        else:
            assert path.read_text()==text, f"legacy input changed: {path}"
    print(f"{len(outputs)} explicit legacy input files verified")


if __name__ == "__main__":
    main()
