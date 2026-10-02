#!/usr/bin/env -S uv run --no-project
"""Explicit complete C reader inputs for historical numerical control models."""
import argparse
import os
from pathlib import Path

from check_rbm_contracts_c_parity import NAMES, geometry


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
        has_rbm = any(line.split()[0] in NAMES for line in original.read_text().splitlines())
        namelist = ""
        for line in original.read_text().splitlines():
            kind, filename = line.split()
            source = Path(os.path.normpath(original.parent / filename))
            if kind == "ModPara" and has_rbm:
                contents = source.read_text()
                assert not any(line.split()[0].startswith("Nneuron") for line in contents.splitlines() if line.split())
                contents += "NneuronCharge 2\nNneuronSpin 2\nNneuronGeneral 2\n"
                destination = target / f"modpara_{name}.def"
                outputs[destination] = contents
                filename = destination.name
            elif kind.startswith(("InChargeRBM_", "InSpinRBM_", "InGeneralRBM_")):
                lines = source.read_text().splitlines()
                assert len(lines) == 8
                values = {int(row.split()[0]): row.split()[1:] for row in lines[5:]}
                assert set(values) == {0, 1, 2}
                lines[1] = "NParameter 4"
                # C loads section records in row order, ignoring printed labels.
                lines[5:] = [f"{k} {' '.join(values[k])}" for k in range(3)]+["3 0.0 0.0"]
                destination = target / f"historical_binary_rbm/rbm_input_{name}.def"
                contents = "\n".join(lines)+"\n"
                if destination in outputs: assert outputs[destination] == contents
                outputs[destination] = contents
                filename = os.path.relpath(destination, target)
            elif kind == "Orbital" and name in ("dh2_cmp", "dh4_cmp", "dh24_cmp", "rbm_dh24_cmp", "opt_dh24_rbm_cmp"):
                filename = "ap_hubbard_six_complex.def"
            elif kind.startswith(("ChargeRBM_", "SpinRBM_", "GeneralRBM_")):
                # Keep original nonzero mappings and binary active flags; fill
                # the remaining geometry with a fourth, fixed-zero coefficient.
                # Mapping it to old inactive index 2 would change the physical
                # model, since its explicit overlay value is nonzero.
                lines = source.read_text().splitlines()
                assert [row.split() for row in lines[-3:]] == [["0","1"],["1","2"],["2","0"]]
                section = NAMES.index(kind)
                original_rows = [tuple(map(int, row.split())) for row in lines[5:-3]]
                assert len(original_rows) == 2
                assignments = {row[:-1]: row[-1] for row in original_rows}
                coordinates = geometry(section, 6, 2)
                assert set(assignments).issubset(coordinates)
                lines[1] = "NParameter 4"
                lines[5:] = [" ".join(map(str, coord+(assignments.get(coord, 3),))) for coord in coordinates]
                lines += ["0 1", "1 1", "2 0", "3 0"]
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
    print(f"{len(outputs)} explicit complete control input files verified")


if __name__ == "__main__":
    main()
