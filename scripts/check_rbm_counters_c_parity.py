#!/usr/bin/env -S uv run --no-project
"""Optional actual-C nine-section MakeRBMCnt/UpdateRBMCnt oracle."""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function
from check_rbm_contracts_c_parity import definition, geometry, materialize_readers


def bits(value):
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def models():
    for sites, hidden in ((1, 1), (3, 2), (6, 4)):
        for selected in range(11):
            widths = [0]*9
            if selected < 9:
                widths[selected] = selected+5
            elif selected == 9:
                widths = list(range(5, 14))
            else:
                widths = [97, 5, 5, 5, 5, 5, 5, 5, 5]
            definitions = []
            for section, width in enumerate(widths):
                if not width:
                    definitions.append("-")
                    continue
                coords = geometry(section, sites, hidden)
                # Complete geometry, shuffled order, tied and unused slots.
                rows = [" ".join(map(str, coord+(k%5,)))+"\n" for k, coord in enumerate(coords)]
                definitions.append(definition(width, 1, "".join(reversed(rows)), "".join(f"-7 1\n" for _ in range(width))))
            for pattern in range(2):
                values = []
                nonbinary = [0.1, -0.3, 1e16, 1.0, -1e16, 1/3, 1e-17]
                for k in range(sum(widths)):
                    re = (k+1)/64 if pattern == 0 else nonbinary[k%len(nonbinary)]
                    im = -(k+1)/128 if pattern == 0 else nonbinary[(k+3)%len(nonbinary)]
                    values.extend((bits(re), bits(im)))
                limit = 1 << (2*sites)
                masks = sorted({0, limit-1, limit//3, 2*limit//3, 1, limit//2, 3, limit-2})
                cases = []
                for mask in masks:
                    for spin in range(2):
                        for ri in range(sites):
                            # No-op and every physically legal single hop.
                            cases.append((mask, ri, ri, spin))
                            if not (mask >> (ri+spin*sites))&1:
                                continue
                            for rj in range(sites):
                                if ri != rj and not (mask >> (rj+spin*sites))&1:
                                    cases.append((mask, ri, rj, spin))
                yield f"sites{sites}_hidden{hidden}_section{selected}_pattern{pattern}", sites, hidden, widths, definitions, values, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader, kernel = src / "readdef.c", src / "rbm.c"
    # Verify each native dependency independently of other oracle invocations.
    materialize_readers(root, reader, args.write)
    materialize(root, "rbm_counters_upstream.inc", "\n".join(function(kernel.read_text(), sig) for sig in (
        "void MakeRBMCnt(", "void UpdateRBMCnt(")), [kernel], args.write)
    hashes = "; ".join(f"{p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}" for p in (reader, kernel))
    output = f"# actual C MakeRBMCnt/UpdateRBMCnt and readers; Nneuron=sum of family dimensions; complete assigned mappings; Apple clang 17 -O0 -ffp-contract=off; {hashes}\n"
    checked = tables = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-rbm-counters-") as directory:
        tmp = Path(directory)
        exe = tmp / "probe"
        subprocess.run(["cc", "-O0", "-ffp-contract=off", str(root / "c_toolbox/rbm_counters.c"), "-lm", "-o", str(exe)], check=True)
        for name, sites, hidden, widths, definitions, values, cases in models():
            files = []
            for s, content in enumerate(definitions):
                filename = tmp / f"section{s}.def"
                if content != "-": filename.write_text(content)
                files.append(str(filename) if content != "-" else "-")
            payload = " ".join(values)+"\n"+"\n".join(" ".join(map(str, row)) for row in cases)+"\n"
            result = subprocess.run([str(exe), str(sites), str(hidden)]+files, input=payload, text=True, capture_output=True, check=True, timeout=5)
            lines = result.stdout.splitlines()
            assert len(lines) == 3*len(cases)
            output += f"MODEL {name} {sites} {hidden} {len(cases)} {' '.join(map(str,widths))}\n"
            output += "~".join(content.replace("\n", "|") for content in definitions)+"\n"
            output += " ".join(values)+"\n"
            for i, row in enumerate(cases):
                old, new, inplace = lines[3*i:3*i+3]
                assert new == inplace, (name, row, "native in-place mismatch")
                output += " ".join(map(str, row))+"\n"+old+"\n"+new+"\n"
                checked += 1
            tables += 1
    target = root / "tests/fixtures/rbm/c_counters.txt"
    if args.write:
        if not target.exists() or target.read_text() != output: target.write_text(output)
    else:
        assert target.read_text() == output, "native C RBM counters changed"
    print(f"{checked} native C RBM counter/hop cases across {tables} tables passed (including in-place checks)")


if __name__ == "__main__":
    main()
