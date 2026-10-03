#!/usr/bin/env -S uv run --no-project
"""Optional native clang complex division; pure-Rust Cargo tests use fixtures."""
import argparse
import hashlib
from pathlib import Path
import random
import subprocess
import tempfile

from check_interall_real_c_parity import bits


def cases():
    positive_zero, negative_zero = 0.0, -0.0
    subnormal = float.fromhex('0x0.0000000000001p-1022')
    numerators = [(1.0, 0.0), (0.0, 1.0), (-0.3, 0.7), (positive_zero, negative_zero),
                  (1e308, -1e308), (1e-308, 1e-308), (subnormal, -subnormal),
                  (float('inf'), 0.7), (float('-inf'), float('inf'))]
    denominators = [(3.0, 4.0), (-0.1, -1.7), (1.0, negative_zero),
                    (negative_zero, 1.0), (positive_zero, positive_zero),
                    (negative_zero, negative_zero), (1e308, 1e308),
                    (1e-308, -1e-308), (subnormal, subnormal),
                    (float.fromhex('0x1p-1022'), subnormal),
                    (float('inf'), 3.0), (2.0, float('-inf')),
                    (float('inf'), float('-inf'))]
    for numerator in numerators:
        for denominator in denominators:
            yield ' '.join(map(bits, numerator+denominator))
    generator = random.Random(0x2357)
    for _ in range(256):
        values = []
        for _ in range(4):
            value = generator.getrandbits(64)
            if (value >> 52) & 0x7ff == 0x7ff:
                value ^= 1 << 52
            values.append(f'{value:016x}')
        yield ' '.join(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = root/'c_toolbox/complex_division.c'
    llvm = root/'c_toolbox/llvm_divdc3.c'
    provenance = '; '.join(f'{path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}'
                           for path in (source, llvm))
    inputs = list(cases())
    with tempfile.TemporaryDirectory(prefix='mvmc-c-division-') as directory:
        exe = Path(directory)/'probe'
        subprocess.run(['cc','-O0','-ffp-contract=off',str(source),'-o',str(exe)],check=True)
        actual = subprocess.check_output([str(exe)],input='\n'.join(inputs)+'\n',text=True).splitlines()
    assert len(actual) == len(inputs)
    output = ('# Native Apple clang 17 complex quotient -O0 -ffp-contract=off; '
              'normal/subnormal/range/nonfinite inputs; NaN classification, all other bits exact; '
              'LLVM reference llvmorg-17.0.6/compiler-rt/lib/builtins/divdc3.c; '+provenance+'\n')
    output += '\n'.join(row+' '+value for row,value in zip(inputs,actual))+'\n'
    target = root/'tests/fixtures/interall/c_complex_division.txt'
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, 'Native complex division fixture changed'
    print(f'{len(inputs)} native C complex quotient cases passed')


if __name__ == '__main__':
    main()
