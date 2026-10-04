#!/usr/bin/env -S uv run --no-project
"""Optional actual-C headers/InterAll reader oracle; no Cargo dependency."""
import argparse
import hashlib
from pathlib import Path
import random
import subprocess
import tempfile

from c_toolbox import materialize
from check_general_orbital_c_parity import function


def definition(count, rows, label="NInterAll"):
    return f"===\n{label} {count}\nIgnored 99\n===\n===\n" + rows


def cases(root):
    for sites in (1, 4, 6):
        last = sites - 1
        base = f"0 1 {last} 1 {last} 0 0 0 0.375 -0.625\n"
        variants = {
            "complete": definition(1, base),
            "reordered_duplicate": definition(3, base + f"{last} 0 0 0 0 1 {last} 1 -0.25 0.125\n" + base),
            "zero_width_ignores_body": definition(0, base + "invalid\n"),
            "zero_width_two_headers": "===\nNInterAll 0\n",
            "extra": definition(1, base*2),
            "short": definition(2, base),
            "empty": definition(1, ""),
            "scientific": definition(1, base.rsplit(" ", 2)[0]+" 1e-3 -2.5E+2\n"),
            "hex_float": definition(1, base.rsplit(" ", 2)[0]+" 0x1.8p+1 -0x1p-1\n"),
            "extra_fields": definition(1, base.rstrip()+" 99 trailing\n"),
            "ignored_header_label": definition(1, base, "Whatever"),
            "integer_prefix": definition(2, base+"0.5 1 2 0 0 0 0 0 1 0\n"),
            "short_fields": definition(2, base+f"{last} 0 0 0\n"),
            "real_only": definition(2, base+f"0 1 {last} 1 {last} 0 0 0 -1.25\n"),
            "comment_is_row": definition(2, base+"# comment\n"),
            "blank_is_row": definition(2, base+"\n"),
            "first_comment_is_zero_row": definition(1, "# comment\n"),
            "inline_comments": definition(1, base.rstrip()+" # comment\n"),
            "unicode_is_not_c_whitespace": definition(1, base.replace(" ", "\u2003", 1)),
            "ascii_whitespace": definition(1, base.replace(" ", "\t\v\f\r")),
            "spin_changing": definition(1, f"0 0 0 1 {last} 1 {last} 0 0.5 0.125\n"),
            "bad_site": definition(1, f"{sites} 0 0 0 0 1 0 1 0.5 0\n"),
            "negative_site": definition(1, "-1 0 0 0 0 1 0 1 0.5 0\n"),
            "long_row_chunks": definition(2, base.rstrip()+" "*270+"\n"),
            "no_final_newline": definition(1, base.rstrip()),
        }
        for name, payload in variants.items():
            for sz in (-1, 0, 2):
                yield f"sites{sites}_sz{sz}_{name}", sites, sz, payload
        for k, (re, im) in enumerate((
            ("-0", "0"), ("0", "-0"), ("-0", "-0"), ("-1", "-0"),
            ("1e999", "-1e999"), ("1e-999", "-1e-999"),
            ("nan", "0"), ("0", "nan"), ("Inf", "-Inf"),
            ("-nan", "0"), ("NaN(123)", "0"),
            ("1e+", "0.125"), ("0x1p+", "0.125"),
            ("0.3suffix", "0.125"), ("1.2.3", "0.125"),
            ("0x1.fffffffffffff8p1023", "0"),
            ("0x1.00000000000008p0", "0"),
            ("0x1.0000000000000801p0", "0"),
            ("0x1p-1075", "0"), ("0x1.0000000000001p-1075", "0"),
            ("0x0.fffffffffffff8p-1022", "0"), ("-0x1p-1075", "-0"),
            ("0x00001.8000p-0001", "0"), ("0X.8P1", "0"),
            ("0x1p99999999999999999999", "0"), ("0x1p-99999999999999999999", "0"),
            ("NaN(0x123)", "0"), ("-NaN(077)", "0"), ("nan(payload)", "0"),
            ("infinity", "-Infinity"), ("0x", "0.25"), ("0x.p1", "0.25"),
            ("2.4703282292062327e-324", "0"), ("2.4703282292062328e-324", "0"),
            ("1.7976931348623157e308", "0"), ("1.7976931348623159e308", "0"),
        )):
            yield f"sites{sites}_number{k}", sites, -1, definition(1, base.rsplit(" ", 2)[0]+f" {re} {im}\n")
    # Historical production model has comments that C reads as extra terms.
    historical = root / "tests/fixtures/interall/spin_chain/interall.def"
    yield "historical_spin_chain", 6, -1, historical.read_text()
    yield "c_spin_chain", 6, -1, (root / "tests/fixtures/interall/c_spin_chain.def").read_text()
    # Independent native conversions around varying normal/subnormal scales.
    # This generator stream has no relationship to the VMC/SFMT trajectory.
    generator = random.Random(0x2356)
    for index in range(128):
        digits = ''.join(generator.choice('0123456789abcdef') for _ in range(generator.randint(1, 40)))
        point = generator.randint(0, len(digits))
        value = generator.choice(('', '-'))+'0x'+digits[:point]+'.'+digits[point:]+f'p{generator.randint(-1200, 1100):+}'
        yield f"hex_rounding_{index}", 4, -1, definition(1, f"0 1 3 1 3 0 0 0 {value} -0\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    old = root / "tests/fixtures/interall/spin_chain/interall.def"
    replacement = "\n".join(line for line in old.read_text().splitlines() if not line.startswith("#"))+"\n"
    target = root / "tests/fixtures/interall/c_spin_chain.def"
    if args.write:
        if not target.exists() or target.read_text() != replacement:
            target.write_text(replacement)
    else:
        assert target.read_text() == replacement
    source = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC/readdef.c"
    body = "\n".join(function(source.read_text(), signature) for signature in (
        "char *ReadBuffInt(FILE", "int ReadDefFileError(", "int CheckSite(\n",
        "int CheckPairSite(\n", "int CheckQuadSite(\n", "int GetInfoInterAll(FILE"))
    materialize(root, "interall_reader_upstream.inc", body, [source], args.write)
    output = "# actual C ReadBuffInt/GetInfoInterAll; initialized scan carry; oversized extra-row probe storage; Apple clang 17 -O0 -ffp-contract=off; readdef.c sha256="+hashlib.sha256(source.read_bytes()).hexdigest()+"\n"
    accepted = rejected = 0
    with tempfile.TemporaryDirectory(prefix="mvmc-c-interall-reader-") as directory:
        tmp = Path(directory)
        exe = tmp / "probe"
        subprocess.run(["cc", "-O0", "-ffp-contract=off", str(root / "c_toolbox/interall_reader.c"), "-o", str(exe)], check=True)
        for name, sites, sz, payload in cases(root):
            path = tmp / "interall.def"
            path.write_text(payload)
            result = subprocess.run([str(exe), str(sites), str(sz), str(path)], text=True,
                                    capture_output=True, check=True, timeout=5)
            rows = result.stdout.splitlines()
            ok, width, status = map(int, rows[0].split())
            valid = ok == 1 and status == 0
            accepted += valid
            rejected += not valid
            output += f"{name} {sites} {sz} {int(valid)} {width}\n"+payload.encode().hex()+"\n"
            output += "|".join(rows[1:])+"\n"
    target = root / "tests/fixtures/interall/c_reader.txt"
    if args.write:
        if not target.exists() or target.read_text() != output:
            target.write_text(output)
    else:
        assert target.read_text() == output, "Native C InterAll reader fixture changed"
    print(f"{accepted+rejected} native C InterAll reader cases passed ({accepted} accepted / {rejected} rejected)")


if __name__ == "__main__":
    main()
