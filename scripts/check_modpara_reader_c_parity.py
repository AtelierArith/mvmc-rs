#!/usr/bin/env python3
"""Optional #344 oracle: C modpara.def reader (SetDefaultValuesModPara +
GetInfoFromModPara + NBlockSize_RBMRatio adjustment) on accepted/rejected inputs.

Extracts the authoritative functions verbatim into c_toolbox/, compiles
c_toolbox/modpara_reader.c, runs every variant below and (with --write) stores
the inputs and C results under tests/fixtures/modpara_reader/. Rust tests only
read those fixtures; they never run this script or any C code.
"""
import argparse
import hashlib
from pathlib import Path
import platform
import re
import subprocess
import tempfile

from c_toolbox import materialize
from check_projection_count_c_parity import function

HEADER = ("---\nModel_Parameters   0\n---\nVMC_Cal_Parameters\n---\n"
          "CDataFileHead  zvo\nCParaFileHead  zqp\n---\n")

FULL = """NVMCCalMode    0
NLanczosMode   0
---
NDataIdxStart  1
NDataQtySmp    1
---
Nsite          6
Ncond          0
2Sz            0
NSPGaussLeg    8
NSPStot        0
NMPTrans       -1
NSROptItrStep  1000
NSROptItrSmp   100
DSROptRedCut   0.0000000001
DSROptStaDel   0.0000100000
DSROptStepDt   0.0100000000
NVMCWarmUp     10
NVMCInterval   2
NVMCSample     200
NExUpdatePath  0
RndSeed        11272
NSplitSize     1
NStore         1
NSRCG          0
"""


def variants():
    """(name, full file text). Names are fixture file stems."""
    out = []

    def add(name, body, header=HEADER):
        out.append((name, header + body))

    add("defaults_only", "")
    add("full_typical", FULL)
    add("keywords_lowercase", FULL.lower())
    add("keywords_uppercase", FULL.upper())
    add("keywords_mixed_case", "nSiTe 4\nNCOND 2\nnvmcsample 33\nDsrOptRedCut 1e-3\n")
    add("ne_key", "Ne 5\n")
    add("nelectron_key", "Nelectron 7\n")
    add("nelectron_upper", "NELECTRON 3\n")
    add("ne_and_ncond", "Ne 5\nNcond 4\n")
    add("float_exponent_to_int", "Nsite 1.0e2\nNVMCSample 1e3\n")
    add("truncate_positive", "Nsite 2.9\nNVMCWarmUp 0.5\n")
    add("truncate_negative", "NMPTrans -2.9\nNExUpdatePath -0.9\n")
    add("hex_float_value", "Nsite 0x1c\nNe 0x1.8p2\n")
    add("explicit_plus_sign", "Nsite +6\n")
    add("double_values", "DSROptRedCut 2.5e-3\nDSROptStaDel 1.5e-5\nDSROptStepDt 0.125\nDSROptCGTol 2.5e-9\n")
    add("solver_controls", "NStore 0\nNSRCG 1\nNSROptCGMaxIter 41\nDSROptCGTol 2.5e-9\n")
    add("rbm_keys", "Nneuron 3\nNneuronCharge 4\nNneuronSpin 5\nNneuronGeneral 6\nNBlockSize_RBMRatio 64\n")
    add("itrstep_only_keeps_smp_default", "NSROptItrStep 500\n")
    add("duplicate_keys_last_wins", "Nsite 4\nNsite 8\n")
    add("tab_separators", "Nsite\t12\nNe\t6\n")
    add("blank_and_dash_lines_skipped", "\n---\nNsite 5\n\n-comment\nNe 2\n")
    add("crlf_file", FULL.replace("\n", "\r\n"), HEADER.replace("\n", "\r\n"))
    add("stale_value_equals_syntax", "NVMCSample 7\nNsite = 6\n")
    add("stale_value_non_numeric", "Ncond 4\nNsite abc\n")
    add("stale_value_missing_number", "Ne 6\nNsite\n")
    add("stale_keyword_whitespace_line", "Nsite 5\n   \nNe 3\n   \n")
    add("ex_update_path_two", "NExUpdatePath 2\n")
    add("sz_zero", "2Sz 0\n")
    add("sz_two", "2Sz 2\n")
    add("sz_minus_two", "2Sz -2\n")
    add("sz_half_truncates_to_zero", "2Sz 0.5\n")
    add("rnd_seed_negative_uses_time", "RndSeed -1\n")
    add("rnd_seed_positive", "RndSeed 12345\n")
    add("rnd_seed_fraction", "RndSeed 3.7\n")
    add("rnd_seed_minus_half_is_zero", "RndSeed -0.5\n")
    for value in (13, 5, 1, 9, 15, 23, 16, 200, 0, -3, 7, 8):
        add(f"block_size_{value}".replace("-", "neg"), f"NBlockSize_RBMRatio {value}\n")
    add("int_max", "NVMCSample 2147483647\n")
    add("heads_with_directory", "Nsite 2\n", HEADER.replace("zvo", "dir/zvo"))
    add("heads_extra_words", "Nsite 2\n", HEADER.replace("zvo", "zvo extra words"))
    add("heads_missing_word", "Nsite 2\n", HEADER.replace("CDataFileHead  zvo", "CDataFileHead"))
    add("header_line2_ignored_text", "Nsite 2\n", HEADER.replace("Model_Parameters   0", "anything goes here"))
    # Rejected inputs.
    add("reject_unknown_keyword", "Foo 1\n")
    add("reject_unknown_after_valid", "Nsite 4\nBar 2\n")
    add("reject_keyword_prefix_digits", "Nsite6 1\n")
    add("reject_alias_nelec", "NElec 4\n")
    add("reject_alias_nsite_nlocspin", "NLocSpin 2\n")
    add("reject_alias_vmccalmode", "VMCCalMode 0\n")
    add("reject_alias_lanczosmode", "LanczosMode 0\n")
    add("reject_usediagscale", "useDiagScale 1\n")
    add("reject_rescalesmat", "RescaleSmat 1\n")
    add("reject_nfileflushinterval", "NFileFlushInterval 2\n")
    add("reject_complextype", "ComplexType 1\n")
    add("reject_nsroptfixsmp", "NSROptFixSmp 1\n")
    add("reject_noneobodyg", "NOneBodyG 1\n")
    add("reject_sz_minus_one", "2Sz -1\n")
    add("reject_sz_minus_one_fraction", "2Sz -1.5\n")
    add("reject_long_line_continuation", "- " + "x" * 300 + "\n")
    add("reject_empty_file", "", "")
    add("reject_header_only_five_lines", "", "\n".join(HEADER.splitlines()[:5]) + "\n")
    add("reject_header_only_seven_lines", "", "\n".join(HEADER.splitlines()[:7]) + "\n")
    return out


def extract(reader):
    bodies = []
    for signature in ("int CheckWords(", "int ReadDefFileError(", "void SetDefaultValuesModPara(",
                      "int GetInfoFromModPara("):
        start = next(m.start() for m in re.finditer(re.escape(signature), reader)
                     if ";" not in reader[m.start():reader.index("{", m.start())])
        bodies.append(function(reader[start:], signature))
    marker = "if((bufInt[IdxNBlockSize_RBMRatio]%8)>0)"
    snippet = function(reader[reader.index(marker):], marker)
    wrapper = ("/* Wrapper added by the toolbox: the block below is verbatim from ReadDefFileNInt. */\n"
               "static void ApplyNBlockSizeRBMRatioAdjustment(int *bufInt) {\n  int itmp;\n  "
               + snippet + "\n}")
    bodies.append(wrapper)
    return "\n".join(bodies)


def parse_result(text):
    status = None
    values = []
    for line in text.splitlines():
        if line.startswith("RESULT "):
            parts = line.split(" ", 3)[1:]
            if parts[0] == "status":
                status = parts[1]
            else:
                values.append(f"{parts[0]} {parts[1]} {parts[2]}")
    return status, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    src = (args.source or root / "extern/mVMC-1.3.0") / "src/mVMC"
    reader = (src / "readdef.c").read_text()
    sources = [src / "readdef.c", src / "include/readdef.h", src / "include/global.h"]
    materialize(root, "modpara_reader_upstream.inc", extract(reader), sources, args.write)
    cc_version = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    lines = [
        "# C mVMC 1.3.0 modpara reader oracle (#344): SetDefaultValuesModPara + GetInfoFromModPara",
        "# + NBlockSize_RBMRatio adjustment (readdef.c:375-380). time(NULL) replaced by sentinel.",
        *(f"# {name} sha256={digest}" for name, digest in sha.items()),
        f"# compiler: {cc_version}; options: cc -O0 -I extern/mVMC-1.3.0/src/mVMC/include -I c_toolbox",
        f"# platform: {platform.system()} {platform.machine()} {platform.libc_ver()[0]} {platform.libc_ver()[1]}",
        "# reproduce: uv run --no-project python scripts/check_modpara_reader_c_parity.py --write",
        "# format: 'case <name> status <n>' then 'int|double|str <key> <value>' lines",
    ]
    inputs = {}
    with tempfile.TemporaryDirectory(prefix="mvmc-c-modpara-") as directory:
        tmp = Path(directory)
        subprocess.run(["cc", "-O0", "-I", str(src / "include"), "-I", str(root / "c_toolbox"),
                        str(root / "c_toolbox/modpara_reader.c"), "-o", str(tmp / "probe")], check=True)
        for name, text in variants():
            work = tmp / name
            work.mkdir()
            (work / "modpara.def").write_bytes(text.encode())
            result = subprocess.run([str(tmp / "probe"), "modpara.def"], cwd=work,
                                    capture_output=True, text=True, timeout=10)
            status, values = parse_result(result.stdout)
            assert result.returncode == 0 and status is not None, (name, result)
            lines.append(f"case {name} status {status}")
            if status != "0":
                keyword = re.search(r'keyword " (.*) " is incorrect', result.stderr)
                if keyword:
                    lines.append(f"rejected_keyword {keyword.group(1)}")
            lines.extend(values)
            inputs[name] = text
    output = "\n".join(lines) + "\n"
    fixtures = root / "tests/fixtures/modpara_reader"
    if args.write:
        (fixtures / "inputs").mkdir(parents=True, exist_ok=True)
        (fixtures / "c_reader.txt").write_text(output)
        for name, text in inputs.items():
            (fixtures / "inputs" / f"{name}.def").write_bytes(text.encode())
    else:
        assert (fixtures / "c_reader.txt").read_text() == output, "C modpara reader results changed"
        for name, text in inputs.items():
            assert (fixtures / "inputs" / f"{name}.def").read_bytes() == text.encode(), name
    print(f"{len(inputs)} C modpara reader cases checked")


if __name__ == "__main__":
    main()
