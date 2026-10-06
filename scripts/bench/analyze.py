#!/usr/bin/env python3
"""Merge function-suite result archives into one Markdown report (issue #450).

    uv run --no-project scripts/bench/analyze.py ARCHIVE_OR_DIR [ARCHIVE_OR_DIR ...] [--out report.md]

Stdlib only. Each argument is a results directory or a results-*.tar.gz written by
scripts/bench/run_all.sh; every argument is one machine. Sections, in order:

  1. Numerical validation (the primary result): PASS/FAIL per function and size against the
     C-order CPU oracle, summarized across machines (worst error/bound ratio per function).
  2. Timing (reference): speedup tables and break-even sizes.
  3. GPU-ize? recommendation per function (gated by the numerical verdict).
  4. Not available / skipped.
"""
import argparse
import csv
import io
import os
import sys
import tarfile
from collections import defaultdict

GPU_PREFIX = ("cuda", "tenferro-cuda")
FAMILY_FILES = ["pfaffian", "sr", "sr_cpu_1core", "sr_resident", "sampler", "transfers"]


def load(path):
    """Return (metadata dict, rows list) of one archive or directory."""
    files = {}
    if os.path.isdir(path):
        base = path
        for sub in ("csv",):
            d = os.path.join(base, sub)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                files[f] = open(os.path.join(d, f), newline="").read()
        meta_text = open(os.path.join(base, "metadata.txt")).read() if os.path.exists(os.path.join(base, "metadata.txt")) else ""
    else:
        with tarfile.open(path) as tf:
            meta_text = ""
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                data = tf.extractfile(m).read().decode()
                if m.name.endswith("metadata.txt"):
                    meta_text = data
                elif "/csv/" in m.name and m.name.endswith(".csv"):
                    files[os.path.basename(m.name)] = data
    meta = {}
    for line in meta_text.splitlines():
        if line.startswith("---"):
            break
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    rows = []
    for name, text in files.items():
        for r in csv.DictReader(io.StringIO(text)):
            r["_file"] = name
            rows.append(r)
    return meta, rows


def fnum(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def params(r):
    out = {}
    for kv in (r.get("params") or "").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def is_gpu(variant):
    return variant.startswith(GPU_PREFIX)


def label(meta, path):
    return "%s (%s)" % (meta.get("gpu_model", "?"), meta.get("host", os.path.basename(path)))


def md_table(header, rows):
    if not rows:
        return "_none_\n"
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def sci(x):
    return "-" if x is None else "%.2e" % x


def tsec(x):
    if x is None:
        return "-"
    if x < 1e-3:
        return "%.1f us" % (x * 1e6)
    if x < 1:
        return "%.2f ms" % (x * 1e3)
    return "%.2f s" % x


# ------------------------------------------------------------------ numerical validation
def numerical_section(machines):
    out = ["## 1. Numerical validation\n"]
    out.append(
        "Primary result. Every GPU/tenferro/CPU variant is compared with the C-order CPU oracle "
        "(pfapack / `COrderSr` / CPU sampler) using an explicit bound per function (see "
        "`benchmark/function_suite/README.md`). `ratio` = observed deviation / bound; PASS needs "
        "ratio <= 1. RNG state, draw counts and configurations are compared exactly.\n"
    )
    # verdict counts per machine
    hdr = ["machine", "PASS", "FAIL", "ERROR", "NotAvailable", "SKIPPED", "overall"]
    rows = []
    for name, meta, rs in machines:
        c = defaultdict(int)
        for r in rs:
            c[r["verdict"]] += 1
        overall = "**FAIL**" if c["FAIL"] + c["ERROR"] else ("PASS" if c["PASS"] else "no checks")
        rows.append([name, c["PASS"], c["FAIL"], c["ERROR"], c["NotAvailable"], c["SKIPPED"], overall])
    out.append(md_table(hdr, rows))

    # failures
    fails = []
    for name, meta, rs in machines:
        for r in rs:
            if r["verdict"] in ("FAIL", "ERROR"):
                fails.append([name, r["family"], r["function"], r["variant"], r["dtype"], r["params"],
                              r["verdict"], sci(fnum(r["dev_ratio"])), r["note"][:160]])
    out.append("\n### Failures\n")
    out.append(md_table(["machine", "family", "function", "variant", "dtype", "params", "verdict", "ratio", "note"], fails))

    # worst ratio per (function, variant, dtype) across machines
    worst = {}
    for mi, (name, meta, rs) in enumerate(machines):
        for r in rs:
            ratio = fnum(r["dev_ratio"])
            if r["verdict"] in ("PASS", "FAIL") and ratio is not None:
                key = (r["family"], r["function"], r["variant"], r["dtype"])
                d = worst.setdefault(key, {})
                prev = d.get(mi)
                if prev is None or ratio > prev[0]:
                    d[mi] = (ratio, r["params"], r["verdict"])
    out.append("\n### Worst deviation/bound per function and variant, by machine\n")
    out.append("Values are the worst ratio over all sizes (its parameters in parentheses); a cell is `FAIL` when any size fails.\n\n")
    hdr = ["family", "function", "variant", "dtype"] + [m[0] for m in machines]
    rows = []
    for key in sorted(worst):
        cells = []
        for mi in range(len(machines)):
            v = worst[key].get(mi)
            if v is None:
                cells.append("-")
            else:
                bad = any(r["verdict"] == "FAIL" for r in machines[mi][2]
                          if (r["family"], r["function"], r["variant"], r["dtype"]) == key)
                cells.append("%s%.2e (%s)" % ("FAIL " if bad else "", v[0], (v[1] or "")[:60]))
        rows.append(list(key) + cells)
    out.append(md_table(hdr, rows))

    # exactness checks without ratio (rows with PASS and no dev)
    ex = []
    for name, meta, rs in machines:
        for r in rs:
            if r["verdict"] == "PASS" and fnum(r["dev_ratio"]) is None:
                ex.append([name, r["function"], r["variant"], r["dtype"], r["params"], r["note"][:120]])
    out.append("\n### Exact-match checks (no tolerance)\n")
    out.append(md_table(["machine", "function", "variant", "dtype", "params", "note"], ex))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ timing
def index(rs, family, function):
    return [r for r in rs if r["family"] == family and r["function"] == function and fnum(r["median_s"]) is not None]


def timing_pfaffian(rs):
    """Per-plane time; returns table rows and per-(dtype,n) break-even in batch B."""
    rows, table = [], defaultdict(dict)
    for r in index(rs, "pfaffian", "pfaffian_inverse"):
        p = params(r)
        planes = int(p.get("planes", 0)) or 1
        key = (r["dtype"], int(p["n"]), int(p["B"]))
        table[key][r["variant"]] = fnum(r["median_s"]) / planes
    be = defaultdict(dict)
    for (dt, n, b), v in sorted(table.items()):
        one, ray = v.get("cpu-pfapack-1thread"), v.get("cpu-pfapack-rayon")
        tot, ker = v.get("cuda-total-with-transfers"), v.get("cuda-kernel-only")
        sp = lambda base, x: "-" if not base or not x else "%.1fx" % (base / x)
        rows.append([dt, n, b, tsec(one), tsec(ray), tsec(tot), tsec(ker), sp(one, tot), sp(ray, tot), sp(ray, ker)])
        if ray and tot and tot < ray and "all" not in be[(dt, n)]:
            be[(dt, n)]["all"] = b
        if one and tot and tot < one and "one" not in be[(dt, n)]:
            be[(dt, n)]["one"] = b
    return rows, be, table


def timing_sr(rs):
    t = defaultdict(dict)
    for r in rs:
        if r["family"] != "sr" or fnum(r["median_s"]) is None:
            continue
        p = params(r)
        if "NPara" not in p:
            continue
        key = (r["function"], int(p["NPara"]), int(p["samples"]))
        v = r["variant"]
        v = {"cpu-corder-allcores": "cpu_all", "cpu-corder-1core": "cpu_1", "tenferro-cuda": "gpu",
             "tenferro-cpu-faer": "faer"}.get(v, v)
        t[key][v] = fnum(r["median_s"])
    return t


def timing_sampler(rs):
    t = defaultdict(dict)
    for r in index(rs, "sampler", "sampler_end_to_end"):
        p = params(r)
        t[(int(p["L"]), int(p["W"]))][r["variant"]] = fnum(r["median_s"])
    return t


def timing_section(machines):
    out = ["## 2. Timing (reference)\n"]
    out.append("Secondary result: medians after warm-up. CUDA times include host-device transfers unless the "
               "row says kernel-only. Speedup > 1 means the GPU is faster. Timings are only meaningful for variants "
               "whose numerical verdict in section 1 is PASS.\n")
    for name, meta, rs in machines:
        out.append("\n### %s\n" % name)
        rows, be, _ = timing_pfaffian(rs)
        out.append("\n#### Batched Pfaffian + inverse (time per plane, NQP = 8 planes per batch element)\n")
        out.append(md_table(["dtype", "n", "B", "1 thread", "all cores (rayon)", "CUDA total", "CUDA kernel",
                             "x vs 1 thread", "x vs all cores", "kernel x vs all cores"], rows))
        out.append("\nBreak-even (smallest B where CUDA incl. transfers beats the CPU): %s\n" % (
            "; ".join("%s n=%d: all cores B>=%s, 1 thread B>=%s" % (dt, n, d.get("all", "never"), d.get("one", "never"))
                      for (dt, n), d in sorted(be.items())) or "-"))
        t = timing_sr(rs)
        out.append("\n#### SR stages\n")
        rows = []
        for (fn, n, k), v in sorted(t.items()):
            g, a, o = v.get("gpu"), v.get("cpu_all"), v.get("cpu_1")
            rows.append([fn, n, k, tsec(o), tsec(a), tsec(v.get("faer")), tsec(g),
                         "-" if not (a and g) else "%.1fx" % (a / g), "-" if not (o and g) else "%.1fx" % (o / g)])
        out.append(md_table(["stage", "NPara", "samples", "C-order 1 core", "C-order all cores", "tenferro CPU",
                             "tenferro CUDA", "x vs all cores", "x vs 1 core"], rows))
        ts = timing_sampler(rs)
        out.append("\n#### Sampler end to end (one `VMCMakeSample` call per walker)\n")
        rows = []
        for (l, w), v in sorted(ts.items()):
            m, one, g = v.get("cpu-multichain"), v.get("cpu-1thread"), v.get("cuda-pinned")
            rows.append([l, w, tsec(one), tsec(m), tsec(g), tsec(v.get("cuda-pageable")),
                         "-" if not (m and g) else "%.2fx" % (m / g)])
        out.append(md_table(["L", "W", "CPU 1 thread", "CPU multichain", "CUDA pinned", "CUDA pageable", "x vs multichain"], rows))
        st = []
        for r in rs:
            if r["family"] == "sampler" and r["function"].startswith("sampler_") and r["function"] != "sampler_end_to_end" \
                    and r["variant"] == "cuda-pinned" and fnum(r["median_s"]) is not None and "W" in params(r):
                st.append([r["function"], params(r)["L"], params(r)["W"], tsec(fnum(r["median_s"]))])
        if st:
            out.append("\n#### Sampler stages (device-event time of one profiled call)\n")
            out.append(md_table(["stage", "L", "W", "time"], st))
        tr = []
        for r in rs:
            if r["family"] == "transfers" and r["function"] == "transfer_time" and fnum(r["median_s"]) is not None:
                b = int(params(r)["bytes"])
                if b in (32768, 1 << 20, 16 << 20, 256 << 20):
                    tr.append([b, r["variant"], tsec(fnum(r["median_s"])), "%.1f GB/s" % (b / fnum(r["median_s"]) / 1e9)])
        out.append("\n#### Transfers\n")
        out.append(md_table(["bytes", "path", "time", "bandwidth"], tr))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ recommendation
def has_gpu_fail(rs, family, prefixes):
    return any(r["verdict"] in ("FAIL", "ERROR") and r["family"] == family and is_gpu(r["variant"])
               and any(r["function"].startswith(p) for p in prefixes) for r in rs)


def recommend(machines):
    out = ["## 3. GPU-ize? recommendation\n"]
    out.append("Rule: a function is only recommended when its GPU variant passes every numerical check on every machine. "
               "Speed classes (vs the best CPU baseline, all cores, transfers included): >= 2x at a realistic size = YES; "
               "1x-2x = MARGINAL (depends on host core count and transfer overlap); < 1x = NO. Break-even sizes are "
               "per machine.\n")
    rows = []
    # (family, label, function prefixes, numeric-fail prefixes)
    for name, meta, rs in machines:
        peak = meta.get("gpu_fp64_peak_tflops", "unknown")
        # Pfaffian
        _, be, table = timing_pfaffian(rs)
        best = {}
        for (dt, n, b), v in table.items():
            ray, tot = v.get("cpu-pfapack-rayon"), v.get("cuda-total-with-transfers")
            ker = v.get("cuda-kernel-only")
            if ray and tot:
                sp = ray / tot
                if (dt, n) not in best or sp > best[(dt, n)][0]:
                    best[(dt, n)] = (sp, b, ker and tot and tot / ker)
        for (dt, n), (sp, b, ratio) in sorted(best.items()):
            bad = has_gpu_fail(rs, "pfaffian", ["pfaffian_inverse"])
            rows.append([name, "batched Pfaffian+inverse %s n=%d" % (dt, n), "FAIL" if bad else "PASS",
                         "%.1fx (B=%d)" % (sp, b), be.get((dt, n), {}).get("all", "never"),
                         verdict(bad, sp),
                         "transfers are %.1fx the kernel time; FP64 peak %s TFLOPS" % (ratio, peak) if ratio else ""])
        t = timing_sr(rs)
        for fn in ("gram", "assemble_s_g", "cholesky_solve", "cg_matvec", "cg_solve"):
            best_sp, at, first = None, None, None
            for (f, n, k), v in sorted(t.items()):
                if f != fn or not (v.get("gpu") and v.get("cpu_all")):
                    continue
                sp = v["cpu_all"] / v["gpu"]
                if sp > 1 and first is None:
                    first = n
                if best_sp is None or sp > best_sp:
                    best_sp, at = sp, (n, k)
            if best_sp is None:
                continue
            bad = has_gpu_fail(rs, "sr", [fn])
            rows.append([name, "SR %s" % fn, "FAIL" if bad else "PASS", "%.1fx (NPara=%d, samples=%d)" % (best_sp, at[0], at[1]),
                         first if first else "never", verdict(bad, best_sp),
                         "per-call upload/download of the stage operands included" if fn != "cg_matvec" else "constant operand uploaded once per solve"])
        ts = timing_sampler(rs)
        per_l = defaultdict(list)
        for (l, w), v in sorted(ts.items()):
            if v.get("cpu-multichain") and v.get("cuda-pinned"):
                per_l[l].append((w, v["cpu-multichain"] / v["cuda-pinned"]))
        bad = has_gpu_fail(rs, "sampler", ["sampler_teacher_forced", "sampler_free_run", "sampler_resident"])
        for l, lst in sorted(per_l.items()):
            w_best, sp = max(lst, key=lambda x: x[1])
            first = next((w for w, s in lst if s > 1), None)
            rows.append([name, "sampler lock-step L=%d" % l, "FAIL" if bad else "PASS", "%.2fx (W=%d)" % (sp, w_best),
                         first if first else "never", verdict(bad, sp),
                         "vs CPU multichain on %s cores" % meta.get("cpu_available_cores", "?")])
    out.append(md_table(["machine", "function", "numerics", "best speedup vs CPU all cores", "break-even (B / NPara / W)",
                         "GPU-ize?", "reasoning"], rows))
    return "\n".join(out) + "\n"


def verdict(bad, sp):
    if bad:
        return "NO (fix numerics first)"
    if sp >= 2:
        return "YES"
    if sp >= 1:
        return "MARGINAL"
    return "NO"


def skipped_section(machines):
    out = ["## 4. Not available and skipped\n"]
    rows = []
    for name, meta, rs in machines:
        for r in rs:
            if r["verdict"] in ("NotAvailable", "SKIPPED"):
                rows.append([name, r["family"], r["function"], r["variant"], r["params"], r["verdict"], r["note"]])
    out.append(md_table(["machine", "family", "function", "variant", "params", "status", "reason"], rows))
    return "\n".join(out) + "\n"


def metadata_section(machines):
    keys = ["host", "profile", "git_rev", "mode", "gpu_model", "gpu_compute_capability", "gpu_memory_total", "gpu_driver",
            "gpu_fp64_peak_tflops", "cuda_toolkit", "cpu_model", "cpu_logical_cores", "cpu_available_cores", "os", "rustc",
            "crate_tenferro-gpu", "crate_cudarc", "env_OMP_NUM_THREADS", "env_OPENBLAS_NUM_THREADS", "env_RAYON_NUM_THREADS"]
    out = ["## Metadata\n"]
    out.append(md_table(["key"] + [m[0] for m in machines], [[k] + [m[1].get(k, "-") for m in machines] for k in keys]))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    machines = []
    for p in a.inputs:
        meta, rows = load(p)
        if not rows:
            sys.exit("no CSV rows found in %s" % p)
        machines.append((label(meta, p), meta, rows))
    text = "# mvmc-rs function-level GPU/CPU suite report\n\n"
    text += numerical_section(machines) + "\n" + timing_section(machines) + "\n" + recommend(machines) + "\n" \
        + skipped_section(machines) + "\n" + metadata_section(machines)
    if a.out:
        open(a.out, "w").write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
