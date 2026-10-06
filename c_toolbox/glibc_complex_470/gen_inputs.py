#!/usr/bin/env python3
"""Deterministic input list for the glibc complex probe (issue #470): edge cases plus a
reproducible pseudo-random sample. Prints `<re bits> <im bits>` hex pairs."""
import math
import struct


def bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


inf, nan = math.inf, math.nan
tiny = 5e-324  # smallest subnormal
dmin = 2.2250738585072014e-308
vals = []


def add(re, im):
    vals.append((re, im))


# special values
for re in (0.0, -0.0, 1.0, -1.0, inf, -inf, nan):
    for im in (0.0, -0.0, 1.0, -1.0, inf, -inf, nan):
        add(re, im)
# large |Re z| (ccosh/cexp/ctanh overflow thresholds 709, 3*709, ctanh 354, 708)
for re in (300.0, 353.9, 354.0, 354.1, 355.0, 600.0, 708.0, 709.0, 709.5, 710.0, 1000.0,
           1418.0, 1419.0, 2127.0, 2128.0, 1e5):
    for im in (0.0, 0.3, 1.0, -2.5, 3.0, 1e-300, -1e-310):
        add(re, im)
        add(-re, im)
# near branch cuts of clog / tiny imaginary parts / signed zeros
for re in (-1e-300, -1e-5, -0.5, -1.0, -2.0, -1e5, -1e300, -dmin, -tiny):
    for im in (0.0, -0.0, tiny, -tiny, dmin, -dmin, 1e-300, -1e-300, 1e-17, -1e-17, 1e-16, 1e-15):
        add(re, im)
# clog branch coverage: |x| == 1, 1 < |x| < 2, 0.5 <= |x| < 1 (x2y2m1 path), scaling regions
for re in (1.0, 1.0 + 2.0 ** -52, 1.5, 1.9999999999999998, 2.0, 0.5, 0.5 + 2.0 ** -53,
           0.7071067811865476, 0.9, 0.99999999999999989, 0.6):
    for im in (0.0, 1e-17, 2.0 ** -53, 2.0 ** -52, 1e-8, 0.1, 0.5, 0.7, 0.8, 0.9999999999999999,
               1.0, 1.5, 3.0):
        add(re, im)
        add(im, re)
        add(-re, -im)
for re in (1e308, 1.7976931348623157e308, 8.9e307, 9e307, 1e-310, dmin, dmin / 2, tiny, 1e-320):
    for im in (0.0, 1e308, 1e-310, tiny, dmin, 1.0, 5e307):
        add(re, im)
        add(-re, im)
# pseudo-random sample, log-uniform magnitudes, uniform angles
state = 0x9E3779B97F4A7C15


def rnd():
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
    return (state >> 11) / float(1 << 53)


for _ in range(400):
    mag = math.exp((rnd() - 0.5) * 14.0)  # about 1e-3 .. 1e3
    ang = (rnd() * 2 - 1) * math.pi
    add(mag * math.cos(ang), mag * math.sin(ang))
for _ in range(200):
    # RBM-like hidden values: modest real part, arbitrary imaginary part
    add((rnd() - 0.5) * 12.0, (rnd() - 0.5) * 40.0)
for _ in range(100):
    add((rnd() - 0.5) * 1e-3, (rnd() - 0.5) * 2.0)

seen = set()
for re, im in vals:
    key = (bits(re), bits(im))
    if key in seen:
        continue
    seen.add(key)
    print(f"{key[0]:016x} {key[1]:016x}")
