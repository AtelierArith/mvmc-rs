"""Semantic comparisons for optional, independently generated references.

Only caller-classified computed fields are approximate. Everything else,
including inputs, integer controls and source provenance, remains exact.
"""
import math
import re
import struct

GREEN = (1e-13, 1e-13)
ENERGY = (1e-12, 1e-12)
MEASUREMENT = (2e-13, 2e-13)
# Division/libm: 64 binary64 rounding units, with four minimum subnormal ulps.
SCALAR = (4 * math.ulp(0.0), 64 * math.ulp(1.0))


def decode_hex(word):
    assert re.fullmatch(r"[0-9a-fA-F]{16}", word), f"Invalid binary64 encoding: {word}"
    return struct.unpack(">d", bytes.fromhex(word))[0]


def within(actual, expected, abs_tol, rel_tol):
    assert math.isfinite(abs_tol) and abs_tol >= 0
    assert math.isfinite(rel_tol) and rel_tol >= 0
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    scale = max(abs(actual), abs(expected))
    difference = abs(actual - expected)
    if not math.isfinite(difference):
        return abs(actual / scale - expected / scale) <= abs_tol / scale + rel_tol
    return difference <= abs_tol + rel_tol * scale


def assert_values_close(actual, expected, abs_tol, rel_tol, context="computed values"):
    actual, expected = list(actual), list(expected)
    assert len(actual) == len(expected), f"{context}: changed cardinality"
    for index, (a, e) in enumerate(zip(actual, expected)):
        assert within(a, e, abs_tol, rel_tol), (
            f"{context}[{index}]: {a!r} != {e!r}; abs={abs_tol} rel={rel_tol}"
        )


def compare_text(actual, expected, computed_fields):
    """Selector (data_row, column, row_tokens) returns (abs,rel), or None.

    Rows are zero-based excluding comments/blank lines. Comments and blank
    lines retain exact positions/content; source identifiers never disappear.
    """
    a_lines, e_lines = actual.splitlines(), expected.splitlines()
    assert len(a_lines) == len(e_lines), "Changed reference line count"
    row = 0
    for line, (a, e) in enumerate(zip(a_lines, e_lines), 1):
        if not e.strip() or e.lstrip().startswith("#"):
            assert a == e, f"Changed reference header/structure on line {line}"
            continue
        aa, ee = a.split(), e.split()
        assert len(aa) == len(ee), f"Changed token count on line {line}"
        for column, (av, ev) in enumerate(zip(aa, ee)):
            bounds = computed_fields(row, column, ee)
            if bounds is None:
                assert av == ev, f"Changed exact field at line {line}, column {column}"
            else:
                assert_values_close([decode_hex(av)], [decode_hex(ev)], *bounds,
                                    context=f"line {line}, column {column}")
        row += 1
    return row
