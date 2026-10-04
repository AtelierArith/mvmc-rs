"""SOURCE-only ingestion. No execution, inferred observations, or numeric budgets.

Both reviewed recorders emit whitespace-separated d:/n: records. Empty buffers
are omitted by Rust: absence is NOT interpreted as an observed empty buffer.
"""
import math
import re
from decimal import Decimal


def parse_records(text):
    if type(text) is not str or not text or not text.endswith("\n"):
        raise ValueError("missing/truncated record text")
    records = {}
    for ordinal, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if len(fields) < 2 or not re.fullmatch(r"[dn]:[A-Za-z0-9_-]+", fields[0]):
            raise ValueError(f"invalid record at line {ordinal}")
        key = fields[0]
        if key in records:
            raise ValueError(f"duplicate record {key}")
        values = []
        for token in fields[1:]:
            if key.startswith("d:"):
                if not re.fullmatch(r"-?(0|[1-9][0-9]*)", token):
                    raise ValueError(f"noninteger discrete record {key}")
                value = int(token)
                if not -(2**63) <= value < 2**63:
                    raise ValueError(f"out-of-range discrete record {key}")
                values.append(value)
            else:
                if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token):
                    raise ValueError(f"invalid numerical record {key}")
                value = float(token)
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite numerical record {key}")
                if value == 0.0 and Decimal(token) != 0:
                    raise ValueError(f"underflowed numerical token {key}")
                values.append(value)
        records[key] = values
    return records


def require_legacy(records):
    """Check default key/event contract, not model success or numeric accuracy."""
    for key in records:
        if (key in ("d:trace-schema", "d:requested-record-schema",
                    "d:normalized-boundaries") or "normalized-step-" in key or
                re.fullmatch(r"d:checkpoint-[0-9]{6}-(raw624|cursor|native-words-consumed)", key)):
            raise ValueError(f"raw-v2 key in legacy output: {key}")
    count = records.get("d:trace-events")
    if count is None or len(count) != 1 or count[0] <= 0:
        raise ValueError("missing/nonpositive actual trace inventory")
    expected = {f"d:trace-{i:06d}" for i in range(count[0])}
    actual = {key for key in records if re.fullmatch(r"d:trace-[0-9]{6}", key)}
    if actual != expected:
        raise ValueError("incomplete/extra trace inventory")
    for key in expected:
        if records[key][0] not in range(10):
            raise ValueError("nonlegacy event kind")
    return records
