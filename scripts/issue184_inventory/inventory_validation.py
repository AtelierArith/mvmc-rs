"""Strict inventory structure/source checks; never semantic execution proof."""
import hashlib
import re

REVISIONS = {'Julia-mVMC': '8bb1b9e8ae47b1512c00b321be05664ddcac0fd1',
    'PfaPack': '0dcf52c15caec63516d0703f36bfc8a4bc0e58d0',
    'SFMT': '1526553009f318ae78338151460fda78beadddc2'}

def source_package(source):
    return 'PfaPack' if source.startswith('PfaPack.jl/') else 'SFMT' if source.startswith('SFMT.jl/') else 'Julia-mVMC'

def validate_sources(rows, blobs):
    files = [r for r in rows if r['kind'] == 'file']
    names = [r['source'] for r in files]
    if len(names) != len(set(names)) or set(names) != set(blobs):
        raise ValueError('duplicate/missing/extra source inventory')
    declarations = {r['source']: r for r in files}
    for row in rows:
        if row['kind'] not in {'file','export','testset','include','import_or_alias','binding','method','type'}:
            raise ValueError('unknown AST record kind')
        source = row['source']
        package = source_package(source)
        if row.get('package') != package or row.get('revision') != REVISIONS[package]:
            raise ValueError('wrong declared package/revision')
        if source not in blobs:
            raise ValueError('unknown source')
        blob = blobs[source]
        expected = hashlib.sha256(blob).hexdigest()
        if row['sha256'] != expected or row['sha256'] != declarations[source]['sha256']:
            raise ValueError('source hash mismatch')
        value = row['line']
        if not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]*', value):
            raise ValueError('invalid source line integer')
        if int(value) > max(1, len(blob.decode('utf-8').splitlines())):
            raise ValueError('source line outside exact file')
    return 'SourceBoundOnly'

def ledger_anchors(row, blobs):
    result = []
    for item in row['julia_source'].split(';'):
        match = re.fullmatch(r'\s*(.+\.jl)(?::(.*))?\s*', item)
        if not match:
            raise ValueError('malformed source anchor')
        source, suffix = match.groups()
        source = source.strip()
        if source not in blobs:
            raise ValueError('ledger source missing from exact revision')
        if suffix is None:
            result.append({'source':source,'start':None,'end':None,'classification':'FileOnly'})
            continue
        suffix = suffix.strip()
        if row['id'] in {'S454','S455','S456'} and suffix == 'rank0-only stdout / failure scenarios':
            result.append({'source':source,'start':None,'end':None,'classification':'ExplicitReviewedDescriptorOnly'})
            continue
        for part in suffix.split(','):
            line = re.fullmatch(r'([1-9][0-9]*)(?:-([1-9][0-9]*))?',part.strip())
            if not line:
                raise ValueError('malformed numeric source anchor')
            start = int(line[1]); end = int(line[2] or line[1])
            if not 1 <= start <= end <= len(blobs[source].decode('utf-8').splitlines()):
                raise ValueError('ledger line/range outside exact source')
            result.append({'source':source,'start':start,'end':end,'classification':'ExactNumericAnchor'})
    return result

def validate_rows(rows, prefix, expected_count):
    if len(rows) != expected_count:
        raise ValueError('missing/extra ledger row')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate ledger identity')
    if set(ids) != {f'{prefix}{i:03}' for i in range(1, expected_count + 1)}:
        raise ValueError('missing/wrong ledger identity')
    for row in rows:
        for field in ('julia_source', 'julia_symbol_or_scenario', 'owner', 'rust_entry',
                      'settings', 'command', 'result', 'authority_or_intentional_difference'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError('missing classification field: ' + field)
    return 'InventoryOnly; not runtime or semantic proof'
