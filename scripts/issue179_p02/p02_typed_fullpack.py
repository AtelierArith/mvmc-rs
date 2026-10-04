"""SOURCE P02 candidate positive direct-SR diagnostic; no FP acceptance.

Explicit actual writer mappings, original storage retained, no solver/reducer
execution. Caller must establish source/input/sidecar provenance and discrete
join BEFORE comparison. General failure/no-active/CG cases unsupported here.
"""
import math
from strict_text_records import parse_records
from p02_cross_discrete import rank_join, same

COMMON_NUMERIC = {
    'n:initial-parameters': 28, 'n:parameters': 28, 'n:energy': 10,
    'n:local-energy': 10, 'n:reduced-energy': 10,
    'n:local-oo-real': 255, 'n:reduced-oo-real': 255,
    'n:local-ho-real': 15, 'n:reduced-ho-real': 15, 'n:sr-step-0': 28,
    'n:sr-system-000000-regularization': 3,
    'n:sr-system-000000-matrix': 100, 'n:sr-system-000000-rhs': 10,
    'n:sr-system-000000-increment': 10,
}
RUST_NUMERIC = {
    'n:normalized-step-000000-energy': 10,
    'n:normalized-step-000000-oo': 255, 'n:normalized-step-000000-ho': 15,
    **{f'n:parameter-broadcast-{i}-{stage}': 28 for i in (0, 1) for stage in ('before', 'after')},
}
JULIA_NUMERIC = {
    'n:normalized-energy': 10, 'n:normalized-weight': 2,
    'n:normalized-oo-real': 225, 'n:normalized-ho-real': 15,
    'n:presync-step-0': 28,
}
CHECKED_RETURN = {
    'kind': 'CheckedSuccessfulReturn', 'julia_version': '1.13.1',
    'stdlib_sha256': 'cdb3aed84137466de0ca8c5ba303d2d4f9f60f52ba480d43dcc92689633c0af6',
    'call_boundary': 'potrs!:3328-3351/chklapackerror:34-44',
}


def vector(records, key, length):
    values = records.get(key)
    if type(values) is not list or len(values) != length:
        raise ValueError('typed pack missing/shape: ' + key)
    if key.startswith('d:'):
        if any(type(x) is not int for x in values):
            raise ValueError('integer metadata: ' + key)
    elif any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('nonfinite numeric plane: ' + key)
    return values.copy()


def expected(records, key, values):
    same(vector(records, key, len(values)), values, key)


def assemble(text, implementation, *, solve_sidecar=None):
    if implementation not in ('rust', 'julia'):
        raise ValueError('implementation domain')
    records = parse_records(text)
    shapes = COMMON_NUMERIC | (RUST_NUMERIC if implementation == 'rust' else JULIA_NUMERIC)
    if {key for key in records if key.startswith('n:')} != set(shapes):
        raise ValueError('closed actual numeric writer inventory')
    original_numeric = {key: vector(records, key, size) for key, size in shapes.items()}
    for key, value in (('d:sr-kind', [0]), ('d:sr-systems', [1]),
                       ('d:sr-system-000000-dimension', [10]),
                       ('d:sr-system-000000-not-solved', [0]),
                       ('d:sr-system-000000-status', [0]),
                       ('d:sr-system-000000-factor-info', [0])):
        expected(records, key, value)
    active = vector(records, 'd:sr-system-000000-active', 10)
    if len(set(active)) != 10 or active != sorted(active) or any(x < 0 or x >= 28 for x in active):
        raise ValueError('actual active-index domain')
    flags = vector(records, 'd:sr-system-000000-flags', 28)
    settings = vector(records, 'd:sr-system-000000-settings', 5)
    same(settings, [1, 1, 1, 0, 1], 'source input seed/steps/window/CG/store')
    if implementation == 'rust':
        expected(records, 'd:sr-system-000000-triangle', [85])
        expected(records, 'd:sr-system-000000-nrhs', [1])
        expected(records, 'd:sr-system-000000-solve-info', [0])
        expected(records, 'd:normalized-boundaries', [1])
        expected(records, 'd:normalized-step-000000-ordinal', [0])
        expected(records, 'd:normalized-step-000000-complex', [0])
        solve = {'kind': 'NativeINFO', 'info': 0}
        prefix = 'n:normalized-step-000000-'
        presync = original_numeric['n:parameter-broadcast-1-before']
    else:
        for key, value in (('solve-returned', [1]), ('solve-info-available', [0]),
                           ('solve-checked-successful-return', [1])):
            expected(records, 'd:sr-system-000000-' + key, value)
        if type(solve_sidecar) is not str or not solve_sidecar.endswith('\n'):
            raise ValueError('missing original checked-return sidecar')
        metadata = {}
        for line in solve_sidecar.splitlines():
            parts = line.split('=', 1)
            if len(parts) != 2 or parts[0] in metadata:
                raise ValueError('checked-return duplicate/malformed metadata')
            metadata[parts[0]] = parts[1]
        if metadata != CHECKED_RETURN:
            raise ValueError('checked-return source boundary provenance')
        solve = metadata
        prefix = 'n:normalized-'
        presync = original_numeric['n:presync-step-0']
    normalized_energy = original_numeric[prefix + 'energy']
    if implementation == 'julia':
        same(original_numeric['n:normalized-weight'], normalized_energy[:2],
             'same Julia captured normalized WC duplicate')
    oo = original_numeric[prefix + ('oo' if implementation == 'rust' else 'oo-real')]
    ho = original_numeric[prefix + ('ho' if implementation == 'rust' else 'ho-real')]
    planes = {
        'initial-parameters': original_numeric['n:initial-parameters'],
        'final-parameters': original_numeric['n:parameters'],
        'final-energy': original_numeric['n:energy'], 'presync': presync,
        'postsync': original_numeric['n:sr-step-0'],
        'normalized-energy': normalized_energy, 'normalized-weight': normalized_energy[:2],
        'normalized-oo': oo[:225], 'normalized-ho': ho,
        **{key[2:]: original_numeric[key] for key in COMMON_NUMERIC
           if key.startswith(('n:local-', 'n:reduced-', 'n:sr-system-'))},
    }
    # Retain all rawflags and OOtail. No fake inactive zeros or tail equality.
    return {'implementation': implementation, 'planes': planes,
            'original_numeric': original_numeric, 'normalized_oo_tail_audit': oo[225:],
            'sr': {'dimension': 10, 'storage': 'ColumnMajor', 'triangle': 'U',
                   'nrhs': 1, 'native_factor_info': 0, 'solve_observation': solve,
                   'active_indices': active, 'raw_flags': flags, 'settings': settings},
            'numeric_acceptance': False}


def diagnose(rust_text, julia_text, rank, solve_sidecar, *, written_mask):
    discrete = rank_join(rust_text, julia_text, rank)
    if type(written_mask) is not list or len(written_mask) != 28 or any(type(x) is not bool for x in written_mask):
        raise ValueError('independently source-bound C written mask required')
    same(written_mask, [i % 2 == 0 for i in range(28)], 'pinned real Gutz1/Jast1/Orbital12 written mask')
    r, j = assemble(rust_text, 'rust'), assemble(julia_text, 'julia', solve_sidecar=solve_sidecar)
    for key in ('dimension', 'storage', 'triangle', 'nrhs', 'active_indices', 'settings'):
        same(r['sr'][key], j['sr'][key], 'SR/' + key)
    for index, written in enumerate(written_mask):
        if written:
            same(r['sr']['raw_flags'][index], j['sr']['raw_flags'][index], f'C-written-flag/{index}')
            same(r['sr']['raw_flags'][index], 0 if index < 4 else 1,
                 f'original-input-written-flag/{index}')
    if set(r['planes']) != set(j['planes']):
        raise ValueError('all semantic numerical plane keys')
    differences = []
    for key in sorted(r['planes']):
        a, b = r['planes'][key], j['planes'][key]
        if len(a) != len(b):
            raise ValueError('semantic numerical plane shape')
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        differences.append({'plane': key, 'first_unequal_index': first,
                            'maximum_absolute_difference': max((abs(x-y) for x, y in zip(a, b)), default=0.0)})
    return {'discrete': discrete, 'rust': r, 'julia': j, 'differences': differences,
            'runtime_authenticated': False, 'numeric_acceptance': False,
            'first_unequal_scope': 'PerPlaneNotGlobalChronologicalCause',
            'tolerances': 'NoneAppliedDiagnosticOnly'}


def root_outputs(artifacts):
    shapes = {'zvo_out.dat': 6, 'zvo_var.dat': 48, 'zqp_opt.dat': 32}
    if type(artifacts) is not dict or set(artifacts) != set(shapes):
        raise ValueError('source-pinned complete public root inventory')
    result = {}
    for name, size in shapes.items():
        raw = artifacts[name]
        if type(raw) is not bytes:
            raise ValueError('original root output bytes required')
        text = raw.decode('ascii')
        if not text.endswith('\n') or len(text.splitlines()) != 1:
            raise ValueError('source single-window output boundary')
        values = [float(token) for token in text.split()]
        if len(values) != size or any(not math.isfinite(x) for x in values):
            raise ValueError('public root output finite/schema')
        fixed = list(range(2, 48, 3)) if name == 'zvo_var.dat' else []
        if any(values[i] != 0 for i in fixed):
            raise ValueError('C output literal zero slot changed')
        result[name] = {'original_bytes': raw, 'values': values,
                        'source_fixed_columns': fixed, 'rank': 0}
    return result
