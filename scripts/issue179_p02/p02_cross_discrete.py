"""P02 SOURCE candidate, original utility adapted: strict discrete join, no numerical gate.

Inputs are original ON rank text, not reconstructed/reseeded expectations.
Receipt/source/provider verification must precede this utility in its caller.
Julia counts are observed primitive words; native count stays Unavailable.
"""
from strict_text_records import parse_records
from p02_discrete_schema import RUST_REQUIRED, require_original_rust, require_original_julia


# Literal discrete namespaces from frozen Rust263fd2/Julia81835 writers.
# This stage is NOT a complete numeric/SR schema validator: numeric records
# and known solver metadata are retained for the separately typed fullpack.
COMMON_METADATA = {
    'd:sr-kind', 'd:sr-systems',
    *('d:sr-system-000000-' + suffix for suffix in (
        'dimension', 'not-solved', 'status', 'factor-info', 'active', 'flags', 'settings')),
}
RUST_METADATA = {
    'd:trace-schema', 'd:requested-record-schema', 'd:normalized-boundaries',
    'd:normalized-step-000000-ordinal', 'd:normalized-step-000000-complex',
    'd:main-thread-collective-calls', 'd:parameter-broadcast-complete',
    'd:parameter-broadcast-0-frame', 'd:parameter-broadcast-1-frame',
    'd:sr-system-000000-triangle', 'd:sr-system-000000-nrhs',
    'd:sr-system-000000-solve-info',
}
JULIA_METADATA = {
    'd:sr-system-000000-solve-returned',
    'd:sr-system-000000-solve-info-available',
    'd:sr-system-000000-solve-checked-successful-return',
}


def inventory(records, implementation):
    shared = {key for key in RUST_REQUIRED
              if key.startswith('d:')}
    shared -= {'d:before-init-raw-rng', 'd:before-init-rng-cursor',
               'd:initial-raw-rng', 'd:initial-rng-cursor',
               'd:checkpoint-000000-raw624', 'd:checkpoint-000000-cursor',
               'd:checkpoint-000000-native-words-consumed', 'd:raw-rng', 'd:rng-cursor'}
    allowed = shared | COMMON_METADATA | {'d:local-counter', 'd:reduced-counter'}
    if implementation == 'rust':
        allowed |= {key for key in RUST_REQUIRED if key.startswith('d:')} | RUST_METADATA
    else:
        allowed |= JULIA_METADATA | {
            f'd:{label}-{suffix}'
            for label in ('seeded', 'initialized', 'checkpoint-0', 'final')
            for suffix in ('raw624', 'cursor', 'initialized', 'raw-abi', 'observed-primitive-words')
        }
    allowed |= {f'd:trace-{i:06}' for i in range(records['d:trace-events'][0])}
    unknown = {key for key in records if key.startswith('d:')} - allowed
    if unknown:
        raise ValueError('unknown discrete writer keys: ' + repr(sorted(unknown)))
    for key in ('d:local-counter', 'd:reduced-counter'):
        if key not in records or len(records[key]) != 10:
            raise ValueError('missing/truncated integer counter plane: ' + key)


def same(left, right, label):
    if type(left) is not type(right):
        raise ValueError('discrete type mismatch: ' + label)
    if type(left) is list:
        if len(left) != len(right):
            raise ValueError('discrete length mismatch: ' + label)
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, f'{label}/{index}')
    elif left != right:
        raise ValueError('discrete mismatch: ' + label)


def rank_join(rust_text, julia_text, rank):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError('P02 rank domain')
    r, j = parse_records(rust_text), parse_records(julia_text)
    require_original_rust(r)
    require_original_julia(j, 'Unavailable')
    inventory(r, 'rust')
    inventory(j, 'julia')
    for records in (r, j):
        for key, value in (
            ('d:seed', [rank + 1]), ('d:group', [rank, 0, 1]),
            ('d:steps', [1]), ('d:status', [0]),
            ('d:configured-samples', [3]), ('d:chain-samples', [3]),
            ('d:requested-width', [1]),
        ):
            same(records[key], value, key)
    compared = []
    for boundary, raw_key, cursor_key, count_key, julia_label in (
        ('seeded', 'before-init-raw-rng', 'before-init-rng-cursor', None, 'seeded'),
        ('initialized', 'initial-raw-rng', 'initial-rng-cursor', 'initial-draw-count', 'initialized'),
        ('checkpoint0', 'checkpoint-000000-raw624', 'checkpoint-000000-cursor',
         'checkpoint-000000-native-words-consumed', 'checkpoint-0'),
        ('final', 'raw-rng', 'rng-cursor', 'total-draw-count', 'final'),
    ):
        same(r['d:' + raw_key], j[f'd:{julia_label}-raw624'], boundary + '/raw624')
        same(r['d:' + cursor_key], j[f'd:{julia_label}-cursor'], boundary + '/cursor')
        # No seeded Rust counter field is emitted. Do not invent a native zero.
        if count_key is not None:
            same(r['d:' + count_key], j[f'd:{julia_label}-observed-primitive-words'],
                 boundary + '/native-vs-observed-words')
        compared.append(boundary)
    for key in ('initial-rng', 'rng', 'initial-draw-count', 'total-draw-count',
                'sampling-draw-count', 'acceptance-events', 'ele_idx', 'ele_cfg',
                'ele_num', 'ele_proj_cnt', 'counter', 'local-counter', 'reduced-counter'):
        same(r['d:' + key], j['d:' + key], key)
    revents = [r[f'd:trace-{i:06}'] for i in range(r['d:trace-events'][0])]
    jevents = [j[f'd:trace-{i:06}'] for i in range(j['d:trace-events'][0])]
    # require_original_rust independently enforces exactly ONE terminal kind10
    # directly after kind8 and binds it to actual raw/cursor/native-count fields.
    # Julia schema has no kind10. All other events must match exactly, in order.
    same(revents[:-1], jevents, 'ordered-proposals-decisions-draws-saved-future624')
    return {'rank': rank, 'raw_boundaries': compared,
            'semantic_events': len(jevents), 'discrete_equal': True,
            'julia_native_draw_counter': 'Unavailable',
            'seeded_rust_native_count_field': 'NotEmitted',
            'count_comparison': 'NativeWordsConsumed-vs-ObservedPrimitiveWords',
            'retained_numeric_keys': {'rust': sorted(k for k in r if k.startswith('n:')),
                                      'julia': sorted(k for k in j if k.startswith('n:'))},
            'solver_metadata_validation': 'DeferredToTypedFullpack',
            'numerical_acceptance': False}


def join(rust_rank_texts, julia_rank_texts):
    if type(rust_rank_texts) is not list or type(julia_rank_texts) is not list:
        raise ValueError('rank arrays required')
    if len(rust_rank_texts) != 2 or len(julia_rank_texts) != 2:
        raise ValueError('complete WORLD2 rank inventory')
    return [rank_join(rust_rank_texts[i], julia_rank_texts[i], i) for i in range(2)]
