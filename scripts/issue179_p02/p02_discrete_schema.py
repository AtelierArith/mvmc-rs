"""P02 SOURCE schema: saved buffers bind samples3 x six-site layout; not executed."""
from strict_text_records import parse_records
import math

PUBLIC_OUTPUT_COLUMNS = {'zvo_out.dat': 6, 'zvo_var.dat': 48, 'zqp_opt.dat': 32}


def require_public_outputs(original, observed, required):
    if type(required) is not list or set(required) != set(PUBLIC_OUTPUT_COLUMNS) or len(required) != 3:
        raise ValueError('source-pinned P01 public output names')
    for outputs in (original, observed):
        if set(outputs) != set(PUBLIC_OUTPUT_COLUMNS):
            raise ValueError('missing/extra public output artifact')
        for name, columns in PUBLIC_OUTPUT_COLUMNS.items():
            data = outputs[name]
            if type(data) is not bytes:
                raise ValueError('public output must be original bytes')
            text = data.decode('ascii')
            if not text.endswith('\n') or len(text.splitlines()) != 1 or len(text.split()) != columns:
                raise ValueError('source-pinned single-window public output shape')
            if any(not math.isfinite(float(token)) for token in text.split()):
                raise ValueError('nonfinite public output')
    if original != observed:
        raise ValueError('same-implementation public output bytes changed')

# Mandatory original writer keys, not learned from either observed run.
# Historical798 mpi_issue179_state.rs: initialization/checkpoint/final writers.
RUST_REQUIRED = {
    'd:seed': 1, 'd:group': 3, 'd:steps': 1, 'd:status': 1,
    'd:configured-samples': 1, 'd:chain-samples': 1, 'd:requested-width': 1,
    'd:before-init-raw-rng': 624, 'd:before-init-rng-cursor': 1,
    'd:initial-raw-rng': 624, 'd:initial-rng-cursor': 1,
    'd:initial-rng': 624, 'd:initial-draw-count': 1,
    'd:checkpoint-000000-raw624': 624, 'd:checkpoint-000000-cursor': 1,
    'd:checkpoint-000000-native-words-consumed': 1,
    'd:raw-rng': 624, 'd:rng-cursor': 1, 'd:rng': 624,
    'd:total-draw-count': 1, 'd:sampling-draw-count': 1,
    'd:trace-events': 1, 'd:acceptance-events': 2,
    'd:ele_idx': 18, 'd:ele_cfg': 36, 'd:ele_num': 36,
    'd:ele_proj_cnt': 6, 'd:counter': 10,
    'n:initial-parameters': 28, 'n:parameters': 28,
}


def require_original_rust(records):
    if 'd:ele_spn' in records:
        raise ValueError('ordinary conserved-spin writer omits empty spin plane')
    for key, size in RUST_REQUIRED.items():
        if key not in records or len(records[key]) != size:
            raise ValueError('missing/truncated independent original writer field: ' + key)
    if records['d:steps'] != [1] or records['d:status'] != [0]:
        raise ValueError('P01 successful prefix1 required')
    count = records['d:trace-events'][0]
    if type(count) is not int or count <= 0:
        raise ValueError('actual trace event count')
    keys = {key for key in records if key.startswith('d:trace-') and key not in {'d:trace-events', 'd:trace-schema'}}
    if keys != {f'd:trace-{index:06}' for index in range(count)}:
        raise ValueError('complete contiguous original trace inventory')
    if records.get('d:trace-schema') != [2] or records.get('d:requested-record-schema') != [2]:
        raise ValueError('pinned Rust raw-v2 schema required')
    validate_trace(records, raw_extension=True)


def validate_trace(records, *, raw_extension):
    count = records.get('d:trace-events')
    if count is None or len(count) != 1 or count[0] <= 0:
        raise ValueError('missing actual trace count')
    expected = {f'd:trace-{i:06}' for i in range(count[0])}
    actual = {k for k in records if k.startswith('d:trace-') and k not in {'d:trace-events', 'd:trace-schema'}}
    if actual != expected:
        raise ValueError('trace framing inventory')
    checkpoints = 0
    raw_frames = 0
    for i in range(count[0]):
        event = records[f'd:trace-{i:06}']
        if not event:
            raise ValueError('empty trace event')
        kind = event[0]
        if kind in (0, 1, 2, 7, 9):
            if len(event) != {0: 2, 1: 6, 2: 8, 7: 3, 9: 2}[kind]:
                raise ValueError('pinned ordinary sampling event shape')
            # Pinned NExUpdatePath=2, ordinary orbital: Exchange (code1).
            if kind == 0 and event[1] != 1:
                raise ValueError('update domain')
            if kind in (1, 2) and (event[-1] not in (0, 1) or event[4] not in (0, 1)):
                raise ValueError('candidate spin/reject domain')
            if kind in (7, 9) and not 0 <= event[-1] < 2**32:
                raise ValueError('actual primitive word domain')
            if kind == 7 and event[1] not in (0, 1):
                raise ValueError('acceptance domain')
            if kind == 7:
                if i == 0 or records[f'd:trace-{i-1:06}'] != [9, event[2]]:
                    raise ValueError('decision must follow its actual same-word draw')
        elif kind == 8:
            pos = 1
            for size in (18, 36, 36, 6, 0, 10, 624):
                if pos >= len(event) or event[pos] != size:
                    raise ValueError('saved checkpoint plane shape')
                pos += size + 1
            if pos != len(event) or any(not 0 <= x < 2**32 for x in event[-624:]):
                raise ValueError('checkpoint future words/trailing fields')
            checkpoints += 1
        elif kind == 10 and raw_extension:
            raw_frames += 1
            if len(event) != 627 or not 0 <= event[1] <= 624 or event[2] < 0 or any(not 0 <= x < 2**32 for x in event[3:]):
                raise ValueError('raw extension domains')
            if i == 0 or records[f'd:trace-{i-1:06}'][0] != 8:
                raise ValueError('raw extension checkpoint placement')
            if i != count[0] - 1:
                raise ValueError('prefix1 raw checkpoint must end trace frame')
        else:
            raise ValueError('event outside pinned ordinary-real schema')
    if checkpoints != 1:
        raise ValueError('prefix1 exactly one saved checkpoint')
    if raw_frames != int(raw_extension):
        raise ValueError('exactly one raw frame for Rust; none for Julia')
    if raw_extension:
        terminal = records[f'd:trace-{count[0]-1:06}']
        for key, expected in (
            ('d:checkpoint-000000-cursor', terminal[1:2]),
            ('d:checkpoint-000000-native-words-consumed', terminal[2:3]),
            ('d:checkpoint-000000-raw624', terminal[3:]),
        ):
            if key in records and records[key] != expected:
                raise ValueError('raw checkpoint record/event disagreement')


def require_original_julia(records, native_counter_observation):
    if 'd:ele_spn' in records:
        raise ValueError('Julia ordinary conserved-spin writer omits empty spin plane')
    if native_counter_observation != 'Unavailable':
        raise ValueError('Julia ABI exposes no native draw counter')
    sizes = {k: v for k, v in RUST_REQUIRED.items()
             if k in ('d:seed', 'd:group', 'd:steps', 'd:status', 'd:configured-samples',
                      'd:chain-samples', 'd:requested-width', 'd:initial-rng',
                      'd:initial-draw-count', 'd:rng', 'd:total-draw-count',
                      'd:sampling-draw-count', 'd:trace-events', 'd:acceptance-events',
                      'd:ele_idx', 'd:ele_cfg', 'd:ele_num', 'd:ele_proj_cnt',
                      'd:ele_spn', 'd:counter', 'n:initial-parameters', 'n:parameters')}
    # Pinned Julia writer 81835d31 lines206/214/215 initializes -1,
    # increments before writing, hence prefix1's only sample step is zero.
    for label in ('seeded', 'initialized', 'checkpoint-0', 'final'):
        sizes.update({f'd:{label}-{suffix}': size for suffix, size in
                      (('raw624', 624), ('cursor', 1), ('initialized', 1),
                       ('raw-abi', 10), ('observed-primitive-words', 1))})
    for key, size in sizes.items():
        if key not in records or len(records[key]) != size:
            raise ValueError('missing/truncated original Julia field: ' + key)
    for label in ('seeded', 'initialized', 'checkpoint-0', 'final'):
        cursor = records[f'd:{label}-cursor'][0]
        if not 0 <= cursor <= 624 or records[f'd:{label}-initialized'] != [1]:
            raise ValueError('Julia native cursor/initialization domain')
        if records[f'd:{label}-raw-abi'] != [1, 19937, 156, 624, 312, 16, 4, cursor, 1, 1]:
            raise ValueError('Julia pinned actual accessor ABI')
        if any(not 0 <= x < 2**32 for x in records[f'd:{label}-raw624']) or records[f'd:{label}-observed-primitive-words'][0] < 0:
            raise ValueError('Julia raw word/observed count domain')
    if records['d:steps'] != [1] or records['d:status'] != [0]:
        raise ValueError('Julia P01 successful prefix1')
    validate_trace(records, raw_extension=False)


def compare_rust(original_text, observed_text, original_outputs, observed_outputs, required_outputs):
    require_public_outputs(original_outputs, observed_outputs, required_outputs)
    original = parse_records(original_text)
    observed = parse_records(observed_text)
    require_original_rust(original)
    require_original_rust(observed)
    # Frozen writer 263fd2f0: ordinal starts at zero, increments after the
    # forwarded WORLD complex broadcast; initial sync precedes optimization,
    # then successful step0 sync. This literal P01 contract is source-derived,
    # never inferred from either observed output.
    added = {'d:parameter-broadcast-complete'} | {
        f'{namespace}:parameter-broadcast-{ordinal}-{part}'
        for ordinal in (0, 1)
        for namespace, part in (('d', 'frame'), ('n', 'before'), ('n', 'after'))
    }
    if set(observed) != set(original) | added or set(original) & added:
        raise ValueError('exact original/new observation key inventory')
    if observed['d:parameter-broadcast-complete'] != [2]:
        raise ValueError('initial plus step0 broadcast completion')
    for ordinal in (0, 1):
        if observed[f'd:parameter-broadcast-{ordinal}-frame'] != [0, ordinal, 0, 2]:
            raise ValueError('actual WORLD initial/step0 broadcast frame')
        for part in ('before', 'after'):
            if len(observed[f'n:parameter-broadcast-{ordinal}-{part}']) != 28:
                raise ValueError('actual packed14parameter components')
    # Original text rows compare without numerical tolerance: observation must
    # not alter a single original field in the SAME implementation/config.
    left = {line.split()[0]: line for line in original_text.splitlines()}
    right = {line.split()[0]: line for line in observed_text.splitlines()}
    if any(left[key] != right[key] for key in original):
        raise ValueError('original rank state/config/RNG/numerical record altered')
    if type(required_outputs) is not list or not required_outputs or len(set(required_outputs)) != len(required_outputs):
        raise ValueError('independently pinned public output inventory required')
    if set(original_outputs) != set(required_outputs) or set(observed_outputs) != set(required_outputs):
        raise ValueError('public output inventory differs')
    if any(type(original_outputs[name]) is not bytes or type(observed_outputs[name]) is not bytes
           or original_outputs[name] != observed_outputs[name] for name in required_outputs):
        raise ValueError('same-implementation public output bytes altered')
    return {'scope': 'Rust P01 observer passivity only', 'original_keys': len(original),
            'added_keys': len(added), 'cross_language_numeric_acceptance': False}


def compare_julia(original_text, observed_text, original_outputs, observed_outputs, required_outputs,
                  *, native_counter_observation):
    require_public_outputs(original_outputs, observed_outputs, required_outputs)
    require_original_julia(parse_records(original_text), native_counter_observation)
    require_original_julia(parse_records(observed_text), native_counter_observation)
    if original_text != observed_text:
        raise ValueError('Julia original rank fields altered by sidecar extension')
    if type(required_outputs) is not list or not required_outputs or len(set(required_outputs)) != len(required_outputs):
        raise ValueError('independently pinned public output inventory required')
    if set(original_outputs) != set(required_outputs) or set(observed_outputs) != set(required_outputs):
        raise ValueError('Julia public output inventory differs')
    if any(type(original_outputs[name]) is not bytes or type(observed_outputs[name]) is not bytes
           or original_outputs[name] != observed_outputs[name] for name in required_outputs):
        raise ValueError('Julia same-implementation output bytes altered')
    return {'scope': 'Julia P01 sidecar passivity only', 'cross_language_numeric_acceptance': False}
