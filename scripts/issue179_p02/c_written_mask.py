"""SOURCE: exact historical P01 C-reader-written slots, no malloc expectation."""
import hashlib

READER = '6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9'
INPUTS = {
    'gutzwilleridx.def': 'c1362e2dcf537556dc4db465322ff4126962c77e28dc52e18c0ac363f4e82ae4',
    'jastrowidx.def': 'cc5206fe01738c353efdbbcbf1bb6f6730a532c04466a39efd9307c04c7f9fe0',
    'orbitalidx.def': 'c2a1a275bfab65ff94a19e1ed6f9a630c02a0e1eb4ee80cd0787e166b5997ff6',
}


def derive(reader_bytes, input_bytes):
    if type(reader_bytes) is not bytes or hashlib.sha256(reader_bytes).hexdigest() != READER:
        raise ValueError('original C reader source pin')
    if type(input_bytes) is not dict or set(input_bytes) != set(INPUTS):
        raise ValueError('bounded original parameter-family inventory')
    for name, digest in INPUTS.items():
        if type(input_bytes[name]) is not bytes or hashlib.sha256(input_bytes[name]).hexdigest() != digest:
            raise ValueError('original parameter-family input pin: ' + name)
    # Exact pinned input headers: realComplexType0, counts1/1/12. C callers
    # offsets0/NGutz=1/NProj=2. GetInfoOpt writes2*fidx; oddslot onlyifcomplex>0.
    return {'written_mask': [i % 2 == 0 for i in range(28)],
            'written_flag_values': {2*i: 0 if i < 2 else 1 for i in range(14)},
            'unwritten_slots': list(range(1, 28, 2)),
            'unwritten_allocation_values': 'Unspecified',
            'reader_sha256': READER, 'input_sha256': INPUTS.copy(),
            'family_offsets': {'Gutzwiller': 0, 'Jastrow': 1, 'OrbitalAntiParallel': 2}}
