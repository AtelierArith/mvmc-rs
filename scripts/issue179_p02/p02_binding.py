"""P02 SOURCE-only authority/settings gate; does not execute an oracle.

This is not the historical P01 assembler. Callers must still validate the
complete discrete inventory, actual providers, and source-derived shapes.
"""
import hashlib
from pathlib import Path

PINNED = {
    'rust-recorder': '08e138f226ef188045ab2d338be2ad7780d4d09a924c66cef4022936253a65a5',
    'julia-recorder': 'bc568fbe3da572711adfed65165aeae1dcbdf9231b2d42bbbb70846b7167f307',
    'julia-helper': '07a217c0d9fcd5f02a60484f6e06db78cce91b5633209a728ce132da7d4d6661',
    'julia-manifest': '09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc',
    'rust-elf': '49f8d71386b2575254ab4ca10b8c94a38b5616435fbc29966d83fab79c6f8ca1',
    'modpara': '4a1ce2aee43e6fd9408dc0458c7d4c3a3d2457d4e673e0567678dbb0e344aa00',
}
SETTINGS = [1, 1, 1, 0, 1]


def verify_sources(paths):
    """Check actual regular bytes against independently declared original pins.

    No output-derived expectation, authority regeneration or directory search.
    Full dependency membership and PRE/POST replay remain caller obligations.
    """
    if type(paths) is not dict or set(paths) != set(PINNED):
        raise ValueError('closed P02 source binding inventory')
    result = {}
    for name, expected in PINNED.items():
        value = paths[name]
        if type(value) is not str or not value.startswith('/'):
            raise ValueError('absolute explicit source path: ' + name)
        original = Path(value)
        if original.is_symlink():
            raise ValueError('source binding must name canonical regular file: ' + name)
        path = original.resolve(strict=True)
        if str(path) != value:
            raise ValueError('source binding path must be canonical: ' + name)
        if not path.is_file():
            raise ValueError('regular source required: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('original source hash mismatch: ' + name)
        result[name] = {'path': str(path), 'sha256': actual}
    return result


def require_settings(records):
    """Typed P02 direct/store1 binding, never accept the P01 store0 default."""
    if type(records) is not dict:
        raise ValueError('record mapping required')
    values = records.get('d:sr-system-000000-settings')
    if (type(values) is not list or len(values) != 5
            or any(type(x) is not int for x in values) or values != SETTINGS):
        raise ValueError('P02 seed/steps/window/CG/store binding')
    return values.copy()


def require_rank_pairs(rust, julia):
    """No zip truncation; declared rank order is an explicit typed boundary."""
    for collection in (rust, julia):
        if type(collection) is not list or len(collection) != 2:
            raise ValueError('world2 requires exactly two rank packs')
        for rank, pack in enumerate(collection):
            if type(pack) is not dict:
                raise ValueError('rank pack required')
            if 'd:rank' in pack:
                raise ValueError('d:rank is not emitted by pinned P02 writers')
            for key, expected in (('d:seed', [rank + 1]), ('d:group', [rank, 0, 1])):
                values = pack.get(key)
                if (type(values) is not list or len(values) != len(expected)
                        or any(type(x) is not int for x in values) or values != expected):
                    raise ValueError('typed rank/order binding: ' + key)
            require_settings(pack)
    return [(rust[i], julia[i]) for i in range(2)]
