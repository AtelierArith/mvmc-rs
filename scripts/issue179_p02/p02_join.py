"""SOURCE-only P02 diagnostic integration; no runtime/model acceptance."""
from p02_binding import verify_sources, require_settings, require_rank_pairs
from strict_text_records import parse_records
from p02_typed_fullpack import diagnose, root_outputs
from c_written_mask import derive


def join(rust_texts, julia_texts, sidecars, *, source_paths, reader_bytes, family_bytes,
         rust_outputs, julia_outputs):
    sources = verify_sources(source_paths)
    mask_authority = derive(reader_bytes, family_bytes)
    for collection in (rust_texts, julia_texts, sidecars):
        if type(collection) is not list or len(collection) != 2:
            raise ValueError('P02 exactly two original rank artifacts required')
        if any(type(value) is not str for value in collection):
            raise ValueError('original text artifacts required')
    # Metadata gates precede even diagnostic numerical assembly.
    for text in rust_texts + julia_texts:
        require_settings(parse_records(text))
    require_rank_pairs([parse_records(text) for text in rust_texts],
                       [parse_records(text) for text in julia_texts])
    ranks = [diagnose(rust_texts[i], julia_texts[i], i, sidecars[i],
                      written_mask=mask_authority['written_mask']) for i in range(2)]
    roots = {'rust': root_outputs(rust_outputs),
             'julia': root_outputs(julia_outputs)}
    return {'sources': sources, 'c_written_authority': mask_authority,
            'ranks': ranks, 'roots': roots,
            'scope': 'P02 retained artifact diagnostic only',
            'runtime_authenticated': False, 'model_acceptance': False,
            'numeric_acceptance': False, 'tolerances': 'NoneApplied'}
