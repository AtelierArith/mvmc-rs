"""SOURCE-only independent rank-association negatives; unexecuted."""
import unittest
from test_p02_cross_discrete import fixture, text
from p02_cross_discrete import join, rank_join
from p02_binding import require_rank_pairs


class RankControls(unittest.TestCase):
    def test_reversed_rust_or_julia_artifacts_rejected(self):
        pairs = [fixture(rank) for rank in range(2)]
        r = [text(pair[0]) for pair in pairs]
        j = [text(pair[1]) for pair in pairs]
        for left, right in ((r[::-1], j), (r, j[::-1])):
            with self.assertRaises(ValueError):
                join(left, right)

    def test_unemitted_rank_metadata_rejected_even_if_correct(self):
        for language in (0, 1):
            for value in (0, 1, True):
                pair = fixture()
                pair[language]['d:rank'] = [value]
                with self.assertRaises(ValueError):
                    rank_join(text(pair[0]), text(pair[1]), 0)

    def test_explicit_pair_gate_rejects_seed_group_and_bool(self):
        for key, values in (('d:seed', [2]), ('d:group', [1, 0, 1]),
                            ('d:seed', [True])):
            pairs = [fixture(rank) for rank in range(2)]
            r, j = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            for pack in r + j:
                pack['d:sr-system-000000-settings'] = [1, 1, 1, 0, 1]
            j[0][key] = values
            with self.assertRaises(ValueError):
                require_rank_pairs(r, j)
