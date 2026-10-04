"""SOURCE-only additional controls. Not executed."""
import unittest
from p02_binding import require_settings, require_rank_pairs, verify_sources
from test_p02_typed_fullpack import positive
from test_p02_cross_discrete import text
from p02_typed_fullpack import assemble


class BindingControls(unittest.TestCase):
    def test_store1_positive(self):
        r, j, sidecar = positive()
        for records in (r, j):
            self.assertEqual(require_settings(records), [1, 1, 1, 0, 1])
        self.assertEqual(assemble(text(r), 'rust')['sr']['settings'], [1, 1, 1, 0, 1])
        self.assertEqual(assemble(text(j), 'julia', solve_sidecar=sidecar)['sr']['settings'], [1, 1, 1, 0, 1])

    def test_wrong_or_bool_settings(self):
        for values in ([1, 1, 1, 0, 0], [True, 1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1, 1]):
            with self.assertRaises(ValueError):
                require_settings({'d:sr-system-000000-settings': values})

    def test_no_zip_truncation(self):
        with self.assertRaises(ValueError):
            require_rank_pairs([], [])

    def test_unknown_source_binding(self):
        with self.assertRaises(ValueError):
            verify_sources({'historical-p01': '/unavailable'})
