"""SOURCE-only independent positive/negative schema fixtures; no native inputs."""
import unittest
from test_p02_cross_discrete import fixture, text
from p02_typed_fullpack import (COMMON_NUMERIC, RUST_NUMERIC, JULIA_NUMERIC,
                            CHECKED_RETURN, assemble, diagnose, root_outputs, vector)


def positive():
    r, j = fixture()
    for implementation, records, extra in (('rust', r, RUST_NUMERIC), ('julia', j, JULIA_NUMERIC)):
        records.update({key: [0.0] * size for key, size in (COMMON_NUMERIC | extra).items()})
        records.update({'d:sr-kind': [0], 'd:sr-systems': [1],
                        'd:sr-system-000000-dimension': [10], 'd:sr-system-000000-not-solved': [0],
                        'd:sr-system-000000-status': [0], 'd:sr-system-000000-factor-info': [0],
                        'd:sr-system-000000-active': list(range(4, 24, 2)),
                        'd:sr-system-000000-flags': [1 if i >= 4 and i % 2 == 0 else 0 for i in range(28)],
                        'd:sr-system-000000-settings': [1, 1, 1, 0, 1]})
        if implementation == 'rust':
            records.update({'d:sr-system-000000-triangle': [85], 'd:sr-system-000000-nrhs': [1],
                            'd:sr-system-000000-solve-info': [0], 'd:normalized-boundaries': [1],
                            'd:normalized-step-000000-ordinal': [0],
                            'd:normalized-step-000000-complex': [0],
                            'd:parameter-broadcast-complete': [2],
                            'd:parameter-broadcast-0-frame': [0, 0, 0, 2],
                            'd:parameter-broadcast-1-frame': [0, 1, 0, 2]})
        else:
            records.update({'d:sr-system-000000-solve-returned': [1],
                            'd:sr-system-000000-solve-info-available': [0],
                            'd:sr-system-000000-solve-checked-successful-return': [1]})
    return r, j, ''.join(key + '=' + value + '\n' for key, value in CHECKED_RETURN.items())


class TypedControls(unittest.TestCase):
    def test_positive_distinct_solver_metadata_and_tail_audit(self):
        r, j, sidecar = positive()
        result = diagnose(text(r), text(j), 0, sidecar, written_mask=[i % 2 == 0 for i in range(28)])
        self.assertEqual(result['rust']['sr']['solve_observation'], {'kind': 'NativeINFO', 'info': 0})
        self.assertEqual(result['julia']['sr']['solve_observation']['kind'], 'CheckedSuccessfulReturn')
        self.assertEqual(len(result['rust']['normalized_oo_tail_audit']), 30)
        self.assertEqual(result['julia']['normalized_oo_tail_audit'], [])
        self.assertFalse(result['numeric_acceptance'])

    def test_every_numeric_plane_missing_wrongshape_nonfinite(self):
        for implementation, shapes in (('rust', COMMON_NUMERIC | RUST_NUMERIC), ('julia', COMMON_NUMERIC | JULIA_NUMERIC)):
            for key in shapes:
                for mutation in ('missing', 'truncated', 'nonfinite'):
                    r, j, sidecar = positive()
                    records = r if implementation == 'rust' else j
                    if mutation == 'missing': del records[key]
                    elif mutation == 'truncated': records[key] = records[key][:-1]
                    else: records[key][0] = float('nan')
                    with self.subTest(implementation=implementation, key=key, mutation=mutation), self.assertRaises(ValueError):
                        assemble(text(records), implementation, solve_sidecar=sidecar)

    def test_extra_numeric_plane_rejected(self):
        r, _, _ = positive()
        r['n:unreviewed'] = [0.0]
        with self.assertRaises(ValueError): assemble(text(r), 'rust')

    def test_metadata_bool_and_wrong_step_settings(self):
        with self.assertRaises(ValueError): vector({'d:value': [True]}, 'd:value', 1)
        for key in ('d:sr-system-000000-settings', 'd:sr-system-000000-dimension',
                    'd:normalized-step-000000-ordinal', 'd:sr-system-000000-solve-info'):
            r, _, _ = positive()
            r[key][0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError): assemble(text(r), 'rust')

    def test_active_duplicate_outofrange_truncated(self):
        for active in ([0] * 10, list(range(9)), list(range(9)) + [28]):
            r, _, _ = positive()
            r['d:sr-system-000000-active'] = active
            with self.assertRaises(ValueError): assemble(text(r), 'rust')

    def test_checked_return_duplicate_missing_extra_wrongstdlib(self):
        _, j, sidecar = positive()
        for bad in (None, sidecar + 'kind=CheckedSuccessfulReturn\n', sidecar + 'native_info=0\n',
                    sidecar.replace('kind=CheckedSuccessfulReturn\n', ''),
                    sidecar.replace(CHECKED_RETURN['stdlib_sha256'], '0' * 64)):
            with self.assertRaises(ValueError): assemble(text(j), 'julia', solve_sidecar=bad)

    def test_written_flag_drift_rejected_unwritten_kept(self):
        r, j, sidecar = positive()
        mask = [i % 2 == 0 for i in range(28)]
        j['d:sr-system-000000-flags'][1] = -7
        result = diagnose(text(r), text(j), 0, sidecar, written_mask=mask)
        self.assertEqual(result['julia']['sr']['raw_flags'][1], -7)
        j['d:sr-system-000000-flags'][4] = 0
        with self.assertRaises(ValueError): diagnose(text(r), text(j), 0, sidecar, written_mask=mask)
        r['d:sr-system-000000-flags'][4] = 0
        with self.assertRaises(ValueError): diagnose(text(r), text(j), 0, sidecar, written_mask=mask)

    def test_discrete_drift_fails_before_numeric_diagnosis(self):
        r, j, sidecar = positive()
        j['d:seed'] = [2]
        del r['n:local-energy']
        with self.assertRaisesRegex(ValueError, 'd:seed'):
            diagnose(text(r), text(j), 0, sidecar, written_mask=[False] * 28)

    def test_non_symmetric_lower_matrix_retained_no_solver_rewrite(self):
        r, _, _ = positive()
        r['n:sr-system-000000-matrix'][1] = -99.0
        result = assemble(text(r), 'rust')
        self.assertEqual(result['planes']['sr-system-000000-matrix'][1], -99.0)
        self.assertEqual(result['sr']['triangle'], 'U')

    def test_presync_uses_step_broadcast_not_initial(self):
        r, _, _ = positive()
        r['n:parameter-broadcast-0-before'] = [9.0] * 28
        r['n:parameter-broadcast-1-before'] = [3.0] * 28
        self.assertEqual(assemble(text(r), 'rust')['planes']['presync'], [3.0] * 28)

    def test_root_schemas_fixedzeros_and_malformed(self):
        original = {name: (' '.join(['0'] * count) + '\n').encode()
                    for name, count in (('zvo_out.dat', 6), ('zvo_var.dat', 48), ('zqp_opt.dat', 32))}
        self.assertEqual(set(root_outputs(original)), set(original))
        for name in original:
            for mutation in (b'0\n', original[name].rstrip(b'\n'), b'nan\n'):
                bad = dict(original); bad[name] = mutation
                with self.assertRaises(ValueError): root_outputs(bad)
        bad = dict(original)
        values = ['0'] * 48; values[2] = '1'
        bad['zvo_var.dat'] = (' '.join(values) + '\n').encode()
        with self.assertRaises(ValueError): root_outputs(bad)
        with self.assertRaises(ValueError): root_outputs({**original, 'timer': b'0\n'})
        with self.assertRaises(ValueError): root_outputs({})
