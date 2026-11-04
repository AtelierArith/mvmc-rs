"""Native variant of benchmark/cpu_scaling_l128/contract.py (same checks, no Docker).

usage: contract_native.py OUT BEFORE_PROBE AFTER_PROBE [--modes opt phys] [--threads 1 4 8 16]
Exact RNG/discrete equality; numerical fields within the same abs+rel 1e-11 bound.
"""
import argparse, hashlib, json, math, os, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
INPUTS = REPO / 'benchmark/cpu_scaling_l128/inputs'
p = argparse.ArgumentParser()
p.add_argument('out', type=Path)
p.add_argument('before')
p.add_argument('after')
p.add_argument('--modes', nargs='+', default=['opt', 'phys'])
p.add_argument('--threads', nargs='+', type=int, default=[1, 4, 8, 16])
p.add_argument('--first-cpu', type=int, default=0)
a = p.parse_args()
out = a.out.resolve(); out.mkdir(parents=True)
(out / 'binary-sha256.txt').write_text(''.join(
    f"{hashlib.sha256(Path(b).read_bytes()).hexdigest()}  {b}\n" for b in (a.before, a.after)))
checks = []
for mode in a.modes:
    reference = None
    cases = [('before', 1, 0)] + [('after', n, 0) for n in a.threads] + [('after', max(a.threads), 1)]
    for version, n, rep in cases:
        label = f'{mode}-{version}-t{n}-r{rep}'
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   BLIS_NUM_THREADS='1', MVMC_RS_INNER_THREADS=str(n), RAYON_NUM_THREADS=str(n),
                   MVMC_RS_SR_BACKEND='c-order', MVMC_RS_MEASURE_PF_BACKEND='c-order', MVMC_C_TIMER='1')
        binary = a.before if version == 'before' else a.after
        cpus = f'{a.first_cpu}-{a.first_cpu + n - 1}'
        cmd = ['taskset', '-c', cpus, binary, mode, str(INPUTS / mode / 'namelist.def'),
               str(INPUTS / 'phys/zqp_opt.dat'), str(out / label)]
        result = subprocess.run(cmd, text=True, capture_output=True, env=env)
        (out / f'{label}.stdout').write_text(result.stdout); (out / f'{label}.stderr').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{label}: probe exited {result.returncode}')
        current = json.loads((out / label / 'contract.json').read_text())
        if reference is None:
            reference = current
        if current['exact'] != reference['exact']:
            different = [k for k in reference['exact'] if current['exact'].get(k) != reference['exact'][k]]
            raise AssertionError(f'{label}: exact RNG/discrete divergence in {different}')
        expected = reference['numerical']; actual = current['numerical']
        assert len(actual) == len(expected) and actual, f'{label}: numerical shape mismatch'
        maximum = 0.0; nonzero = 0
        for i, (x, y) in enumerate(zip(actual, expected)):
            if not (math.isfinite(x) and math.isfinite(y)):
                raise AssertionError(f'{label}: non-finite numerical field {i}')
            error = abs(x - y); maximum = max(maximum, error); nonzero += error != 0
            if error > 1e-11 + 1e-11 * max(abs(x), abs(y)):
                raise AssertionError(f'{label}: numerical field {i}: {x} vs {y}')
        checks.append({'case': label, 'exact_rng_discrete_match': True, 'numerical_fields': len(actual),
                       'max_absolute_difference': maximum, 'differing_fields': nonzero,
                       'decision_count': len(current['exact']['decisions']),
                       'rng_words_consumed': current['exact']['rng_words_consumed']})
        (out / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps(checks[-1]), flush=True)
