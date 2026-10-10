"""Optional isolated native C / reviewed Julia Opt300 confirmation; run through uv."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, '/workspaces/mvmc-rs/scripts')
from bench_c_mpi import validate_world
from bench_cpu_round import parse_timer

cache = Path('/home/vscode/.cache/mvmc')
out = cache / 'issue496-julia-b672-fresh-c-pair-root'
source = cache / 'julia-issue496'
expected_head = 'b672e05587cf9d4cd72f567c3ac43b8a4542e1e6'
cbin = cache / 'c-benchmark-independent-20261010/build/src/mVMC/vmc.out'
jbin = cache / 'tools/julia-1.13.1/bin/julia'
project = cache / 'julia-issue496-project'
worker = Path('/workspaces/mvmc-rs/benchmark/mpi_comparison/worker.jl')
observer = cache / 'mpi-world-observer-495.so'
inputs = Path('/workspaces/mvmc-rs/bench-out/issue502-production-opt-physcal-20261010')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()

assert git('rev-parse', 'HEAD') == expected_head
assert not git('status', '--porcelain', '--untracked-files=no')
assert digest(cbin) == '6ae44625e227ccd6b9c38d2a7731fb445955108772e8741b452f70c6a641ac28'
hashes = {str(p): digest(p) for package in ('MVMCOptimizers.jl', 'MVMCExpertModeParsers.jl')
          for p in (source / package / 'src').rglob('*.jl')}
hashes.update({str(p): digest(p) for p in (cbin, jbin, worker, observer, project / 'Manifest-v1.13.toml', project / 'LocalPreferences.toml')})
env = dict(os.environ)
for key in list(env):
    if key.startswith('MVMC_') or key in ('LD_PRELOAD', 'JULIA_MVMC_THREAD_PROFILE'):
        env.pop(key)
env.update(OPENBLAS_NUM_THREADS='1', BLIS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OMP_NUM_THREADS='1', JULIA_NUM_THREADS='4,0', JULIA_NUM_GC_THREADS='1',
           JULIA_MVMC_INNER_THREADS='1', JULIA_MVMC_PFAPACK_THREADS='0',
           JULIA_DEPOT_PATH=str(cache / 'julia-depot'), UCX_MEMTYPE_CACHE='no',
           UCX_ERROR_SIGNALS='SIGILL,SIGBUS,SIGFPE',
           LD_LIBRARY_PATH='/opt/mpich/lib:' + env.get('LD_LIBRARY_PATH', ''))
out.mkdir(exist_ok=False)
shutil.copy2(__file__, out / 'reproduce.py')
shutil.copy2(worker, out / 'worker.jl')
provenance = dict(julia_head=expected_head, source_and_binary_hashes=hashes,
    environment={k: v for k, v in env.items() if k.startswith(('JULIA_', 'OMP_', 'OPENBLAS_', 'BLIS_', 'MKL_', 'UCX_'))},
    policy='Opt300; MPI4 x4 workers; BLAS1; averaging300; alternating C/Julia batches. C initial warmup then fresh processes. Julia warmup1/measurement1 per batch. C rank0 internal All; Julia warmed production API maximum rank duration. Boundaries differ.')
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
rows = []
differences = []
def check_output(path, reference, size, implementation, rep):
    data = [[float(v) for v in line.split()] for line in path.read_text().splitlines()]
    baseline = [[float(v) for v in line.split()] for line in reference.read_text().splitlines()]
    assert len(data) == len(baseline) == 300
    assert all(len(row) == 6 and all(math.isfinite(v) for v in row) for row in data)
    assert all(len(row) == 6 for row in baseline)
    differences.append(dict(sites=size, implementation=implementation, rep=rep,
        max_abs_all_columns=max(abs(x-y) for a, b in zip(data, baseline) for x, y in zip(a, b))))
    (out / 'output-observations.json').write_text(json.dumps(differences, indent=2) + '\n')

def c_run(size, rep):
    dest = out / f'L{size}-c-{rep}'
    dest.mkdir()
    baseline = inputs / f'L{size}/Opt-C/run-1'
    for p in baseline.glob('*.def'):
        shutil.copy2(p, dest / p.name)
    assert re.search(r'NSROptItrSmp\s+300\b', (dest / 'modpara.def').read_text())
    (dest / 'output').mkdir()
    with (dest / 'run.log').open('w') as log:
        subprocess.run(['/opt/mpich/bin/mpiexec', '-n', '4', str(cbin), 'namelist.def'],
            cwd=dest, env=dict(env, OMP_NUM_THREADS='4', LD_PRELOAD=str(observer)),
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    validate_world((dest / 'run.log').read_text(), 4, 4)
    check_output(next((dest / 'output').glob('zvo_out_*.dat')),
                 next((baseline / 'output').glob('zvo_out_*.dat')), size, 'c', rep)
    return parse_timer(dest / 'zvo_CalcTimer.dat')[0]

def julia_run(size, rep):
    dest = out / f'L{size}-julia-{rep}'
    dest.mkdir()
    with (dest / 'run.log').open('w') as log:
        subprocess.run(['/opt/mpich/bin/mpiexec', '-n', '4', str(jbin), '--project=' + str(project),
            '--startup-file=no', str(out / 'worker.jl'),
            str(inputs / f'L{size}/opt-inputs/namelist.def'), '300', '1', '1', str(dest), '4'],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    text = (dest / 'run.log').read_text()
    validate_world(text, 4, 4)
    assert sorted((int(a), int(b)) for a, b in re.findall(r'^NATIVE_BLAS_THREADS (\d+) (\d+)$', text, re.M)) == [(r, 1) for r in range(4)]
    values = re.findall(r'^BENCH 1 ([0-9.]+) (\S+)$', text, re.M)
    assert len(values) == 1
    check_output(dest / 'run-1/zvo_out.dat',
        cache / f'issue496-julia-owned-typed-production/L{size}/run-1/zvo_out.dat', size, 'julia', rep)
    return float(values[0][0])

for size in (32, 64):
    c_run(size, 0)
    for rep in (1, 2, 3):
        for implementation in (('c', 'julia') if rep % 2 else ('julia', 'c')):
            seconds = (c_run if implementation == 'c' else julia_run)(size, rep)
            rows.append(dict(sites=size, implementation=implementation, rep=rep, seconds=seconds))
            print(json.dumps(rows[-1]), flush=True)
            (out / 'measurements.json').write_text(json.dumps(rows, indent=2) + '\n')
summary = [dict(sites=size, implementation=implementation,
    median_seconds=statistics.median(r['seconds'] for r in rows if r['sites'] == size and r['implementation'] == implementation))
    for size in (32, 64) for implementation in ('c', 'julia')]
assert git('rev-parse', 'HEAD') == expected_head
assert not git('status', '--porcelain', '--untracked-files=no')
assert hashes == {name: digest(Path(name)) for name in hashes}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(out / 'terminal.json').write_text(json.dumps(dict(success=True, measurements=len(rows))) + '\n')
print(json.dumps(summary), flush=True)
