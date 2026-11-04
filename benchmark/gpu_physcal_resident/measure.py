"""Sequential warmed API comparison; one warmup and all requested repetitions."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('entrypoints', ROOT / 'scripts/benchmark/run.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resident', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--legacy-source', required=True)
    p.add_argument('--julia-project', type=Path, required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--cpu', type=int, default=0)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.reps > 0 and args.cpu in os.sched_getaffinity(0)
    out = args.output.resolve()
    out.mkdir()
    source = ROOT / 'benchmark/cpu_scaling_l128/inputs/phys'
    inp = out / 'input'
    shutil.copytree(source, inp)
    assert sha(inp / 'zqp_opt.dat') == bench.PARAM_SHA
    env = bench.clean_environment()
    env.update(CUDA_VISIBLE_DEVICES=args.device, RAYON_NUM_THREADS='1',
               MVMC_RS_INNER_THREADS='1', MVMC_BENCH_REPS=str(args.reps), JULIA_NUM_THREADS='1')
    commands = []

    def run(name, cmd, case_env=env, stdin=None):
        commands.append(dict(name=name, command=list(map(str, cmd)), stdin=stdin,
                             load_before=os.getloadavg(), start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             environment={k:v for k,v in case_env.items() if k.startswith(('MVMC_', 'JULIA_', 'CUDA_', 'OPENBLAS_', 'OMP_', 'LD_LIBRARY_PATH'))}))
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(list(map(str, cmd)), cwd=ROOT, env=case_env, text=True,
                           input=stdin, stdout=log, stderr=subprocess.STDOUT, check=True)
        commands[-1]['load_after'] = os.getloadavg()
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        return (out/f'{name}.log').read_text()

    julia_root = args.julia_project.resolve().parents[1]
    git = lambda path, *cmd: subprocess.check_output(['git', '-C', str(path), *cmd], text=True).strip()
    metadata = dict(
        source=git(ROOT, 'rev-parse', 'HEAD'), status=git(ROOT, 'status', '--short'),
        legacy_source=args.legacy_source, julia_source=git(julia_root, 'rev-parse', 'HEAD'),
        julia_lock_sha256=sha(args.julia_project/'Manifest-v1.13.toml'),
        binary_sha256={kind: sha(path) for kind,path in [('legacy', args.legacy), ('resident', args.resident)]},
        input_sha256={f.name: sha(f) for f in inp.iterdir() if f.is_file()},
        uname=list(os.uname()), cpu=args.cpu, gpu_selector=args.device,
        environment={k:v for k,v in env.items() if k.startswith(('MVMC_', 'JULIA_', 'CUDA_', 'OPENBLAS_', 'OMP_', 'LD_LIBRARY_PATH'))},
        scope='Warm public APIs including parse/init/sample/observables/host results; startup, rep0, explicit Julia GC and serialization excluded. Rust internal state teardown included.',
    )
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    run('platform', ['uname', '-a'])
    run('rust-version', ['rustc', '-Vv'])
    run('julia-version', ['julia', '+1.13', '--version'])
    run('gpus', ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used', '--format=csv,noheader'])
    for kind,binary in [('legacy', args.legacy), ('resident', args.resident)]:
        run(f'{kind}-libraries', ['ldd', binary])
    legacy_env = dict(env, MVMC_RS_CUDA_RNG='sfmt')
    for selector in ('SAMPLER','MEASURE','MEASURE_PF','SR'):
        legacy_env[f'MVMC_RS_{selector}_BACKEND'] = 'cuda'
    prefix = ['taskset', '-c', str(args.cpu)]
    rows = []
    text = run('legacy', [*prefix, args.legacy, inp/'namelist.def', inp/'zqp_opt.dat', out/'legacy', '--no-output'], legacy_env)
    rows += [dict(kind='Rust single SFMT', **r) for r in bench.parse_times(text, args.reps)]
    stdin = ''.join(f'{rep}\n' for rep in range(args.reps+1))+'quit\n'
    text = run('resident', [*prefix, args.resident, inp/'namelist.def', inp/'zqp_opt.dat', out/'resident', 320, 1, 128, 128], stdin=stdin)
    rows += [dict(kind='Rust resident XORWOW', **r) for r in bench.parse_times(text, args.reps)]
    traffic = [dict(rep=int(rep), **{name:int(value) for name,value in re.findall(r'(\w+): (\d+)', fields)})
               for rep,fields in re.findall(r'BENCH rep=(\d+).*?traffic=RawExecCounters \{ (.*?) \}', text)]
    text = run('julia', [*prefix, 'julia', '+1.13', '--startup-file=no', '--threads=1',
                        f'--project={args.julia_project.resolve()}', ROOT/'scripts/benchmark/julia.jl',
                        'gpu', 'phys', inp, out/'julia', 20, 320, args.reps])
    rows += [dict(kind='Julia XORWOW', **r) for r in bench.parse_times(text, args.reps)]
    (out/'timings.json').write_text(json.dumps(rows, indent=2)+'\n')
    (out/'traffic.json').write_text(json.dumps(traffic, indent=2)+'\n')
    summary = {kind: dict(median=statistics.median(r['seconds'] for r in rows if r['kind']==kind and r['rep']>0),
                          measured=[r['seconds'] for r in rows if r['kind']==kind and r['rep']>0])
               for kind in dict.fromkeys(r['kind'] for r in rows)}
    assert all(math.isfinite(group['median']) for group in summary.values())
    comparisons = []
    for rep in range(args.reps+1):
        tokens = iter((out/f'resident/rep-{rep}.txt').read_text().split())
        assert next(tokens)=='seconds'; next(tokens)
        assert next(tokens)=='energy'; actual_energy=float(next(tokens))
        expected_energy=float((out/f'julia/rep{rep}/energy.txt').read_text())
        comparisons.append(dict(rep=rep, name='energy', max_abs=abs(actual_energy-expected_energy)))
        for name in ('one','two','two_ex'):
            assert next(tokens)==name
            actual = [float(next(tokens)) for _ in range(int(next(tokens)))]
            expected = list(map(float,(out/f'julia/rep{rep}/{name}.txt').read_text().split()))
            assert len(actual)==len(expected)
            errors = [abs(a-b) for a,b in zip(actual,expected)]
            assert all(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-10+1e-10*max(abs(a),abs(b)) for a,b in zip(actual,expected))
            comparisons.append(dict(rep=rep, name=name, max_abs=max(errors, default=0.0)))
        assert abs(actual_energy-expected_energy)<=1e-10+1e-10*max(abs(actual_energy),abs(expected_energy))
    (out/'numerical-comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    main()
