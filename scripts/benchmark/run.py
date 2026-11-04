"""One-command, shared-input CPU/GPU benchmarks (Linux, Juliaup 1.13)."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
JULIA = ROOT / 'extern/Julia-mVMC'
INPUTS = ROOT / 'benchmark/cpu_scaling_l128/inputs'
TOOLS = ROOT / 'scripts/benchmark'
PARAM_SHA = '9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90'


def physical_cpus():
    """One allowed logical CPU from each physical core, without assuming CPU 0."""
    found = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple((topology / f).read_text().strip() for f in ('physical_package_id', 'core_id'))
        found.setdefault(key, cpu)
    return list(found.values())


def clean_environment():
    # Ambient selectors/profilers must not silently change workload or timings.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('MVMC_', 'JULIA_MVMC_'))}
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               BLIS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', JULIA_NUM_GC_THREADS='1',
               MVMC_C_TIMER='1', MVMC_BLAS_PROVIDER='openblas', MVMC_RS_SR_BACKEND='c-order',
               MVMC_RS_MEASURE_PF_BACKEND='c-order', JULIA_MVMC_INNER_THREADS='1',
               JULIA_MVMC_PFAPACK_THREADS='1')
    return env


def command(cmd, log=None, env=None, stdin=None):
    print('+ ' + shlex.join(map(str, cmd)), flush=True)
    lines = []
    stream = Path(log).open('w') if log else None
    try:
        with subprocess.Popen(list(map(str, cmd)), cwd=ROOT, env=env,
                              stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            if stdin is not None:
                process.stdin.write(stdin)
                process.stdin.close()
            for line in process.stdout:
                lines.append(line)
                if stream:
                    stream.write(line)
                    stream.flush()
            status = process.wait()
    finally:
        if stream:
            stream.close()
    text = ''.join(lines)
    if status:
        raise RuntimeError(f'command failed ({status}); log: {log}\n{text[-4000:]}')
    return text


def uv(script, *args):
    return ['uv', 'run', '--no-project', 'python', str(script), *map(str, args)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(out, steps, samples):
    assert sha(INPUTS / 'phys/zqp_opt.dat') == PARAM_SHA
    for workload in ('opt', 'phys'):
        dest = out / 'inputs' / workload
        shutil.copytree(INPUTS / workload, dest)
        path = dest / 'modpara.def'
        text = path.read_text()
        for key, value in {'NSROptItrStep': steps, 'NVMCSample': samples}.items():
            text, count = re.subn(rf'(?m)^{key}\s+\S+', f'{key} {value}', text)
            assert count == 1, key
        path.write_text(text)


def parse_times(text, reps):
    rows = [{'rep': int(rep), 'seconds': float(seconds)} for rep, seconds in
            re.findall(r'(?:BENCH )?rep=(\d+) seconds=([\d.eE+-]+)', text)]
    assert [r['rep'] for r in rows] == list(range(reps + 1)), rows
    assert all(math.isfinite(r['seconds']) and r['seconds'] > 0 for r in rows)
    return rows


def pick_image(explicit, out, env):
    if explicit:
        command(['docker', 'image', 'inspect', explicit], out / 'image.json', env)
        return explicit
    available = command(['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}'], env=env).splitlines()
    candidates = [x for x in available if x.startswith('vsc-mvmc-rs-') and x.endswith(':latest')]
    image = 'mvmc-rs-benchmark:local'
    if image not in available and candidates:
        image = sorted(candidates)[0]
    if image not in available:
        command(['docker', 'build', '-t', image, ROOT / '.devcontainer'], out / 'image-build.log', env)
    command(['docker', 'image', 'inspect', image], out / 'image.json', env)
    return image


def prepare_c(image, build, out, env):
    c_source = ROOT / 'extern/mVMC-1.3.0'
    if command(['git', '-C', c_source, 'status', '--porcelain', '--untracked-files=no'], env=env).strip():
        raise ValueError('C reference has tracked modifications; preserve them and use a clean reference checkout')
    source = command(['git', '-C', c_source, 'rev-parse', 'HEAD'], env=env).strip()
    image_id = json.loads((out / 'image.json').read_text())[0]['Id']
    cached = build / f'c-{source[:12]}-{image_id.removeprefix("sha256:")[:12]}'
    binary = cached / 'build/src/mVMC/vmc.out'
    manifest = cached / 'benchmark-build.json'
    if not manifest.exists():
        temp = cached.with_name(cached.name + f'-building-{os.getpid()}')
        command(['bash', ROOT / 'benchmark/cpu_scaling_l128/build-c-reference.sh', ROOT, image, temp], out / 'c-build.log', env)
        data = dict(source=source, image=image_id, binary_sha256=sha(temp / 'build/src/mVMC/vmc.out'))
        (temp / 'benchmark-build.json').write_text(json.dumps(data, indent=2) + '\n')
        temp.rename(cached)
    record = json.loads(manifest.read_text())
    assert record == dict(source=source, image=image_id, binary_sha256=sha(binary)), 'C cache changed'
    shutil.copyfile(manifest, out / 'c-build.json')
    return cached


def julia_run(mode, workload, n, args, out, env, cpus):
    case_env = dict(env, JULIA_NUM_THREADS=str(n))
    project = JULIA if mode == 'cpu' else JULIA / 'test/gpu'
    dest = out / f'julia-{workload}-t{n}'
    text = command(['taskset', '-c', ','.join(map(str, cpus[:n])), 'julia', '+1.13',
                    '--startup-file=no', f'--threads={n}', f'--project={project}',
                    TOOLS / 'julia.jl', mode, workload, out / 'inputs' / workload,
                    dest, args.steps, args.samples, args.reps], out / f'julia-{workload}-t{n}.log', case_env)
    return parse_times(text, args.reps)


def cuda_library_environment(env, out):
    """Use installed CUDA 12 libraries, including Julia artifacts, without fixed paths."""
    search = list(filter(None, env.get('LD_LIBRARY_PATH', '').split(':')))
    override = os.environ.get('MVMC_BENCH_CUDA_LIBRARY_PATH')
    if override:
        search = override.split(':') + search
    else:
        depots = command(['julia', '+1.13', '--startup-file=no', '-e',
                          'foreach(println,DEPOT_PATH)'], out / 'julia-depots.txt', env).splitlines()
        for depot in depots:
            for library in ('libnvrtc.so.12', 'libcublas.so.12', 'libcusolver.so.11', 'libcurand.so.10'):
                search += [str(p.parent) for p in sorted((Path(depot) / 'artifacts').glob(f'*/lib/{library}'))]
        for path in ('/usr/local/cuda/lib64', '/usr/local/cuda-12.8/lib64'):
            if Path(path).is_dir():
                search.append(path)
    result = dict(env)
    result['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(search))
    (out / 'rust-library-paths.json').write_text(json.dumps(search, indent=2) + '\n')
    return result


def select_gpu(query, explicit, visible):
    devices = [[field.strip() for field in line.split(',')] for line in query.splitlines()]
    selectors = [explicit] if explicit is not None else (visible.split(',') if visible is not None else None)
    if selectors is not None:
        selectors = [value.strip() for value in selectors]
        if not selectors or any(not value or value == '-1' or ',' in value for value in selectors):
            raise ValueError('No single usable GPU selector; use --device INDEX or GPU-UUID')
        devices = [row for row in devices if any(row[0] == value or row[1].startswith(value) for value in selectors)]
    if not devices:
        raise ValueError('No selected NVIDIA GPU is available; use --device INDEX or GPU-UUID')
    # Preserve the visible set, selecting one physical device by UUID before remapping to ordinal 0.
    return min(devices, key=lambda row: int(row[-1].split()[0]))[1]


def gpu_measure(args, out, build, env, cpus):
    query = command(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used',
                     '--format=csv,noheader'], out / 'gpus.txt', env)
    device = select_gpu(query, args.device, env.get('CUDA_VISIBLE_DEVICES'))
    gpu_env = dict(env, CUDA_VISIBLE_DEVICES=device, MVMC_RS_INNER_THREADS='1',
                   RAYON_NUM_THREADS='1', MVMC_BENCH_REPS=str(args.reps))
    rust_env = cuda_library_environment(gpu_env, out)
    target = build / 'gpu'
    command(['cargo', 'build', '--locked', '--release', '--manifest-path',
             ROOT / 'gpu/mvmc-gpu-cuda/Cargo.toml', '--example', 'resident_opt_worker',
             '--example', 'resident_physcal_worker'], out / 'rust-build.log', dict(rust_env, CARGO_TARGET_DIR=str(target)))
    rows = []
    for workload in args.workloads:
        binary = target / 'release/examples' / ('resident_opt_worker' if workload == 'opt' else 'resident_physcal_worker')
        dest = out / f'rust-{workload}'
        if workload == 'opt':
            cmd = [binary, out / 'inputs/opt/namelist.def', dest, args.samples, args.steps, 128, 128]
        else:
            cmd = [binary, out / 'inputs/phys/namelist.def', out / 'inputs/phys/zqp_opt.dat',
                   dest, args.samples, 1, 128, 128]
        stdin = ''.join(f'{rep}\n' for rep in range(args.reps + 1)) + 'quit\n'
        text = command(['taskset', '-c', cpus[0], *cmd], out / f'rust-{workload}.log', rust_env, stdin)
        rust_rows = parse_times(text, args.reps)
        julia_rows = julia_run('gpu', workload, 1, args, out, gpu_env, cpus)
        for kind, times in [('Rust GPU', rust_rows), ('Julia GPU', julia_rows)]:
            rows += [dict(workload=workload, kind=kind, threads=1, **r) for r in times]
        (out / 'timings.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows, dict(device=device, gpu_rng='cuRAND XORWOW resident ensemble',
                      cuda_library_path=rust_env['LD_LIBRARY_PATH'],
                      binary_sha256={name: sha(target / 'release/examples' / name)
                                     for name in ('resident_opt_worker', 'resident_physcal_worker')},
                      rust_environment={k:v for k,v in rust_env.items()
                                        if k.startswith(('MVMC_', 'CUDA_VISIBLE', 'RAYON_'))})


def cpu_measure(args, out, build, env, cpus):
    image = pick_image(args.image, out, env)
    c_dir = prepare_c(image, build, out, env)
    target = build / 'cpu'
    command(['cargo', 'build', '--locked', '--release', '-p', 'mvmc-cli', '--bin', 'mvmc'],
            out / 'rust-build.log', dict(env, CARGO_TARGET_DIR=str(target)))
    command(uv(ROOT / 'benchmark/cpu_scaling_l128/measure.py', '--repo', ROOT,
               '--image', image, '--target-volume', f'mvmc-benchmark-unused-{os.getuid()}',
               '--c-dir', c_dir, '--rust-binary', target / 'release/mvmc',
               '--opt-input', out / 'inputs/opt', '--phys-input', out / 'inputs/phys',
               '--output', out / 'c-rust', '--threads', *args.threads,
               '--reps', args.reps, '--workloads', *args.workloads,
               '--cpu-ids', *cpus[:max(args.threads)],
               '--container-name', f'mvmc-bench-{os.getuid()}-{os.getpid()}'), out / 'c-rust.log', env)
    original = json.loads((out / 'c-rust/timings.json').read_text())
    rows = [dict(workload=r['mode'], kind={'rust': 'Rust', 'c': 'C'}[r['kind']],
                 threads=r['threads'], rep=r['rep'], seconds=r['wall']) for r in original]
    for workload in args.workloads:
        for n in args.threads:
            rows += [dict(workload=workload, kind='Julia', threads=n, **r)
                     for r in julia_run('cpu', workload, n, args, out, env, cpus)]
            (out / 'timings.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows, dict(image=image, binary_sha256={'Rust':sha(target / 'release/mvmc'),
                                                  'C':sha(c_dir / 'build/src/mVMC/vmc.out')})


def write_report(args, out, rows):
    kinds = ['C', 'Julia', 'Rust'] if args.mode == 'cpu' else ['Julia GPU', 'Rust GPU']
    summaries = []
    for workload in args.workloads:
        for n in args.threads if args.mode == 'cpu' else [1]:
            values = {}
            for kind in kinds:
                group = [r for r in rows if (r['workload'], r['threads'], r['kind']) == (workload, n, kind)]
                assert sorted(r['rep'] for r in group) == list(range(args.reps + 1))
                assert all(math.isfinite(r['seconds']) and r['seconds'] > 0 for r in group)
                values[kind] = statistics.median(r['seconds'] for r in group if r['rep'] > 0)
            summaries.append(dict(workload=workload, threads=n, medians=values))
    lines = ['# Hubbard L128 benchmark', '',
             f'Mode: {args.mode}; steps={args.steps}; samples/walkers={args.samples}; '
             f'warmup=1; measured repetitions={args.reps}; smoke={args.smoke}.', '',
             'CPU: C/Rust process wall (C includes mpirun, Docker launch excluded); '
             'Julia warm API including input/output, JIT and explicit GC excluded. '
             'GPU: both warm public APIs; setup/JIT/rep0/explicit GC/serialization excluded.', '',
             'CPU uses c-order PF/SR, physical cores, BLAS=1. GPU Opt and PhysCal use '
             'resident XORWOW ensembles with 128 warmup and 128 interval proposals per '
             'walker. CPU uses a single SFMT chain. Common definitions and '
             'fixed parameters do not imply identical sampling work or kernel speedup.', '']
    for workload in args.workloads:
        lines += [f'## {workload}', '']
        if args.mode == 'cpu':
            lines += ['| CPU threads | C | Julia | Rust |', '|---:|---:|---:|---:|']
            for row in summaries:
                if row['workload'] == workload:
                    lines.append('| ' + str(row['threads']) + ' | ' + ' | '.join(f"{row['medians'][k]:.3f} s" for k in kinds) + ' |')
        else:
            row = next(r for r in summaries if r['workload'] == workload)
            lines += ['| GPU | Median |', '|---|---:|']
            lines += [f"| {kind} | {row['medians'][kind]:.3f} s |" for kind in kinds]
            lines += ['| C | No GPU implementation |']
        lines += ['']
    lines += ['This driver validates completed repetitions, finite timings and runtime output '
              'checks. It does not establish cross-implementation numerical/RNG parity. '
              'Performance acceptance is a separate recorded cohort (#629).', '']
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    (out / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['cpu', 'gpu'])
    p.add_argument('--output', type=Path, help='fresh directory (default: target/bench/<mode>-UTC-pid)')
    p.add_argument('--build-dir', type=Path, help='persistent build cache, separate for each checkout')
    p.add_argument('--threads', type=int, nargs='+', help='CPU thread counts (default: 1 4 8 16)')
    p.add_argument('--device', help='one NVIDIA GPU index/UUID (default: visible GPU or least memory used)')
    p.add_argument('--image', default=os.environ.get('MVMC_BENCH_IMAGE'), help='CPU development image; auto-detect/build if omitted')
    p.add_argument('--workloads', nargs='+', choices=['opt', 'phys'], default=['opt', 'phys'])
    p.add_argument('--reps', type=int, help='measured repetitions (default 3; smoke 1)')
    p.add_argument('--smoke', action='store_true', help='2 steps, 16 samples/walkers; CPU 1/4 threads; one measured repetition')
    p.add_argument('--dry-run', action='store_true', help='print plan without preparing, building or measuring')
    args = p.parse_args()
    cpu_options_explicit = args.threads is not None or '--image' in sys.argv
    args.reps = args.reps if args.reps is not None else (1 if args.smoke else 3)
    args.threads = args.threads if args.threads is not None else ([1, 4] if args.smoke else [1, 4, 8, 16])
    args.steps, args.samples = (2, 16) if args.smoke else (20, 320)
    if args.reps < 1 or any(n < 1 for n in args.threads) or len(set(args.threads)) != len(args.threads):
        p.error('repetitions and distinct thread counts must be positive')
    if args.mode == 'gpu' and cpu_options_explicit:
        p.error('--image and --threads are CPU options')
    if args.mode == 'gpu':
        args.image = None
        args.threads = [1]
    if args.mode == 'cpu' and args.device is not None:
        p.error('--device is a GPU option')
    if len(set(args.workloads)) != len(args.workloads):
        p.error('workloads must be distinct')
    out = (args.output or ROOT / f'target/bench/{args.mode}-{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}').resolve()
    build = (args.build_dir or ROOT / 'target/benchmark-entrypoints').resolve()
    print(f'Benchmark {args.mode}: L128 Opt/PhysCal; steps={args.steps}; samples={args.samples}; '
          f'threads={args.threads if args.mode=="cpu" else 1}; warmup=1; reps={args.reps}\nOutput: {out}', flush=True)
    if args.dry_run:
        executables = 'Rust and CPU C reference' if args.mode == 'cpu' else 'Rust CUDA examples'
        print('Plan: initialize pinned submodules; instantiate/build Julia 1.13 reference; '
              f'build {executables}; run sequential measurements; write report.md and raw evidence.\n'
              'Every Python command uses uv run --no-project python. GPU uses one selected NVIDIA device.')
        return
    if sys.platform != 'linux':
        p.error('reference benchmark requires Linux; use docs/DEV_CONTAINER.md')
    for name in ['git', 'cargo', 'rustc', 'julia', 'uv', 'taskset'] + (['docker'] if args.mode == 'cpu' else ['nvidia-smi']):
        if not shutil.which(name):
            p.error(f'missing prerequisite: {name}; see scripts/benchmark/README.md')
    if out.exists():
        p.error(f'output already exists; choose a fresh directory: {out}')
    cpus = physical_cpus()
    if args.mode == 'cpu' and max(args.threads) > len(cpus):
        p.error(f'{len(cpus)} allowed physical cores; select --threads explicitly')
    lock_path = Path('/tmp') / f'mvmc-benchmark-{os.getuid()}.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error(f'another CPU/GPU benchmark holds {lock_path}')
        out.mkdir(parents=True)
        build.mkdir(parents=True, exist_ok=True)
        env = clean_environment()
        command(['git', 'submodule', 'update', '--init', '--recursive', 'extern/Julia-mVMC'] +
                (['extern/mVMC-1.3.0'] if args.mode == 'cpu' else []), out / 'submodules.log', env)
        project = JULIA if args.mode == 'cpu' else JULIA / 'test/gpu'
        command(['julia', '+1.13', '--startup-file=no', '--threads=1', f'--project={project}',
                 TOOLS / 'prepare_julia.jl'], out / 'julia-prepare.log', dict(env, JULIA_NUM_THREADS='1'))
        locked = project / 'Manifest-v1.13.toml'
        expected_lock = command(['git', '-C', JULIA, 'show', 'HEAD:' + str(locked.relative_to(JULIA))], env=env)
        if hashlib.sha256(expected_lock.encode()).hexdigest() != sha(locked):
            raise ValueError('Julia manifest differs from pinned upstream; preserve local changes and use a clean reference')
        prepare_inputs(out, args.steps, args.samples)
        metadata = dict(args={k: str(v) for k, v in vars(args).items()}, source=command(['git','rev-parse','HEAD'], env=env).strip(),
                        status=command(['git','status','--short'], env=env), physical_cpu_ids=cpus,
                        rust_version=command(['rustc','-Vv'], env=env),
                        julia_version=command(['julia','+1.13','--startup-file=no','--version'], env=env).strip(),
                        julia_source=command(['git','-C',JULIA,'rev-parse','HEAD'], env=env).strip(),
                        julia_lock_sha256=sha(project / 'Manifest-v1.13.toml'),
                        input_sha256={str(f.relative_to(out)):sha(f) for f in (out/'inputs').rglob('*') if f.is_file()},
                        environment={k:v for k,v in env.items() if k.startswith(('MVMC_', 'JULIA_', 'OPENBLAS_', 'OMP_', 'MKL_', 'BLIS_'))})
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        rows, extra = (cpu_measure if args.mode == 'cpu' else gpu_measure)(args, out, build, env, cpus)
        metadata.update(extra)
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        write_report(args, out, rows)
        print(f'Completed. Results: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, AssertionError) as error:
        print(f'Benchmark failed: {error}', file=sys.stderr)
        sys.exit(1)
