from pathlib import Path
import hashlib, json, math, os, re, shutil, statistics, subprocess, sys

sys.path.insert(0, '/workspaces/mvmc-rs/scripts')
from bench_c_mpi import validate_world
from bench_cpu_round import parse_timer

cache = Path('/home/vscode/.cache/mvmc')
out = cache / 'issue496-direct-sr-c-confirmation'
out.mkdir(exist_ok=False)
cbin = cache / 'c-benchmark-independent-20261010/build/src/mVMC/vmc.out'
rbin = cache / 'target/issue496-direct-sr-final/release/examples/mpi_benchmark'
observer = cache / 'mpi-world-observer-495.so'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(cbin) == '6ae44625e227ccd6b9c38d2a7731fb445955108772e8741b452f70c6a641ac28'
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', BLIS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', MVMC_RS_INNER_THREADS='4',
           UCX_MEMTYPE_CACHE='no', UCX_ERROR_SIGNALS='SIGILL,SIGBUS,SIGFPE')
env['LD_LIBRARY_PATH'] = '/opt/mpich/lib:' + env.get('LD_LIBRARY_PATH', '')
for key in ('MVMC_C_TIMER', 'MVMC_RS_INNER_PROFILE', 'MVMC_CALHAM_DIAGNOSTICS',
            'MVMC_RS_INNER_THRESHOLD', 'MVMC_RS_INNER_MIN_WORK_NS',
            'MVMC_RS_INNER_MIN_SIZE', 'LD_PRELOAD', 'MVMC_TIMER',
            'MVMC_MAINCAL_DIAG', 'MVMC_WEIGHTAVG_DIAG', 'MVMC_CALHAM1_DIAG',
            'MVMC_SLATER_DIAG', 'MVMC_RS_SR_IN_PLACE_PROBE'):
    env.pop(key, None)
env['MVMC_RS_MEASURE_PF_BACKEND']='calc-m-all'
env['MVMC_RS_SR_PF_BACKEND']='c-order'
provenance = dict(base_head='f04cfca70b1680e60995ac2444fe651a0eba003f',
                  source_patch=(cache/'issue496-direct-sr-final.patch').read_text(),
                  final_sources={name:digest(cache/'issue496-direct-sr-final/crates/mvmc-core/src'/name) for name in ('run.rs','sr_accumulator.rs','state.rs')},
                  binaries={str(p): digest(p) for p in (cbin,rbin,observer)},
                  environment={k:v for k,v in env.items() if k.startswith(('MVMC_', 'OMP_', 'OPENBLAS_', 'BLIS_', 'MKL_', 'UCX_'))},
                  policy='Final direct SR candidate: three alternating C/Rust batches per size; Rust full warmup1, measurement1 per batch; C fresh process, initial discarded warmup per size. NSROptItrSmp=300 for both.')
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
rows=[]
output_differences=[]
def check_output(path, reference, size, implementation, rep):
    values=[[float(x) for x in line.split()] for line in path.read_text().splitlines()]
    expected=[[float(x) for x in line.split()] for line in reference.read_text().splitlines()]
    assert len(values)==len(expected)==300
    assert all(len(row)==6 and all(math.isfinite(x) for x in row) for row in values)
    output_differences.append(dict(sites=size,implementation=implementation,rep=rep,
        max_abs_all_columns=max(abs(x-y) for a,b in zip(values,expected) for x,y in zip(a,b))))
    (out/'numerical-output-comparison.json').write_text(json.dumps(output_differences,indent=2)+'\n')
def c_run(size, rep):
    dest=out/f'L{size}-c-{rep}'
    dest.mkdir()
    source=Path('/workspaces/mvmc-rs/bench-out/issue502-production-opt-physcal-20261010')/f'L{size}/Opt-C/run-1'
    for p in source.glob('*.def'): shutil.copy2(p,dest/p.name)
    (dest/'output').mkdir()
    ce=dict(env,OMP_NUM_THREADS='4',LD_PRELOAD=str(observer))
    with (dest/'run.log').open('w') as f:
        subprocess.run(['/opt/mpich/bin/mpiexec','-n','4',str(cbin),'namelist.def'],
                       cwd=dest,env=ce,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    validate_world((dest/'run.log').read_text(),4,4)
    seconds=parse_timer(dest/'zvo_CalcTimer.dat')[0]
    data=next((dest/'output').glob('zvo_out_*.dat')).read_text().splitlines()
    assert len(data)==300
    check_output(next((dest/'output').glob('zvo_out_*.dat')),
                 next((source/'output').glob('zvo_out_*.dat')),size,'c',rep)
    return seconds
def rust_run(size, rep):
    dest=out/f'L{size}-rust-{rep}'
    dest.mkdir()
    label='main-07a2f9f9-julia-3254752e-4x4-rj' if size==32 else 'main-07a2f9f9-julia-3254752e-L64-4x4-rj'
    namelist=Path('/workspaces/mvmc-rs/bench-out/issue502-production-opt-physcal-20261010')/f'L{size}/opt-inputs/namelist.def'
    with (dest/'run.log').open('w') as f:
        subprocess.run(['/opt/mpich/bin/mpiexec','-n','4',str(rbin),str(namelist),
                        '300','1','1',str(dest),'4'],env=env,stdout=f,
                        stderr=subprocess.STDOUT,check=True,timeout=300)
    log=(dest/'run.log').read_text()
    validate_world(log,4,4)
    result=re.findall(r'^BENCH 1 ([0-9.]+) (\S+)$',log,re.M)
    assert len(result)==1
    assert len((dest/'run-1/zvo_out.dat').read_text().splitlines())==300
    check_output(dest/'run-1/zvo_out.dat',Path('/workspaces/mvmc-rs/bench-out/issue502-production-opt-physcal-20261010')/f'L{size}/Opt-Rust/run-1/zvo_out.dat',
                 size,'rust',rep)
    return float(result[0][0])
for size in (32,64):
    c_run(size,0)
    for rep in (1,2,3):
        order=('c','rust') if rep%2 else ('rust','c')
        for implementation in order:
            seconds=(c_run if implementation=='c' else rust_run)(size,rep)
            row=dict(sites=size,implementation=implementation,rep=rep,seconds=seconds)
            rows.append(row)
            print(json.dumps(row),flush=True)
            (out/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=[dict(sites=size,implementation=implementation,
              median_seconds=statistics.median(r['seconds'] for r in rows
                  if r['sites']==size and r['implementation']==implementation))
         for size in (32,64) for implementation in ('c','rust')]
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert provenance['binaries']=={str(p):digest(p) for p in (cbin,rbin,observer)}
print(json.dumps(summary),flush=True)
