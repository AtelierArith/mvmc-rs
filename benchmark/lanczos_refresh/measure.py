"""Reproduce #630: sequential warm Julia API and Rust CLI Lanczos timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, default=Path.cwd())
p.add_argument('--rust', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
repo, binary, out = a.repo.resolve(), a.rust.resolve(), a.output.resolve()
out.mkdir(parents=True)
def capture(cmd):
    return subprocess.check_output(cmd, cwd=repo, text=True).strip()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
meta = {'rust_source': capture(['git', 'rev-parse', 'HEAD']),
        'rust_status': capture(['git', 'status', '--short']),
        'julia_source': capture(['git', '-C', 'extern/Julia-mVMC', 'rev-parse', 'HEAD']),
        'rust_binary': str(binary), 'rust_binary_sha256': digest(binary),
        'rustc': capture(['rustc', '-Vv']), 'julia': capture(['julia', '+1.13.1', '--version']),
        'lscpu': capture(['lscpu']), 'ldd_rust': capture(['ldd', str(binary)]),
        'scope': 'Rust CLI launch-to-exit; Julia in-process public API, GC and first invocation excluded. Each condition warmup1+measured3. Sequential. No CPU affinity/frequency isolation, matching original Lanczos procedure.',
        'load_before': os.getloadavg(), 'inputs': {}}
cases = {'phys_lanczos': 'tests/fixtures/physcal_181/hubbard_all_terms_lanczos1',
         'phys_interall_lanczos1': 'tests/fixtures/native_c_lanczos_576/interall_real1',
         'phys_interall_lanczos2': 'tests/fixtures/native_c_lanczos_576/interall_real2'}
for name, rel in cases.items():
    src = repo / rel
    dst = out / 'inputs' / name
    shutil.copytree(src / 'inputs', dst)
    shutil.copy2(src / 'zqp_opt.dat', dst / 'zqp_opt.dat')
    mod = dst / 'modpara.def'
    content = mod.read_text()
    for key, value in [('NVMCSample', 20000), ('NDataQtySmp', 20), ('NVMCCalMode', 1)]:
        content, count = re.subn(rf'(?m)^{key}\s+.*$', f'{key} {value}', content)
        assert count == 1, (key, count)
    mod.write_text(content)
    meta['inputs'][name] = {str(f.relative_to(dst)): digest(f) for f in sorted(dst.rglob('*')) if f.is_file()}
(out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
rows = []
for n in [1, 4, 8, 16]:
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               MVMC_RS_INNER_THREADS=str(n), RAYON_NUM_THREADS=str(n), MVMC_RS_SR_BACKEND='c-order',
               MVMC_RS_MEASURE_PF_BACKEND='c-order', JULIA_NUM_THREADS=str(n),
               JULIA_MVMC_INNER_THREADS='1', JULIA_MVMC_PFAPACK_THREADS='1', MVMC_C_TIMER='1')
    for name in cases:
        src = out / 'inputs' / name
        dest = out / f'{n}threads' / name
        dest.mkdir(parents=True)
        for kind in (['rust', 'julia'] if n in [1, 8] else ['julia', 'rust']):
            if kind == 'rust':
                for rep in range(4):
                    work = dest / f'rust-{rep}'
                    shutil.copytree(src, work)
                    before = os.getloadavg()
                    start = time.perf_counter()
                    with (work / 'run.stdout').open('w') as stdout, (work / 'run.stderr').open('w') as stderr:
                        subprocess.run([str(binary), 'namelist.def', 'zqp_opt.dat'], cwd=work, env=env, stdout=stdout, stderr=stderr, check=True)
                    seconds = time.perf_counter() - start
                    files = list((work / 'output').glob('zvo_ls_qqqq_*.dat'))
                    assert len(files) == 20 and all(f.stat().st_size for f in files)
                    row = dict(threads=n, case=name, impl=kind, rep=rep, seconds=seconds, load_before=before, load_after=os.getloadavg())
                    rows.append(row)
                    (out / 'timings.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print(json.dumps(row), flush=True)
            else:
                cmd = ['julia', '+1.13.1', f'--project={repo}/extern/Julia-mVMC', '--startup-file=no', f'--threads={n}',
                       str(repo / 'benchmark/lanczos_refresh/julia_physcal.jl'), str(src), str(dest)]
                with (dest / 'julia.log').open('w') as log:
                    subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                for rep, sec in re.findall(r'BENCH rep=(\d+) seconds=([\d.eE+-]+)', (dest / 'julia.log').read_text()):
                    row = dict(threads=n, case=name, impl=kind, rep=int(rep), seconds=float(sec))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                assert len([r for r in rows if r['threads'] == n and r['case'] == name and r['impl'] == kind]) == 4
                (out / 'timings.json').write_text(json.dumps(rows, indent=2) + '\n')
        summary = [dict(threads=t, case=c, impl=k, median=statistics.median(r['seconds'] for r in rows if r['threads']==t and r['case']==c and r['impl']==k and r['rep']>0))
                   for t,c,k in sorted({(r['threads'],r['case'],r['impl']) for r in rows})]
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
meta['load_after'] = os.getloadavg()
(out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
