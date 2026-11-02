"""Extra diagnostic runs: separate from the 64 uninstrumented baseline timings."""
import argparse,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--image',required=True)
p.add_argument('--target-volume',required=True)
p.add_argument('--c-dir',type=Path,required=True)
p.add_argument('--only-rust',action='store_true')
p.add_argument('--threads',default='1,4,8,16')
a=p.parse_args()
threads=[int(value) for value in a.threads.split(',')]
assert threads and len(set(threads))==len(threads) and all(n in [1,4,8,16] for n in threads)
rows=json.loads((a.baseline/'timings.json').read_text())
baseline_meta=json.loads((a.baseline/'metadata.json').read_text())
kinds=['rust'] if baseline_meta['args'].get('only_rust')=='True' else ['rust','c']
expected={(m,k,n,r) for m in ['opt','phys'] for k in kinds for n in [1,4,8,16] for r in range(4)}
assert len(rows)==len(expected) and {(r['mode'],r['kind'],r['threads'],r['rep']) for r in rows}==expected
assert a.only_rust or 'c' in kinds, 'use --only-rust for a Rust-only baseline'
out=a.out.resolve(); out.mkdir()
plan=Path(__file__).resolve().parent
name='mvmc-586-profile'
def run(cmd):
    return subprocess.run(cmd,check=True,capture_output=True,text=True).stdout
run(['docker','run','-d','--name',name,'--shm-size=1g','-v',f'{out}:/out','-v',f'{plan}:/plan:ro','-v',f'{a.target_volume}:/target:ro','-v',f'{a.c_dir.resolve()}:/c:ro',a.image,'sleep','infinity'])
try:
    run(['docker','exec',name,'cp','/target/release/mvmc','/out/measured-mvmc'])
    hashes=run(['docker','exec',name,'sha256sum','/target/release/mvmc','/c/build/src/mVMC/vmc.out'])
    baseline_env=(a.baseline/'container-environment.txt').read_text()
    for line in hashes.splitlines():
        assert line.split()[0] in baseline_env, 'profiling binary differs from baseline'
    (out/'binary-sha256.txt').write_text(hashes)
    for mode in ['opt','phys']:
        for n in threads:
            for kind in (['rust'] if a.only_rust else ['rust','c']):
                label=f'{mode}-{kind}-t{n}'
                shutil.copytree(a.baseline/'inputs'/mode,out/label)
                command=(['/target/release/mvmc','namelist.def'] if kind=='rust' else ['/opt/mpich/bin/mpirun','-np','1','/c/build/src/mVMC/vmc.out','namelist.def'])
                if mode=='phys': command+=['zqp_opt.dat']
                env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':str(n) if kind=='c' else '1','MKL_NUM_THREADS':'1','BLIS_NUM_THREADS':'1','MVMC_RS_INNER_THREADS':str(n),'RAYON_NUM_THREADS':str(n),'MVMC_RS_SR_BACKEND':'c-order','MVMC_RS_MEASURE_PF_BACKEND':'c-order','MVMC_C_TIMER':'1'}
                if kind=='rust': env.update(MVMC_RS_INNER_PROFILE='1',MVMC_RS_INNER_GAP_PROFILE='1',MVMC_RS_INNER_GAP_SITE_PROFILE='1',MVMC_RS_INNER_OBSERVE='1')
                cmd=['docker','exec','-w',f'/out/{label}']
                for k,v in env.items(): cmd+=['-e',f'{k}={v}']
                cmd +=[name,'uv','run','--no-project','--python','/usr/bin/python3','/plan/thread_profile.py','--out',f'/out/{label}-profile','--','taskset','-c',f'0-{n-1}',*command]
                result=subprocess.run(cmd,text=True,capture_output=True)
                (out/f'{label}.launcher.stdout').write_text(result.stdout)
                (out/f'{label}.launcher.stderr').write_text(result.stderr)
                if result.returncode: raise RuntimeError(f'{label}: diagnostic exit {result.returncode}')
                if not list((out/label/'output').glob('zvo_out*.dat')): raise RuntimeError(f'{label}: missing output')
                print(label,'complete',flush=True)
finally:
    run(['docker','rm','-f',name])
