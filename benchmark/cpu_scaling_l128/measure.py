from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, time, threading
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--image',required=True)
p.add_argument('--target-volume',required=True)
p.add_argument('--c-dir',type=Path,required=True)
p.add_argument('--opt-input',type=Path,required=True)
p.add_argument('--phys-input',type=Path,required=True)
p.add_argument('--prepare-only',action='store_true')
p.add_argument('--pf-backend',choices=['c-order','calc-m-all'],default='c-order')
p.add_argument('--rust-binary',type=Path)
p.add_argument('--only-rust',action='store_true')
p.add_argument('--threads',type=int,nargs='+',default=[1,4,8,16])
p.add_argument('--workloads',choices=['opt','phys'],nargs='+',default=['opt','phys'])
p.add_argument('--reps',type=int,default=3)
p.add_argument('--container-name',default='mvmc-586-baseline')
p.add_argument('--batch-size',type=int)
a=p.parse_args()
if a.reps < 1 or any(n < 1 for n in a.threads) or (a.batch_size is not None and a.batch_size < 1):
    p.error('thread counts, measured repetitions and batch size must be positive')
def run(cmd,**kw): return subprocess.run(cmd,check=True,text=True,capture_output=True,**kw).stdout
out=a.output.resolve(); out.mkdir()
for mode,src in [('opt',a.opt_input),('phys',a.phys_input)]:
    shutil.copytree(src,out/'inputs'/mode,ignore=shutil.ignore_patterns('output','zvo_*'))
assert hashlib.sha256((out/'inputs/phys/zqp_opt.dat').read_bytes()).hexdigest()=='9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90'
meta={'args':{k:str(v) for k,v in vars(a).items()},'source':run(['git','rev-parse','HEAD'],cwd=a.repo).strip(),'status':run(['git','status','--short'],cwd=a.repo),'image':run(['docker','image','inspect',a.image,'--format','{{.Id}}']).strip(),'lscpu':run(['lscpu']),'affinity':sorted(os.sched_getaffinity(0)),'inputs':{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'inputs').rglob('*') if f.is_file()},'scope':'Inside one persistent development container; bash time covers executable (C includes mpirun), docker exec wall recorded separately; no Docker startup included. rep0 warmup, rep1-3 measured. Same first N physical cores, no SMT siblings, BLAS1. No frequency/isolation control.'}
meta['scope']=meta['scope'].replace('rep1-3 measured',f'rep1-{a.reps} measured')
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if a.prepare_only: raise SystemExit(0)
name=a.container_name
rust_path='/rust/mvmc' if a.rust_binary else '/target/release/mvmc'
launch=['docker','run','-d','--name',name,'--shm-size=1g','-v',f'{out}:/out','-v',f'{a.target_volume}:/target:ro','-v',f'{a.c_dir.resolve()}:/c:ro']
if a.rust_binary: launch+=['-v',f'{a.rust_binary.resolve()}:/rust/mvmc:ro']
run(launch+[a.image,'sleep','infinity'])
rows=[]
stop=threading.Event()
def observe():
    with (out/'host-observer.jsonl').open('w') as f:
        while not stop.is_set():
            f.write(json.dumps({'time':time.time(),'load':os.getloadavg(),'cpu':Path('/proc/stat').read_text().splitlines()[0]})+'\n'); f.flush()
            stop.wait(1)
observer=threading.Thread(target=observe,daemon=True)
try:
    metadata=run(['docker','exec',name,'bash','-lc',f'set -eu; uname -a; rustc -Vv; gcc --version; /opt/mpich/bin/mpichversion; pkg-config --modversion openblas; ldd --version; sha256sum {rust_path} /c/build/src/mVMC/vmc.out; ldd {rust_path}; ldd /c/build/src/mVMC/vmc.out; cat /sys/fs/cgroup/cpu.max; grep Cpus_allowed_list /proc/self/status'])
    (out/'container-environment.txt').write_text(metadata)
    observer.start()
    for mode in a.workloads:
        for n in a.threads:
            for rep in range(a.reps+1):
                for kind in (['rust'] if a.only_rust else (['rust','c'] if rep%2==0 else ['c','rust'])):
                    label=f'{mode}-{kind}-t{n}-r{rep}'
                    dest=out/label
                    shutil.copytree(out/'inputs'/mode,dest)
                    cmd=([rust_path,'namelist.def'] if kind=='rust' else ['/opt/mpich/bin/mpirun','-np','1','/c/build/src/mVMC/vmc.out','namelist.def'])
                    if mode=='phys': cmd+=['zqp_opt.dat']
                    shell='set -euo pipefail; TIMEFORMAT="wall=%R user=%U sys=%S"; { time taskset -c "$1" "${@:2}" > run.stdout 2> run.stderr; } 2> time.txt'
                    env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':str(n) if kind=='c' else '1','MKL_NUM_THREADS':'1','BLIS_NUM_THREADS':'1','MVMC_RS_INNER_THREADS':str(n),'RAYON_NUM_THREADS':str(n),'MVMC_RS_SR_BACKEND':'c-order','MVMC_RS_MEASURE_PF_BACKEND':a.pf_backend,'MVMC_C_TIMER':'1'}
                    if a.batch_size is not None:
                        env['MVMC_RS_MEASURE_BATCH']=str(a.batch_size)
                    launch=['docker','exec','-w',f'/out/{label}']
                    for k,v in env.items(): launch+=['-e',f'{k}={v}']
                    launch += [name,'bash','-lc',shell,'bench',f'0-{n-1}',*cmd]
                    start=time.perf_counter(); result=subprocess.run(launch,text=True,capture_output=True); elapsed=time.perf_counter()-start
                    (dest/'launcher.stdout').write_text(result.stdout); (dest/'launcher.stderr').write_text(result.stderr)
                    if result.returncode: raise RuntimeError(f'{label}: exit {result.returncode}')
                    values=dict(item.split('=') for item in (dest/'time.txt').read_text().split())
                    if not list((dest/'output').glob('zvo_out*.dat')): raise RuntimeError(f'{label}: missing output')
                    row={'mode':mode,'kind':kind,'threads':n,'rep':rep,'docker_exec_wall':elapsed,**{k:float(v) for k,v in values.items()}}
                    row['cpu_percent']=100*(row['user']+row['sys'])/row['wall']
                    rows.append(row); (out/'timings.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(json.dumps(row),flush=True)
finally:
    stop.set()
    if observer.is_alive(): observer.join()
    (out/'container-host-config.json').write_text(run(['docker','inspect',name,'--format','{{json .HostConfig}}']))
    (out/'cpu-max-final.txt').write_text(run(['docker','exec',name,'cat','/sys/fs/cgroup/cpu.max']))
    run(['docker','rm','-f',name])
