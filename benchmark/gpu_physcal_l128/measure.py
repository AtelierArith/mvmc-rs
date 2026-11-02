"""Uninstrumented CLI repetitions followed by a separate diagnostic run."""
import argparse,hashlib,json,os,shutil,subprocess,time,resource
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--binary',type=Path,required=True); p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--cold-cache',action='store_true'); p.add_argument('--device',type=int,default=1); p.add_argument('--cpu',type=int,default=0); a=p.parse_args()
out=a.out.resolve(); out.mkdir(); binary=a.binary.resolve(); inp=a.input.resolve()
env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(a.device),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',MVMC_RS_INNER_THREADS='1',MVMC_RS_SR_BACKEND='cuda',MVMC_RS_MEASURE_PF_BACKEND='cuda',MVMC_RS_MEASURE_BACKEND='cuda',MVMC_RS_SAMPLER_BACKEND='cuda',MVMC_C_TIMER='1',MVMC_RS_CUDA_RNG='sfmt')
for k in ['MVMC_RS_SAMPLER_PROFILE','MVMC_RS_CUDA_INIT_PROFILE','MVMC_RS_INNER_PROFILE','MVMC_RS_INNER_OBSERVE']: env.pop(k,None)
metadata={'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'input_sha256':{str(f.relative_to(inp)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inp.rglob('*') if f.is_file()},'environment':{k:v for k,v in env.items() if k.startswith(('MVMC','CUDA_VISIBLE','OPENBLAS','OMP_','RAYON','LD_LIBRARY'))},'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_status':subprocess.check_output(['git','status','--short'],text=True),'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cold_cache_each_process':a.cold_cache,'scope':'process wall includes CLI startup/exit; rep0 warmup; diagnostic excluded'}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n'); rows=[]
for rep in range(5):
 dest=out/('diagnostic' if rep==4 else f'rep{rep}'); shutil.copytree(inp,dest)
 case_env=env.copy()
 if a.cold_cache: case_env['MVMC_RS_CUDA_KERNEL_CACHE']=str(dest/'kernel-cache')
 if rep==4: case_env['MVMC_RS_SAMPLER_PROFILE']='1'
 usage_start=resource.getrusage(resource.RUSAGE_CHILDREN)
 start=time.perf_counter()
 with (dest/'stdout.txt').open('w') as stdout,(dest/'stderr.txt').open('w') as stderr:
  result=subprocess.run(['taskset','-c',str(a.cpu),str(binary),'namelist.def','zqp_opt.dat'],cwd=dest,env=case_env,stdout=stdout,stderr=stderr)
 seconds=time.perf_counter()-start
 usage_end=resource.getrusage(resource.RUSAGE_CHILDREN)
 assert result.returncode==0, (rep,result.returncode)
 assert list((dest/'output').glob('zvo_out*.dat')), 'missing output'
 row={'rep':rep,'seconds':seconds,'diagnostic':rep==4,'user':usage_end.ru_utime-usage_start.ru_utime,'sys':usage_end.ru_stime-usage_start.ru_stime,'cache_directory':case_env.get('MVMC_RS_CUDA_KERNEL_CACHE','default')}; rows.append(row); (out/'timings.json').write_text(json.dumps(rows,indent=2)+'\n'); print(row,flush=True)
