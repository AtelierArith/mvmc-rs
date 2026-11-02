"""Repeat the public Rust PhysCal API; rep0 warmup, rep1-3 measured."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--backend',choices=['cpu','gpu'],required=True);p.add_argument('--threads',default='1');p.add_argument('--device',type=int,default=1);p.add_argument('--output-files',action='store_true');a=p.parse_args()
threads=list(map(int,a.threads.split(',')));assert all(n>0 for n in threads)
assert a.backend=='cpu' or threads==[1], 'GPU comparison uses one host thread'
out=a.out.resolve();out.mkdir();binary=a.binary.resolve();inp=a.input.resolve()
metadata={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'input_hashes':{str(f.relative_to(inp)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inp.rglob('*') if f.is_file()},'backend':a.backend,'threads':threads,'output_files':a.output_files,'scope':'prepare and run PhysCal API per repetition; initial pool/process startup excluded; rep0 warmup'}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');results=[]
for n in threads:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.device),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',RAYON_NUM_THREADS=str(n),MVMC_RS_INNER_THREADS=str(n),MVMC_RS_CUDA_RNG='sfmt')
 for key in ['MVMC_RS_SAMPLER_PROFILE','MVMC_RS_CUDA_INIT_PROFILE','MVMC_RS_INNER_PROFILE','MVMC_RS_INNER_OBSERVE']:env.pop(key,None)
 for selector in ['SR','MEASURE_PF','MEASURE','SAMPLER']:env[f'MVMC_RS_{selector}_BACKEND']='cuda' if a.backend=='gpu' else 'c-order'
 cmd=['taskset','-c',f'0-{n-1}',str(binary),str(inp/'namelist.def'),str(inp/'zqp_opt.dat'),str(out/f't{n}')]
 if not a.output_files:cmd.append('--no-output')
 with (out/f't{n}.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(n,result.returncode)
 rows=re.findall(r'BENCH rep=(\d+) seconds=([\d.]+)',(out/f't{n}.log').read_text());assert len(rows)==4
 results.extend({'threads':n,'rep':int(rep),'seconds':float(seconds)} for rep,seconds in rows)
 (out/'timings.json').write_text(json.dumps(results,indent=2)+'\n')
