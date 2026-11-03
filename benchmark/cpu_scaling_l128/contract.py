"""Correctness probes, not performance measurements: full RNG/discrete equality."""
import argparse,hashlib,json,math,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--image',required=True)
p.add_argument('--before-volume',required=True)
after=p.add_mutually_exclusive_group(required=True)
after.add_argument('--after-volume')
after.add_argument('--after-dir',type=Path,help='bind a native Cargo target directory instead of a named volume')
p.add_argument('--reference',type=Path,help='reuse previously captured before records on the same environment')
a=p.parse_args(); out=a.out.resolve(); out.mkdir()
def run(cmd): return subprocess.run(cmd,check=True,capture_output=True,text=True).stdout
name='mvmc-586-contract'
after_target=str(a.after_dir.resolve()) if a.after_dir else a.after_volume
run(['docker','run','-d','--name',name,'--shm-size=1g','-v',f'{out}:/out','-v',f'{a.inputs.resolve()}:/inputs:ro','-v',f'{a.before_volume}:/before:ro','-v',f'{after_target}:/after:ro',a.image,'sleep','infinity'])
checks=[]
try:
    hashes=run(['docker','exec',name,'sha256sum','/before/release/examples/cpu_scaling_contract_586','/after/release/examples/cpu_scaling_contract_586'])
    (out/'binary-sha256.txt').write_text(hashes)
    if a.reference:
        expected_before=(a.reference/'binary-sha256.txt').read_text().splitlines()[0].split()[0]
        assert hashes.splitlines()[0].split()[0]==expected_before, 'cached before executable changed'
    for mode in ['opt','phys']:
        reference=None
        cases=[('before',1,0),('after',1,0),('after',4,0),('after',8,0),('after',16,0),('after',16,1)]
        if a.reference:
            source=a.reference/f'{mode}-before-t1-r0/contract.json'
            shutil.copyfile(source,out/f'{mode}-before-reference.json')
            reference=json.loads(source.read_text())
            (out/f'{mode}-before-reference.sha256').write_text(hashlib.sha256(source.read_bytes()).hexdigest()+'\n')
            cases=cases[1:]
        for version,n,rep in cases:
            label=f'{mode}-{version}-t{n}-r{rep}'
            env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','BLIS_NUM_THREADS':'1','MVMC_RS_INNER_THREADS':str(n),'RAYON_NUM_THREADS':str(n),'MVMC_RS_SR_BACKEND':'c-order','MVMC_RS_MEASURE_PF_BACKEND':'c-order','MVMC_C_TIMER':'1'}
            cmd=['docker','exec']
            for k,v in env.items(): cmd+=['-e',f'{k}={v}']
            cmd += [name,'taskset','-c',f'0-{n-1}',f'/{version}/release/examples/cpu_scaling_contract_586',mode,f'/inputs/{mode}/namelist.def','/inputs/phys/zqp_opt.dat',f'/out/{label}']
            result=subprocess.run(cmd,text=True,capture_output=True)
            (out/f'{label}.stdout').write_text(result.stdout); (out/f'{label}.stderr').write_text(result.stderr)
            if result.returncode: raise RuntimeError(f'{label}: probe exited {result.returncode}')
            current=json.loads((out/label/'contract.json').read_text())
            if reference is None: reference=current
            if current['exact']!=reference['exact']:
                different=[key for key in reference['exact'] if current['exact'].get(key)!=reference['exact'][key]]
                raise AssertionError(f'{label}: exact RNG/discrete divergence in {different}; preserve both raw records and locate the first divergence')
            expected=reference['numerical']; actual=current['numerical']
            assert len(actual)==len(expected) and actual, f'{label}: numerical shape mismatch'
            maximum=0.0
            for i,(x,y) in enumerate(zip(actual,expected)):
                if not (math.isfinite(x) and math.isfinite(y)): raise AssertionError(f'{label}: non-finite numerical field {i}')
                error=abs(x-y); maximum=max(maximum,error)
                if error>1e-11+1e-11*max(abs(x),abs(y)): raise AssertionError(f'{label}: numerical field {i}: {x} vs {y}; locate first divergence')
            checks.append({'case':label,'exact_rng_discrete_match':True,'numerical_fields':len(actual),'max_absolute_difference':maximum,'decision_count':len(current['exact']['decisions']),'rng_words_consumed':current['exact']['rng_words_consumed']})
            (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
            print(json.dumps(checks[-1]),flush=True)
finally:
    run(['docker','rm','-f',name])
