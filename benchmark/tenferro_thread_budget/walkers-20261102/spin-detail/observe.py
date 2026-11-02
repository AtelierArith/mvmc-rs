#!/usr/bin/env python3
"""Sample process thread CPU ticks, switch counts and wait channels; diagnostic, not timing."""
import collections,json,os,subprocess,sys,time
from pathlib import Path
image,volume,inputs,output,lifetime,spin=sys.argv[1:]
inputs=str(Path(inputs).resolve()); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
name=f'mvmc-583-detail-{os.getpid()}'
cmd=['docker','run','--rm','--name',name,'--cpuset-cpus=0-3','--shm-size=1g','-v',f'{inputs}:/inputs:ro','-v',f'{output}:/out','-v',f'{volume}:/target:ro','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','RAYON_NUM_THREADS=4','-e','MVMC_RS_INNER_THREADS=1','-e','MVMC_RS_TENFERRO_THREADS=4','-e','MVMC_RS_SR_BACKEND=tenferro','-e',f'MVMC_RS_TENFERRO_TEAM_LIFETIME={lifetime}',*(['-e',f'MVMC_RS_TENFERRO_SPIN_POLLS={spin}'] if spin!='default' else []),'--entrypoint','/target/release/examples/bench_walkers_583',image,'opt','/inputs/opt/namelist.def','/inputs/phys/zqp_opt.dat','4','/out/opt']
previous={}; samples=[]
with (output/'run.txt').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 try:
  while proc.poll() is None:
   listing=subprocess.run(['docker','top',name,'-eo','pid,args'],capture_output=True,text=True); now=time.monotonic()
   for line in listing.stdout.splitlines()[1:]:
    fields=line.strip().split(None,1)
    if len(fields)!=2 or not fields[1].startswith('/target/release/examples/bench_walkers_583 '): continue
    pid=int(fields[0]); tasks=Path(f'/proc/{pid}/task')
    try: entries=list(tasks.iterdir())
    except FileNotFoundError: continue
    for task in entries:
     try:
      raw=(task/'stat').read_text(); name0=raw[raw.find('(')+1:raw.rfind(')')]; fs=raw[raw.rfind(')')+2:].split(); tid=int(task.name); cpu=int(fs[11])+int(fs[12]); state=fs[0]
      status=(task/'status').read_text().splitlines(); counts={k.strip():int(v) for line0 in status if ':' in line0 for k,v in [line0.split(':',1)] if k.strip() in ('voluntary_ctxt_switches','nonvoluntary_ctxt_switches')}
      try: wait=(task/'wchan').read_text().strip()
      except PermissionError: wait='unavailable'
     except (FileNotFoundError,ProcessLookupError): continue
     if len(counts)!=2: continue
     curr=(cpu,counts['voluntary_ctxt_switches'],counts['nonvoluntary_ctxt_switches']); old=previous.get(tid); previous[tid]=curr
     delta=None if old is None else [a-b for a,b in zip(curr,old)]
     samples.append(dict(monotonic=now,tid=tid,name=name0,state=state,wchan=wait,cpu_ticks=cpu,voluntary=curr[1],involuntary=curr[2],delta=delta))
   time.sleep(.1)
 finally:
  if proc.poll() is None: subprocess.run(['docker','stop',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); proc.wait()
if proc.returncode: raise SystemExit(f'container failed: {proc.returncode}')
if not samples: raise SystemExit('no thread samples')
report=dict(lifetime=lifetime,spin=spin,clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),samples=samples)
(output/'detail-samples.json').write_text(json.dumps(report,indent=2)+'\n')
names=collections.Counter(s['name'] for s in samples); waits=collections.Counter(s['wchan'] for s in samples); totals=[sum(s['delta'][i] for s in samples if s['delta'] is not None) for i in range(3)]
print(json.dumps(dict(lifetime=lifetime,spin=spin,thread_samples=len(samples),names=names,wait_channels=waits,observed_delta_ticks_vol_invol=totals)))
