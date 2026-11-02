#!/usr/bin/env python3
"""Sample one 4-walker Opt process; diagnostic only, not timing."""
import json, os, subprocess, sys, time
from pathlib import Path
image, volume, inputs, output, engine, lifetime = sys.argv[1:]
inputs=str(Path(inputs).resolve()); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
name=f'mvmc-583-walker-observe-{os.getpid()}'
cmd=['docker','run','--rm','--name',name,'--cpuset-cpus=0-3','--shm-size=1g','-v',f'{inputs}:/inputs:ro','-v',f'{output}:/out','-v',f'{volume}:/target:ro','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','RAYON_NUM_THREADS=4','-e','MVMC_RS_INNER_THREADS=1','-e',f'MVMC_RS_TENFERRO_THREADS={engine}','-e','MVMC_RS_SR_BACKEND=tenferro','-e',f'MVMC_RS_TENFERRO_TEAM_LIFETIME={lifetime}','--entrypoint','/target/release/examples/bench_walkers_583',image,'opt','/inputs/opt/namelist.def','/inputs/phys/zqp_opt.dat','4','/out/opt']
previous={}; seen=set(); samples=[]
with (output/'run.txt').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 try:
  while proc.poll() is None:
   listing=subprocess.run(['docker','top',name,'-eo','pid,args'],capture_output=True,text=True)
   now=time.monotonic()
   for line in listing.stdout.splitlines()[1:]:
    fields=line.strip().split(None,1)
    if len(fields)!=2 or not fields[1].startswith('/target/release/examples/bench_walkers_583 '): continue
    pid=int(fields[0]); tasks=Path(f'/proc/{pid}/task')
    try: entries=list(tasks.iterdir())
    except FileNotFoundError: continue
    active=[]; count=0
    for task in entries:
     try:
      stat=(task/'stat').read_text(); fields=stat[stat.rfind(')')+2:].split(); ticks=int(fields[11])+int(fields[12]); tid=int(task.name)
     except (FileNotFoundError,ProcessLookupError): continue
     count+=1
     if tid in previous and ticks>previous[tid]: active.append(tid); seen.add(tid)
     previous[tid]=ticks
    samples.append(dict(monotonic=now,pid=pid,thread_count=count,active_tids=active))
   time.sleep(.1)
 finally:
  if proc.poll() is None:
   subprocess.run(['docker','stop',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); proc.wait()
if proc.returncode: raise SystemExit(f'container failed: {proc.returncode}')
if not seen: raise SystemExit('no CPU-active threads observed')
report=dict(engine_threads=int(engine),clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),distinct_active_tids=sorted(seen),samples=samples)
(output/'thread-samples.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(engine=engine,samples=len(samples),distinct_active=len(seen),peak_threads=max(s['thread_count'] for s in samples),peak_active=max(len(s['active_tids']) for s in samples))))
