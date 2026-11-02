import json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); hz=int((root/'ticks.txt').read_text()); report={}
for threads in (1,4):
 report[str(threads)]={}
 for inside in (0,1):
  bins={ms:[] for ms in (10,50,100,500,1000)}
  for rep in (1,2,3,4,5):
   lines=(root/f't{threads}-i{inside}-r{rep}.log').read_text().splitlines()
   seen=set()
   for line in lines:
    if not line.startswith('idle '): continue
    fields=dict(re.findall(r'(\w+)=([\w.]+)',line)); ms=int(fields['requested_ms']); seen.add(ms)
    bins[ms].append({'elapsed_s':float(fields['elapsed_s']),'process_ticks':int(fields['process_ticks']),'caller_ticks':int(fields['caller_ticks']),'other_ticks':int(fields['other_ticks'])})
   if seen!=set(bins): raise ValueError(f'missing timings: {threads} {inside} {rep}')
  report[str(threads)][str(inside)]={str(ms):{'samples':samples,'median_other_cpu_s':statistics.median(s['other_ticks'] for s in samples)/hz} for ms,samples in bins.items()}
print(json.dumps({'ticks_per_second':hz,'results':report},indent=2))
