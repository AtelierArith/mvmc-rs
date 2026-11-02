import json
from pathlib import Path
import re
import statistics
import sys
root=Path(sys.argv[1]); result=[]
for callers in (1,4):
 for gap in (0,60):
  for spin in ('default','0'):
   runs=[]
   for rep in range(4):
    s=(root/f'c{callers}-g{gap}-s{spin}-r{rep}.log').read_text()
    metrics={k:int(v) for k,v in re.findall(r'(calls|sum|lock_ns|dispatch_ns)=(\d+)',s)}
    assert metrics['calls']==callers*1000
    assert metrics['sum']==metrics['calls']*2080
    times={k:float(v) for k,v in re.findall(r'(wall|user|sys)=([0-9.]+)',s)}
    assert set(times)=={'wall','user','sys'}
    if rep:runs.append(dict(**times,lock_ns_per_call=metrics['lock_ns']/metrics['calls'],dispatch_ns_per_call=metrics['dispatch_ns']/metrics['calls']))
   result.append(dict(callers=callers,gap_us=gap,spin=spin,samples=runs,medians={k:statistics.median(r[k] for r in runs) for k in runs[0]}))
print(json.dumps(result,indent=2))
