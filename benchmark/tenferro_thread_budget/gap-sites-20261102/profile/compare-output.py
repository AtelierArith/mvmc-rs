import json,math,sys
from pathlib import Path
p=Path(sys.argv[1]); ref=Path(sys.argv[2]); result={}
def nums(path):
 vals=[float(x) for line in path.read_text().splitlines() if not line.lstrip().startswith('#') for x in line.split()]
 if not all(map(math.isfinite,vals)): raise ValueError(f'nonfinite {path}')
 return vals
for case in ('l32-opt','l32-phys','l64-opt'):
 result[case]={}
 for workers in (1,4):
  a=p/f'{case}-{workers}-output'; b=ref/f'{case}-g40000-r1-output'
  names=sorted(x.name for x in a.iterdir() if x.is_file() and '_time_' not in x.name)
  if not names or names!=sorted(x.name for x in b.iterdir() if x.is_file() and '_time_' not in x.name): raise ValueError(f'{case} {workers}: files')
  mx=0.0
  for name in names:
   left,right=nums(a/name),nums(b/name)
   if len(left)!=len(right): raise ValueError(f'{case} {name}: count')
   mx=max(mx,max((abs(x-y) for x,y in zip(left,right)),default=0.0))
  result[case][str(workers)]={'files':len(names),'max_abs_from_reference':mx}
print(json.dumps(result,indent=2))
