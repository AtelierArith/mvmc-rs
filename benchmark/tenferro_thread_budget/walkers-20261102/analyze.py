import json, math, re, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]); result={}
for mode in ('opt','phys'):
 result[mode]={}
 for walkers in (1,4):
  rec={}; timing={}
  for eng in (1,4):
   samples=[]
   for rep in (1,2,3):
    label=f'{mode}-w{walkers}-e{eng}-r{rep}'
    text=(root/f'{label}.stdout').read_text()
    discrete=[s for s in text.splitlines() if s.startswith(('D|','H|'))]
    nums=[s for s in text.splitlines() if s.startswith('N|')]
    if len(discrete)!=walkers or len(nums)!=walkers: raise ValueError(f'{label}: missing records')
    values=[]
    for line in nums:
     if mode == "opt" and not line.startswith("N|0|"):
      if line.split("|",2)[2] != "NaN": raise ValueError(f"{label}: expected non-root NaN sentinel")
      continue
     values += [float(x) for x in line.split('|',2)[2].split()]
    if not all(map(math.isfinite,values)): raise ValueError(f'{label}: nonfinite')
    rec[(eng,rep)]=(discrete,values)
    match=re.search(r'wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)',(root/f'{label}.time').read_text())
    if not match: raise ValueError(f'{label}: timing')
    w,u,s=map(float,match.groups()); samples.append(dict(wall=w,user=u,system=s))
   timing[str(eng)]={'samples':samples,**{f'{k}_median':statistics.median(v[k] for v in samples) for k in ('wall','user','system')}}
  refd,refn=rec[(1,1)];
  if any(len(n) != len(refn) for d,n in rec.values()): raise ValueError(f"{mode} {walkers}: numeric count mismatch")
  same=all(d==refd for d,n in rec.values()); max_abs=max(abs(a-b) for d,n in rec.values() for a,b in zip(n,refn));
  result[mode][str(walkers)]={'discrete_match':same,'numeric_max_abs':max_abs,'numeric_components':len(refn),'timings':timing}
print(json.dumps(result,indent=2))
