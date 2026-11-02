import json,math,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); records={}; timing={}
for life in ('held','stage'):
 timing[life]={}
 for spin in ('default','0'):
  samples=[]
  for rep in (1,2,3):
   label=f'opt-{life}-p{spin}-r{rep}'
   lines=(root/f'{label}.stdout').read_text().splitlines()
   discrete=[s for s in lines if s.startswith(('D|','H|'))]; numeric=[s for s in lines if s.startswith('N|')]
   if len(discrete)!=1 or len(numeric)!=1: raise ValueError(f'{label}: missing records')
   values=[float(line.split('|',2)[2]) for line in numeric if line.startswith('N|0|')]
   if len(values)!=1 or not all(map(math.isfinite,values)) or any(line.split('|',2)[2]!='NaN' for line in numeric if not line.startswith('N|0|')): raise ValueError(f'{label}: numeric record')
   records[(life,spin,rep)]=(discrete,values)
   match=re.search(r'wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)',(root/f'{label}.time').read_text())
   if not match: raise ValueError(f'{label}: timing')
   w,u,s=map(float,match.groups()); samples.append(dict(wall=w,user=u,system=s))
  timing[life][spin]={'samples':samples,**{f'{k}_median':statistics.median(v[k] for v in samples) for k in ('wall','user','system')}}
refd,refn=records[('held','default',1)]
print(json.dumps({'discrete_match':all(d==refd for d,n in records.values()),'root_energy_max_abs':max(abs(n[0]-refn[0]) for d,n in records.values()),'timings':timing},indent=2))
