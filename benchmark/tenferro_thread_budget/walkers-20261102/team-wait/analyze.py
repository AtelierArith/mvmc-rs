import json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); result={}
for callers in (1,4):
 result[str(callers)]={}
 for gap in (0,60):
  result[str(callers)][str(gap)]={}
  for spin in ('default','0'):
   samples=[]
   for rep in (1,2,3):
    stem=f'c{callers}-g{gap}-p{spin}-r{rep}'
    vals=dict(re.findall(r'(\w+)=(\d+)',(root/f'{stem}.stdout').read_text()))
    if int(vals['calls']) != callers*5000 or int(vals['sum']) != callers*5000*2080: raise ValueError(f'{stem}: work mismatch')
    match=re.search(r'wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)',(root/f'{stem}.time').read_text())
    if not match: raise ValueError(f'{stem}: timing')
    w,u,s=map(float,match.groups())
    samples.append(dict(wall=w,user=u,system=s,join_sleeps=int(vals['join_sleeps']),barrier_sleeps=int(vals['barrier_sleeps']),member_parks=int(vals['member_parks'])))
   result[str(callers)][str(gap)][spin]={'samples':samples,**{f'{k}_median':statistics.median(v[k] for v in samples) for k in ('wall','user','system','join_sleeps','barrier_sleeps','member_parks')}}
print(json.dumps(result,indent=2))
