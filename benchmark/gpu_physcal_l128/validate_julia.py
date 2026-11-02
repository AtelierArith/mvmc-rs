"""Validate finite/repeated Julia GPU results without comparing different trajectories."""
import argparse,json,math,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();worst=0
for name,count in [('one',256),('two',768),('two_ex',0)]:
 baseline=None
 for rep in range(4):
  values=list(map(float,(a.root/'julia-final'/f'rep{rep}-{name}.txt').read_text().split()));assert len(values)==count and all(map(math.isfinite,values))
  if baseline is None:baseline=values
  for x,y in zip(baseline,values):
   d=abs(x-y);assert d<=1e-11+1e-11*max(abs(x),abs(y));worst=max(worst,d)
rows=[(int(rep),float(sec),float(energy)) for rep,sec,energy in re.findall(r'BENCH rep=(\d+) seconds=([\d.]+) energy=([\d.eE+-]+)',(a.root/'julia-final.log').read_text())];assert len(rows)==4
for rep,sec,energy in rows:
 assert math.isfinite(energy) and abs(energy-rows[0][2])<=1e-11+1e-11*max(abs(energy),abs(rows[0][2]))
result={'runs':4,'one_terms':256,'two_terms':768,'factored_terms':0,'max_repeat_green_abs_difference':worst,'scope':'Same Julia implementation/seed repeatability; not a comparison of different RNG trajectories.'}
(a.root/'julia-repeatability.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
