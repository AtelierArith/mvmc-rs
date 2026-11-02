import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root;cpu=r/'initial-cpu/rep0';jul=r/'julia-initial';summary={}
# C/Rust cisajscktalt stores direct GreenTwo terms; Julia returns these in green_two.
for filename,julia in [('zvo_cisajs_001.dat','one'),('zvo_cisajscktalt_001.dat','two')]:
 rows=[line.split() for line in (cpu/filename).read_text().splitlines() if line.strip()]
 expected=[float(row[-2]) for row in rows];imag=[float(row[-1]) for row in rows]
 actual=[float(line) for line in (jul/f'common-configuration-{julia}.txt').read_text().splitlines()]
 assert len(actual)==len(expected)
 diffs=[]
 for x,y,im in zip(expected,actual,imag):
  assert math.isfinite(x) and math.isfinite(y) and abs(im)<=1e-11
  diff=abs(x-y); assert diff<=1e-11+1e-11*max(abs(x),abs(y)),(filename,x,y,diff)
  diffs.append(diff)
 summary[julia]={'values':len(actual),'max_abs':max(diffs,default=0)}
assert not (jul/'common-configuration-two_ex.txt').read_text().strip(), 'input has no factored terms'
e=float((cpu/'zvo_out_001.dat').read_text().split()[0]);j=float((jul/'common-configuration-energy.txt').read_text());assert abs(e-j)<=1e-11+1e-11*max(abs(e),abs(j))
summary['energy']={'rust':e,'julia':j,'abs_difference':abs(e-j)}
record=json.loads((r/'initial-cpu-record.json').read_text()); words=list(map(int,(jul/'first-initial-words.txt').read_text().split()));cfg=list(map(int,(jul/'first-walker.txt').read_text().split()))
summary['initial_divergence']={'SFMT_first_word':record['first_initial_words'][0],'XORWOW_first_word':words[0],'cpu_first_site':record['ele_idx'][0],'julia_first_site':cfg[0],'first_differing_configuration_index':next(i for i,(a,b) in enumerate(zip(record['ele_idx'],cfg)) if a!=b),'scope':'before burn-in/proposals; CPU single-chain SFMT vs first Julia GPU XORWOW walker'}
(r/'julia-numerical-comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
