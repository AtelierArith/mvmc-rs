import json,math,re,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]); baseline=Path(sys.argv[2]);summary=[]
def discrete(lines):
 return sorted(s if s.startswith('D|sha256|') else 'D|sha256|'+hashlib.sha256((s+'\n').encode()).hexdigest() for s in lines if s.startswith('D|'))
for mode in ('opt','phys'):
 for walkers in (1,4):
  ref=(baseline/f'{mode}-w{walkers}-bunset-r0.stdout').read_text().splitlines()
  expected=discrete(ref)
  numeric_ref={s.split('|')[1]:tuple(map(float,s.split('|')[2].split())) for s in ref if s.startswith('N|')}
  for spin in ('default','0'):
   observed=[];maximum=0.0
   for rep in (1,2,3):
    stem=f'{mode}-w{walkers}-s{spin}-r{rep}'
    lines=(root/f'{stem}.stdout').read_text().splitlines()
    assert discrete(lines)==expected,stem
    numeric={s.split('|')[1]:tuple(map(float,s.split('|')[2].split())) for s in lines if s.startswith('N|')}
    assert numeric.keys()==numeric_ref.keys(),stem
    for k,values in numeric.items():
     assert len(values)==len(numeric_ref[k]),stem
     for a,b in zip(values,numeric_ref[k]):
      if mode=='opt' and k!='0':assert math.isnan(a) and math.isnan(b),stem
      else:
       assert math.isfinite(a) and math.isfinite(b),stem
       maximum=max(maximum,abs(a-b))
    text=(root/f'{stem}.stderr').read_text()
    match=re.search(r'engine-wait-counters=(None|Some\(\[(\d+), (\d+), (\d+)\]\))',text)
    assert match,stem
    observed.append(None if match[1]=='None' else [int(match[i]) for i in (2,3,4)])
   summary.append(dict(mode=mode,walkers=walkers,spin=spin,wait_counters=observed,discrete_matches_baseline=True,max_absolute_numeric_difference=maximum))
print(json.dumps(summary,indent=2))
