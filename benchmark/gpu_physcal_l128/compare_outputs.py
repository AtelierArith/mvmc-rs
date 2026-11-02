"""Check saved CLI/API outputs against an independent captured CPU cohort."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('reference',type=Path);a=p.parse_args();rows=[];folders=[]
for cohort in ['gpu-before','cache-disabled','cache-cold','cache-hit','final-disabled','final-cold','final-hit']:
 folders.extend((a.root/cohort).glob('rep*/output'))
folders.extend((a.root/'final-api-gpu-files').glob('rep*'))
if (a.root/'nsight-probe/output').exists():folders.append(a.root/'nsight-probe/output')
assert folders
for folder in sorted(folders):
 count=0;headers=0;worst=0
 for name,width in [('zvo_out_001.dat',0),('zvo_cisajs_001.dat',4),('zvo_cisajscktalt_001.dat',8)]:
  expected=[line.split() for line in (a.reference/name).read_text().splitlines() if line.strip()]
  actual=[line.split() for line in (folder/name).read_text().splitlines() if line.strip()]
  assert len(expected)==len(actual),(folder,name)
  for xrow,yrow in zip(expected,actual):
   assert len(xrow)==len(yrow),(folder,name)
   assert list(map(int,xrow[:width]))==list(map(int,yrow[:width])),(folder,name,'indices')
   headers+=width
   for x,y in zip(xrow[width:],yrow[width:]):
    x=float(x);y=float(y);assert math.isfinite(x) and math.isfinite(y)
    diff=abs(x-y);assert diff<=1e-11+1e-11*max(abs(x),abs(y)),(folder,name,x,y)
    count+=1;worst=max(worst,diff)
 rows.append({'run':str(folder.relative_to(a.root)),'computed_numeric_values':count,'exact_integer_indices':headers,'max_absolute_difference':worst})
(a.root/'output-comparison.json').write_text(json.dumps({'reference':str(a.reference),'finite_bound':'1e-11 + 1e-11 max(abs(x),abs(y))','rows':rows},indent=2)+'\n');print(len(rows),'output sets verified')
