import ast,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); result={}
for case in ('l32-opt','l32-phys','l64-opt'):
 result[case]={}
 for workers in (1,4):
  text=(root/f'{case}-{workers}.stderr').read_text(); sites=[]; aggregates={}
  for line in text.splitlines():
   m=re.fullmatch(r'\s+(\S+:\d+) (hot|cold) (\[.*\])',line)
   if m: sites.append({'site':m.group(1),'hot':m.group(2)=='hot','counts':ast.literal_eval(m.group(3))})
   m=re.fullmatch(r'\s+(hot|cold): (\[.*\])',line)
   if m: aggregates[m.group(1)]=ast.literal_eval(m.group(2))
  if not sites or set(aggregates)!={'hot','cold'} or any(len(s['counts'])!=7 for s in sites): raise ValueError(f'{case} {workers}: missing/malformed')
  for kind in ('hot','cold'):
   total=[sum(s['counts'][i] for s in sites if s['hot']==(kind=='hot')) for i in range(7)]
   if total!=aggregates[kind]: raise ValueError(f'{case} {workers}: totals mismatch {kind}')
  result[case][str(workers)]={'aggregate':aggregates,'sites':sites}
print(json.dumps(result,indent=2))
