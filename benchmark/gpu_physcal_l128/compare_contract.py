import argparse,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('reference',type=Path);a=p.parse_args();root=a.root
reference=json.loads(gzip.decompress(a.reference.read_bytes()))
rows=[]
for kind in ['cpu','gpu']:
 for rep in range(4):
  path=root/f'contract-{kind}-final'/f'contract-{rep}.txt'; record={}; decisions=[]
  for line in path.read_text().splitlines():
   key,value=line.split('=',1)
   if key=='decision':
    accepted,draw=map(int,value.split(',')); decisions.append([bool(accepted),draw])
   else: record[key]=json.loads(value)
  record['decisions']=decisions
  assert record==reference, (kind,rep,[(k,record.get(k)==v) for k,v in reference.items()])
  rows.append({'kind':kind,'rep':rep,'decisions':len(decisions),'words':record['rng_words_consumed'],'all_exact_fields_match':True})
(root/'contract-comparison.json').write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps(rows,indent=2))
