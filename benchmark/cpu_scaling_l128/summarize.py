from pathlib import Path
import collections,json,re,statistics,sys
base=Path(sys.argv[1]); rows=json.loads((base/'timings.json').read_text())
metadata=json.loads((base/'metadata.json').read_text())
expected_runs=32 if metadata['args'].get('only_rust')=='True' else 64
groups=collections.defaultdict(list)
for row in rows:
    label=f"{row['mode']}-{row['kind']}-t{row['threads']}-r{row['rep']}"
    text=(base/label/'output/zvo_CalcTimer.dat').read_text()
    timer={int(i):float(v) for i,v in re.findall(r'\[(\d+)\]\s+([0-9.eE+-]+)',text)}
    row['timer']=timer
    row['outside_internal_all']=row['wall']-timer[0]
    if row['rep']>0: groups[(row['mode'],row['kind'],row['threads'])].append(row)
summary=[]
for (mode,kind,n),values in sorted(groups.items()):
    result={'mode':mode,'kind':kind,'threads':n,'measured_repetitions':len(values),'complete':len(values)==3}
    for key in ['wall','user','sys','cpu_percent','outside_internal_all']:
        result[key]={'median':statistics.median(r[key] for r in values),'min':min(r[key] for r in values),'max':max(r[key] for r in values)}
    result['inclusive_phase_medians']={i:statistics.median(r['timer'][i] for r in values if i in r['timer']) for i in [0,3,4,5,34,40,41,63,71] if any(i in r['timer'] for r in values)}
    summary.append(result)
(base/'provisional-summary.json').write_text(json.dumps({'note':'Only complete groups have all three measured repetitions. Timers are inclusive; never sum nested phases. Outside All includes all untimed work, not pure startup.','completed_runs':len(rows),'expected_runs':expected_runs,'groups':summary},indent=2)+'\n')
print(json.dumps({'completed_runs':len(rows),'expected_runs':expected_runs,'complete_groups':sum(x['complete'] for x in summary),'groups':[(x['mode'],x['kind'],x['threads'],x['measured_repetitions']) for x in summary]}))
