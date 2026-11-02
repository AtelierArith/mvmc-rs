"""Summarize sampled thread activity, without treating snapshots as exact CPU totals."""
from pathlib import Path
import collections,json,sys
root=Path(sys.argv[1]); records=[]
for path in sorted(root.glob('*-profile/summary.json')):
    summary=json.loads(path.read_text()); threads={}; hist=collections.Counter(); previous={}
    for line in path.with_name('threads.jsonl').read_text().splitlines():
        sample=json.loads(line); active=0
        for t in sample['threads']:
            key=(t['pid'],t['tid'],t['start_ticks']); ticks=t['user_ticks']+t['sys_ticks']
            entry=threads.setdefault(key,{'pid':t['pid'],'tid':t['tid'],'comm':t['comm'],'max_sampled_ticks':0,'cpus':set(),'states':collections.Counter(),'wchan':collections.Counter()})
            entry['max_sampled_ticks']=max(entry['max_sampled_ticks'],ticks)
            entry['cpus'].add(t['processor']); entry['states'][t['state']]+=1; entry['wchan'][t['wchan']]+=1
            if key in previous and ticks>previous[key]: active+=1
            previous[key]=ticks
        hist[active]+=1
    for t in threads.values():
        t['cpus']=sorted(t['cpus']); t['states']=dict(t['states']); t['wchan']=dict(t['wchan'])
        t['sampled_cpu_seconds']=t['max_sampled_ticks']/summary['clock_ticks_per_second']
    records.append({'case':path.parent.name,'resource':summary,'sampled_threads':sorted(threads.values(),key=lambda t:-t['max_sampled_ticks']),'active_thread_sample_histogram':dict(sorted(hist.items()))})
result={'scope':'Diagnostic runs only. A thread is active when its CPU ticks increased since its preceding observation. Samples miss activity between observations, short-lived threads and the final polling interval; max sampled ticks are not exact whole-run totals. Use resource user/sys for full process-tree CPU accounting.','complete_cases':len(records),'cases':records}
(root/'thread-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('Complete diagnostic cases:',len(records))
