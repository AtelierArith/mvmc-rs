"""Diagnostic process-tree sampling; launch through uv, separately from timings."""
import argparse,json,os,resource,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--interval',type=float,default=0.2)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
if a.command and a.command[0]=='--': a.command=a.command[1:]
if not a.command: p.error('missing command')
a.out.mkdir()
start=time.monotonic(); before=resource.getrusage(resource.RUSAGE_CHILDREN)
def tree(root):
    result=set(); pending=[root]
    while pending:
        pid=pending.pop()
        if pid in result: continue
        result.add(pid)
        try: tasks=list(Path(f'/proc/{pid}/task').iterdir())
        except FileNotFoundError: continue
        for task in tasks:
            try: pending.extend(map(int,(task/'children').read_text().split()))
            except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    return result
def sample(root):
    rows=[]
    for pid in tree(root):
        try: tasks=list(Path(f'/proc/{pid}/task').iterdir())
        except FileNotFoundError: continue
        for task in tasks:
            try:
                s=(task/'stat').read_text(); comm=s[s.index('(')+1:s.rindex(')')]; fields=s[s.rindex(')')+2:].split()
                rows.append({'pid':pid,'tid':int(task.name),'comm':comm,'state':fields[0],'user_ticks':int(fields[11]),'sys_ticks':int(fields[12]),'start_ticks':int(fields[19]),'processor':int(fields[36]),'wchan':(task/'wchan').read_text().strip()})
            except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    return rows
with (a.out/'stdout').open('w') as stdout,(a.out/'stderr').open('w') as stderr,(a.out/'threads.jsonl').open('w') as samples:
    child=subprocess.Popen(a.command,stdout=stdout,stderr=stderr)
    while True:
        samples.write(json.dumps({'elapsed':time.monotonic()-start,'threads':sample(child.pid)})+'\n'); samples.flush()
        try:
            child.wait(timeout=a.interval)
            break
        except subprocess.TimeoutExpired:
            pass
    status=child.wait()
after=resource.getrusage(resource.RUSAGE_CHILDREN)
(a.out/'summary.json').write_text(json.dumps({'command':a.command,'status':status,'wall':time.monotonic()-start,'user':after.ru_utime-before.ru_utime,'sys':after.ru_stime-before.ru_stime,'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'),'sampling_interval':a.interval,'scope':'Diagnostic only; per-thread samples are incomplete between observations and include launcher/helper descendants. Rusage is whole-process-tree accounting, not a sum of samples.'},indent=2)+'\n')
raise SystemExit(status)
