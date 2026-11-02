import json,os,platform,subprocess,time
from pathlib import Path
out=Path('/tmp/mvmc-581-scaling-hostload-observer.jsonl')
with out.open('x') as f:
 f.write(json.dumps(dict(event='host',uname=platform.uname()._asdict(),affinity=sorted(os.sched_getaffinity(0)),cpuinfo=Path('/proc/cpuinfo').read_text(),version=Path('/proc/version').read_text()))+'\n');f.flush()
 child=subprocess.Popen(['bash','/tmp/mvmc-581-scaling-hostload-run.sh'])
 while True:
  f.write(json.dumps(dict(event='sample',time=time.time(),monotonic=time.monotonic(),loadavg=Path('/proc/loadavg').read_text(),stat=Path('/proc/stat').read_text(),pressure=Path('/proc/pressure/cpu').read_text()))+'\n');f.flush()
  status=child.poll()
  if status is not None:
   f.write(json.dumps(dict(event='exit',status=status,time=time.time()))+'\n');break
  time.sleep(1)
 raise SystemExit(status)
