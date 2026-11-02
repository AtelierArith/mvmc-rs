import json
import os
from pathlib import Path
import platform
import subprocess
import time

out = Path('/tmp/mvmc-583-final-policy-observer.jsonl')
with out.open('x') as stream:
    stream.write(json.dumps(dict(event='host', uname=platform.uname()._asdict(),
                                 affinity=sorted(os.sched_getaffinity(0)),
                                 cpuinfo=Path('/proc/cpuinfo').read_text())) + '\n')
    stream.flush()
    child = subprocess.Popen(['bash', '/tmp/mvmc-583-final-policy-plan/run.sh'])
    while True:
        stream.write(json.dumps(dict(event='sample', time=time.time(),
                                     monotonic=time.monotonic(),
                                     loadavg=Path('/proc/loadavg').read_text(),
                                     stat=Path('/proc/stat').read_text(),
                                     pressure=Path('/proc/pressure/cpu').read_text())) + '\n')
        stream.flush()
        status = child.poll()
        if status is not None:
            stream.write(json.dumps(dict(event='exit', status=status,
                                         time=time.time())) + '\n')
            break
        time.sleep(1)
    raise SystemExit(status)
