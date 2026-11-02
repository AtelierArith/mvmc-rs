"""Validate the six recorded live-MPI placement diagnostic cases."""
from pathlib import Path
import re
import sys
root=Path(sys.argv[1])
for placement in ('shared','disjoint','partial'):
 for count in (1,2):
  p=root/f'{placement}-{count}.log';text=p.read_text()
  masks={int(r):set(map(int,c.split(','))) for r,c in re.findall(r'world_rank=(\d+) affinity=ok:([0-9,]+)',text)}
  planned={int(r):set(map(int,c.split(','))) for r,c in re.findall(r'world_rank=(\d+) planned_cpus=([0-9,]+)',text)}
  assert set(masks)==set(range(4)),p
  rejected=placement in ('shared','partial') and count==2
  assert ('plan_error=' in text)==rejected,p
  if rejected:
   assert not planned,p
  else:
   assert set(planned)==set(masks),p
   used=set()
   for rank,cpus in planned.items():
    assert len(cpus)==count and cpus<=masks[rank] and used.isdisjoint(cpus),p
    used.update(cpus)
  print(p.name,'verified')
