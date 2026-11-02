import math,sys
from pathlib import Path
for eng in (1,4):
 timed=Path(sys.argv[1])/f'opt-w4-e{eng}-r1.stdout'
 obs=Path(sys.argv[2]+str(eng))/'run.txt'
 def records(path):
  lines=path.read_text().splitlines()
  discrete=[line for line in lines if line.startswith('D|')]
  numeric=[float(line.split('|',2)[2]) for line in lines if line.startswith('N|0|')]
  if len(discrete)!=4 or len(numeric)!=1 or not all(map(math.isfinite,numeric)): raise ValueError(f'{path}: incomplete')
  return discrete,numeric
 a,an=records(timed); b,bn=records(obs)
 print(f'engine={eng} discrete_match={a==b} root_energy_abs_diff={abs(an[0]-bn[0]):.17g}')
