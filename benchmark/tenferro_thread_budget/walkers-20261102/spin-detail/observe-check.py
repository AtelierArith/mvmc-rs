import math,sys
from pathlib import Path
for life in ('held','stage'):
 for spin in ('default','0'):
  timed=Path(sys.argv[1])/f'opt-{life}-p{spin}-r1.stdout'; obs=Path(f'{sys.argv[2]}{life}-{spin}')/'run.txt'
  def records(path):
   lines=path.read_text().splitlines(); discrete=[s for s in lines if s.startswith('D|')]; numeric=[float(s.split('|',2)[2]) for s in lines if s.startswith('N|0|')]
   if len(discrete)!=4 or len(numeric)!=1 or not all(map(math.isfinite,numeric)): raise ValueError(f'{path}: incomplete')
   return discrete,numeric
  a,an=records(timed); b,bn=records(obs)
  print(f'{life} {spin} discrete_match={a==b} root_energy_abs_diff={abs(an[0]-bn[0]):.17g}')
