"""Summarize complete measured cohorts; diagnostic and warmup runs are excluded."""
import argparse,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();data={'cli':{},'api':{}}
for name in ['gpu-before','cache-disabled','cache-cold','cache-hit','final-disabled','final-cold','final-hit']:
 rows=json.loads((a.root/name/'timings.json').read_text());times=[r['seconds'] for r in rows if r['rep'] in [1,2,3]];assert len(times)==3
 phases={}
 for rep in [1,2,3]:
  text=(a.root/name/f'rep{rep}/output/zvo_CalcTimer.dat').read_text()
  for id_,seconds in re.findall(r'\[(\d+)\]\s+([\d.]+)',text):phases.setdefault(id_,[]).append(float(seconds))
 data['cli'][name]={'median':statistics.median(times),'min':min(times),'max':max(times),'inclusive_phase_medians':{k:statistics.median(v) for k,v in phases.items()},'outside_internal_all_median':statistics.median(next(r['seconds'] for r in rows if r['rep']==rep)-all_time for rep,all_time in zip([1,2,3],phases['0']))}
for name in ['final-api-cpu-1','final-api-cpu-4','final-api-cpu-8','final-api-cpu-16','final-api-gpu','final-api-gpu-files']:
 rows=[float(t) for rep,t in re.findall(r'BENCH rep=(\d+) seconds=([\d.]+)',(a.root/f'{name}.log').read_text()) if int(rep) in [1,2,3]];assert len(rows)==3
 data['api'][name]={'median':statistics.median(rows),'min':min(rows),'max':max(rows)}
a.output.write_text(json.dumps(data,indent=2)+'\n')
