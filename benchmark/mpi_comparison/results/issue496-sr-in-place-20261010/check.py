from pathlib import Path
import json, re, statistics, math
root=Path('/home/vscode/.cache/mvmc/issue496-sr-in-place-confirm')
r={'policy':'L64, MPI4 x threads4, BLAS1; warmup1 repetitions3, sequential modes; same binary, diagnostic flag only'}
for mode in ['baseline','inplace']:
    times=[float(x) for x in re.findall(r'^BENCH \d+ ([0-9.]+)',(root/f'{mode}.log').read_text(),re.M)]
    assert len(times)==3
    r[mode]={'seconds':times,'median':statistics.median(times)}
errors=[]
for rep in [1,2,3]:
    a=[[float(x) for x in line.split()] for line in (root/f'baseline/run-{rep}/zvo_out.dat').read_text().splitlines()]
    b=[[float(x) for x in line.split()] for line in (root/f'inplace/run-{rep}/zvo_out.dat').read_text().splitlines()]
    assert len(a)==len(b)==300
    assert all(len(row)==6 and all(math.isfinite(x) for x in row) for row in a+b)
    errors.append(max(abs(x-y) for aa,bb in zip(a,b) for x,y in zip(aa,bb)))
r['max_abs_output_by_rep']=errors
(root/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
