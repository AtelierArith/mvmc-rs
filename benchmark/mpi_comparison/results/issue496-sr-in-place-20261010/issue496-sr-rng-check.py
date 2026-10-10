from pathlib import Path
import json, math
root=Path('/home/vscode/.cache/mvmc/issue496-sr-in-place-rng')
rows=[]
exact=['rank','world_size','steps','rng_words','rng_index','rng_words_consumed','ele_idx','ele_cfg','ele_num','ele_spn','ele_proj_cnt','counter']
for size in [32,64]:
 for steps in [20,300]:
  for rank in range(4):
   a=json.loads((root/f'L{size}-{steps}-baseline/rng-rank-{rank}.json').read_text())
   b=json.loads((root/f'L{size}-{steps}-inplace/rng-rank-{rank}.json').read_text())
   assert len(a['rng_words'])==len(b['rng_words'])==624
   for key in exact: assert a[key]==b[key],(size,steps,rank,key)
   rows.append({'sites':size,'steps':steps,'rank':rank,'exact_state_config_count':True,'words_consumed':a['rng_words_consumed'],'energy_delta':abs(a['final_energy_per_site']-b['final_energy_per_site']) if rank==0 else None})
  a=[[float(x) for x in line.split()] for line in (root/f'L{size}-{steps}-baseline/production/zvo_out.dat').read_text().splitlines()]
  b=[[float(x) for x in line.split()] for line in (root/f'L{size}-{steps}-inplace/production/zvo_out.dat').read_text().splitlines()]
  assert len(a)==len(b)==steps
  assert all(len(row)==6 and all(math.isfinite(x) for x in row) for row in a+b)
  print(size,steps,'output_max_abs',max(abs(x-y) for aa,bb in zip(a,b) for x,y in zip(aa,bb)))
(root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print('all 16 rank cases exact RNG and configurations')
