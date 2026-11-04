"""Summarize an Nsight Systems 2022 SQLite export; no GPU/runtime required."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('database',type=Path);p.add_argument('output',type=Path);p.add_argument('--scope',default='One instrumented CLI run.');a=p.parse_args()
db=sqlite3.connect(f'file:{a.database.resolve()}?mode=ro',uri=True)
strings=dict(db.execute('select id,value from StringIds'))
kernels=list(db.execute('select start,end,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
assert kernels,'no captured kernels'
def union(intervals):
 result=0; end=None
 for left,right in sorted(intervals):
  assert right>=left
  result+=max(0,right-max(left,end if end is not None else left));end=max(right,end if end is not None else right)
 return result
by_name={}
for start,end,name,x,y,z in kernels:
 row=by_name.setdefault(strings[name],{'calls':0,'device_ns_sum':0,'grid_sizes':set()});row['calls']+=1;row['device_ns_sum']+=end-start;row['grid_sizes'].add((x,y,z))
for row in by_name.values():row['grid_sizes']=sorted(row['grid_sizes'])
api=[]
for name,count,ns in db.execute('select nameId,count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId order by sum(end-start) desc'):
 api.append({'name':strings[name],'calls':count,'host_ns_sum':ns})
copies=[]
for name,label,count,bytes_,ns in db.execute('select e.name,e.label,count(*),sum(m.bytes),sum(m.end-m.start) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id group by m.copyKind'):
 copies.append({'kind':name,'label':label,'calls':count,'bytes':bytes_,'device_ns_sum':ns})
launches=sum(row['calls'] for row in api if row['name']=='cuLaunchKernel')
runtime_launches=sum(row['calls'] for row in api if row['name'].startswith('cudaLaunchKernel_v'))
launch_ids={correlation for correlation,name in db.execute('select correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME') if strings[name]=='cuLaunchKernel' or strings[name].startswith('cudaLaunchKernel_v')}
kernel_ids={row[0] for row in db.execute('select correlationId from CUPTI_ACTIVITY_KIND_KERNEL')}
assert launches+runtime_launches==len(kernels) and launch_ids==kernel_ids,(launches,runtime_launches,len(kernels))
active_window=max(k[1] for k in kernels)-kernels[0][0];busy=union((k[0],k[1]) for k in kernels)
result={'scope':a.scope+' Device busy fraction is the union of kernel intervals / first-to-last kernel span; it is not SM occupancy, and includes no other process. API/device times overlap; do not sum them.','kernel_count':len(kernels),'matching_driver_launch_count':launches,'matching_runtime_launch_count':runtime_launches,'kernel_window_ns':active_window,'kernel_busy_union_ns':busy,'kernel_busy_fraction':busy/active_window,'kernels':by_name,'api':api,'copies':copies}
result['gpu']=[dict(zip(['id','name','uuid','sm_count','compute_major','compute_minor'],row)) for row in db.execute('select id,name,uuid,smCount,computeMajor,computeMinor from TARGET_INFO_GPU')]
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['kernels','api']},indent=2))
