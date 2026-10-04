'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const [corrected,output]=process.argv.slice(2);
const inputs={3:'/tmp/mvmc-ctest-prefix-2706311-1791027148323380957',20:'/tmp/mvmc-ctest-prefix-2706794-1791027169285011463'};
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const rows=[];
for(const [steps,rust] of Object.entries(inputs)){
 const reference=path.join(corrected,'step-'+steps);
 for(const name of ['rng-state.txt','rng.txt','configs.txt','parameters.txt','energy.txt','sr_oo.txt','sr_ho.txt','zvo_out.dat','zvo_var.dat']){
  const a=path.join(rust,name),b=path.join(reference,name==='zvo_var.dat'?'zvo_c_slots_var.dat':name);
  const discrete=name.startsWith('rng')||name==='configs.txt';
  let x=fs.readFileSync(a,'utf8').trim().split(/\s+/),y=fs.readFileSync(b,'utf8').trim().split(/\s+/);
  if(x.length!==y.length)throw Error('shape '+steps+' '+name);
  if(!discrete){x=x.map(Number);y=y.map(Number);if(!x.every(Number.isFinite)||!y.every(Number.isFinite))throw Error('nonfinite '+name);}
  let differences=0,first=-1,max=0;
  for(let i=0;i<x.length;i++)if(x[i]!==y[i]){differences++;if(first<0)first=i;if(!discrete)max=Math.max(max,Math.abs(x[i]-y[i]));}
  rows.push({steps:Number(steps),name,left:a,left_sha256:sha(a),right:b,right_sha256:sha(b),length:x.length,discrete,differences,first,max_absolute_difference:discrete?null:max});
 }
 const a=path.join(rust,'solve-'+steps+'-'),b=path.join(reference,'direct-sr-');
 for(const kind of ['matrix','rhs','increment']){
  const ap=a+kind+'.txt',bp=b+kind+'.txt',x=fs.readFileSync(ap,'utf8').trim().split(/\s+/).map(Number),y=fs.readFileSync(bp,'utf8').trim().split(/\s+/).map(Number);
  if(x.length!==y.length||!x.every(Number.isFinite)||!y.every(Number.isFinite))throw Error('solve shape/finite');
  let differences=0,max=0;for(let i=0;i<x.length;i++)if(x[i]!==y[i]){differences++;max=Math.max(max,Math.abs(x[i]-y[i]));}
  rows.push({steps:Number(steps),name:'original-solve-'+kind,left:ap,left_sha256:sha(ap),right:bp,right_sha256:sha(bp),length:x.length,differences,max_absolute_difference:max});
 }
}
fs.writeFileSync(output,JSON.stringify({producer78040_exit:1,scope:'actual newly executed Rust3/20 vs independent corrected Julia3d0f captures; historical gates failed old62b numerical expectations; no fixture adoption, no bounds change, Cwindow not compared here',script_sha256:sha(__filename),rows},null,2)+'\n',{flag:'wx'});
for(const r of rows)console.log(`step${r.steps} ${r.name} length=${r.length} differences=${r.differences} max=${r.max_absolute_difference}`);
if(rows.some(r=>r.differences))process.exitCode=1;
