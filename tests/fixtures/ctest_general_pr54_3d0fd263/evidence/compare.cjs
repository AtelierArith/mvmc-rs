'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const [rust, corrected, historical, output]=process.argv.slice(2);
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const values=p=>fs.readFileSync(p,'utf8').trim().split(/\s+/).map(Number);
const rows=[];
function compare(label,a,b,numeric){
 const x=numeric?values(a):fs.readFileSync(a,'utf8').trim().split(/\s+/),y=numeric?values(b):fs.readFileSync(b,'utf8').trim().split(/\s+/);
 if(x.length!==y.length||numeric&&(!x.every(Number.isFinite)||!y.every(Number.isFinite)))throw Error('shape/finite '+label);
 let differences=0,first=-1,max=0;
 for(let i=0;i<x.length;i++)if(x[i]!==y[i]){differences++;if(first<0)first=i;if(numeric)max=Math.max(max,Math.abs(x[i]-y[i]));}
 rows.push({label,left:a,left_sha256:sha(a),right:b,right_sha256:sha(b),length:x.length,differences,first,max_absolute_difference:numeric?max:null});
}
for(const file of ['sr_oo.txt','sr_ho.txt'])compare('actual Rust/3d0f Gen2 '+file,path.join(rust,file),path.join(corrected,'step-2',file),true);
for(const step of [1,2])for(const kind of ['matrix','rhs','increment'])compare('actual Rust/3d0f solve'+step+' '+kind,path.join(rust,`solve-${step}-${kind}.txt`),path.join(corrected,`step-${step}`,`direct-sr-${kind}.txt`),true);
for(const step of [1,2,3,20])for(const file of ['rng-state.txt','rng.txt','configs.txt'])compare('historical62b/3d0f discrete step'+step+' '+file,path.join(historical,'step-'+step,file),path.join(corrected,'step-'+step,file),false);
const result={producer78040_exit:1,scope:'read-only comparisons; historical62b discrete evidence is not a newly executed Rust state capture; no fixture adoption or bounds change',script_sha256:sha(__filename),rows};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
for(const r of rows)console.log(r.label+' length='+r.length+' differences='+r.differences+' max='+r.max_absolute_difference);
if(rows.some(r=>r.differences))process.exitCode=1;
