'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=__dirname,sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const captured='/proc/1038648/root/tmp/mvmc-general-pr54-3d0fd263.aEqff0-direct-v2/general_rbm_cmp';
const rust={1:'/tmp/mvmc-ctest-prefix-2605608-1791025868246946114',2:'/tmp/mvmc-ctest-prefix-2438630-1791024155673939246',3:'/tmp/mvmc-ctest-prefix-2706311-1791027148323380957',20:'/tmp/mvmc-ctest-prefix-2706794-1791027169285011463'};
const records=[];
for(const [steps,dir] of Object.entries(rust)){
 const reference=path.join(root,'window-v2-'+steps);
 for(const file of fs.readdirSync(reference).sort()){
  const a=path.join(dir,file.replace('zqp_c_window','zqp')),b=path.join(reference,file);
  const xt=fs.readFileSync(a,'utf8').trim().split(/\s+/),yt=fs.readFileSync(b,'utf8').trim().split(/\s+/);
  const x=xt.map(Number),y=yt.map(Number);
  if(x.length!==y.length)throw Error('shape '+steps+' '+file);
  let differences=0,max=0,nonfinite=0,headers=0;
  for(let i=0;i<x.length;i++){
   if(Number.isNaN(x[i])&&!/^[+-]?nan$/i.test(xt[i])||Number.isNaN(y[i])&&!/^[+-]?nan$/i.test(yt[i])){headers++;if(xt[i]!==yt[i])differences++;}
   else if(!Number.isFinite(x[i])||!Number.isFinite(y[i])){nonfinite++;if(!(Number.isNaN(x[i])&&Number.isNaN(y[i]))&&x[i]!==y[i])differences++;}
   else if(x[i]!==y[i]){differences++;max=Math.max(max,Math.abs(x[i]-y[i]));}
  }
  records.push({steps:Number(steps),file,left:a,left_sha256:sha(a),right:b,right_sha256:sha(b),length:x.length,differences,max_absolute_difference:max,matching_nonfinite_components:nonfinite,header_tokens:headers});
 }
}
const authority='/home/terasaki/work/atelierarith/mvmc-rs';
const provenance={scope:'Mixed reference: independent corrected Julia3d0f Etot/Etot2/postSR history aggregated by verbatim C avevar bodies. Not full C sampler/SR validation. Failed Julia producer78040 remains exit1; standalone Caggregation successful. No tolerance changes.',compiler:'Ubuntu GCC13.3.0-6ubuntu2~24.04.1',compiler_options:'-std=c11 -O0 -ffp-contract=off -lm',first_strict_compile_exit:1,first_strict_compile_reason:'upstream sprintf format-overflow warnings treated as errors; no numerical source edited',successful_compile_exit:0,all_four_probe_exit:0,script_sha256:sha(__filename),sources:{},histories:{},records};
for(const file of ['c_toolbox/ctest_opt_window.c','c_toolbox/ctest_opt_window_upstream.inc','extern/mVMC-1.3.0/src/mVMC/avevar.c'])provenance.sources[path.join(authority,file)]=sha(path.join(authority,file));
provenance.sources[path.join(root,'ctest_opt_window')]=sha(path.join(root,'ctest_opt_window'));
for(const steps of [1,2,3,20]){const file=path.join(captured,'step-'+steps,'c-window-input.txt');provenance.histories[file]=sha(file);}
fs.writeFileSync(path.join(root,'window-comparison-v2.json'),JSON.stringify(provenance,null,2)+'\n',{flag:'wx'});
for(const r of records)console.log(`step${r.steps} ${r.file} length=${r.length} differences=${r.differences} max=${r.max_absolute_difference} matchingNonfinite=${r.matching_nonfinite_components}`);
if(records.some(r=>r.differences))process.exitCode=1;
