// SOURCE-only rank-local worker comparison. No model, oracle or generated expectations.
import fs from 'node:fs';import assert from 'node:assert/strict';
const [one,two,four,output]=process.argv.slice(2);assert.ok(output);
const rows=[one,two,four].map(p=>JSON.parse(fs.readFileSync(p,'utf8')));
for(const [i,r] of rows.entries()){
 assert.equal(r.schema,'issue182-mpi-inner-physcal-v1');assert.equal(r.workers,[1,2,4][i]);
 assert.ok([2,4].includes(r.world));assert.ok([1,2].includes(r.groupWidth));
 assert.ok(Number.isInteger(r.rank)&&r.rank>=0&&r.rank<r.world);
 assert.equal(r.group,Math.floor(r.rank/r.groupWidth));
 assert.equal(r.groupRank,r.rank%r.groupWidth);assert.equal(r.seedOffset,r.group);
 assert.equal(r.threshold,32);assert.equal(r.frames,2);assert.equal(r.originalOne,2);assert.equal(r.originalDirect,36);
 assert.deepEqual(Object.keys(r).sort(),['schema','world','rank','groupWidth','group','groupRank','seedOffset','workers','threshold','frames','originalOne','originalDirect','dimensions','result','discrete','numeric','observation','scope'].sort());
 assert.deepEqual(Object.keys(r.dimensions).sort(),['sites','electronsPerSpin','projection','samples','factored'].sort());
 const d=r.dimensions;assert.equal(d.sites,6);assert.equal(d.electronsPerSpin,3);assert.equal(d.samples,100);assert.equal(d.factored,3);assert.equal(d.projection,2);
 assert.deepEqual(Object.keys(r.discrete).sort(),['initial','final','callbacks','saved','burn','scratch','counter'].sort());
 assert.deepEqual(r.result,{Ok:2});assert.equal(r.discrete.callbacks.length,2);
 for(const [i,c] of r.discrete.callbacks.entries())assert.deepEqual(c,{frame:i,status:0});
 for(const name of ['initial','final']){
  const b=r.discrete[name];assert.deepEqual(Object.keys(b).sort(),['cursor','future624','raw624','words']);assert.equal(b.raw624.length,624);assert.equal(b.future624.length,624);
  for(const n of [...b.raw624,...b.future624])assert.ok(Number.isInteger(n)&&n>=0&&n<=0xffffffff);
  assert.ok(Number.isInteger(b.cursor)&&b.cursor>=0&&b.cursor<=624);
  assert.match(b.words,/^(0|[1-9][0-9]*)$/);assert.ok(BigInt(b.words)<2n**128n);
 }
 for(const [k,values] of Object.entries(r.discrete.saved)){assert.ok(['idx','cfg','num','proj','spin'].includes(k));assert.ok(Array.isArray(values)&&values.every(Number.isSafeInteger));}
 assert.deepEqual(Object.keys(r.discrete.saved).sort(),['cfg','idx','num','proj','spin']);
 for(const plane of ['saved','burn','scratch']){
  const a=r.discrete[plane],n=plane==='saved'?d.samples:1;
  assert.deepEqual(Object.keys(a).sort(),['cfg','idx','num','proj','spin']);
  for(const v of Object.values(a))assert.ok(Array.isArray(v)&&v.every(Number.isSafeInteger));
  // state.rs594–624/633–675: burn.idx stores idx,cfg,num,proj,spin.
  // All captured words remain in the exact discrete comparison below.
  const active=2*d.electronsPerSpin,site2=2*d.sites;
  assert.equal(a.idx.length,plane==='burn'?active+2*site2+d.projection:n*active,'complete packed/active storage shape');
  assert.equal(a.cfg.length,n*site2);assert.equal(a.num.length,n*site2);assert.equal(a.proj.length,n*d.projection);assert.equal(a.spin.length,0);
  if(plane==='burn'){
   for(let i=0;i<active;i++)assert.ok(a.idx[i]>=0&&a.idx[i]<d.sites,'burn active index');
   for(let i=0;i<site2;i++){
    const cfg=a.idx[active+i],num=a.idx[active+site2+i];
    assert.ok(cfg>=-1&&cfg<d.electronsPerSpin,'burn packed cfg');
    assert.ok(num===0||num===1,'burn packed num');
    assert.equal(num,cfg<0?0:1,'burn packed cfg/num consistency');
   }
   for(let s=0;s<2;s++)for(let e=0;e<d.electronsPerSpin;e++){
    const site=a.idx[s*d.electronsPerSpin+e];
    assert.equal(a.idx[active+s*d.sites+site],e,'burn packed index/cfg order');
   }
   assert.ok(a.idx[active+2*site2]>=0&&a.idx[active+2*site2]<=3,'burn packed gutz');
   assert.ok(a.idx[active+2*site2+1]>=-15&&a.idx[active+2*site2+1]<=15,'burn packed jastrow');
  }else assert.ok(a.idx.every(i=>i>=0&&i<d.sites));
  // Separate captured burn buffers may be inactive: domain-check and retain,
  // never invent equality to canonical packed slots or discard their content.
  assert.ok(a.cfg.every(i=>i>=-1&&i<d.electronsPerSpin));assert.ok(a.num.every(i=>i===0||i===1));
  // Fixture has one Gutzwiller slot and one Jastrow slot, all off-diagonal
  // pairs mapped to it. make_proj_cnt adds n_up*n_down (<=3) and at most
  // C(6,2)=15 products of xi*xj in {-1,0,1}. Not a universal DH/RBM bound.
  for(let i=0;i<n;i++){assert.ok(a.proj[2*i]>=0&&a.proj[2*i]<=3);assert.ok(a.proj[2*i+1]>=-15&&a.proj[2*i+1]<=15);}
 }
 assert.deepEqual(Object.keys(r.numeric).sort(),['callbacks','energy','one','factored','direct'].sort());
 assert.equal(r.numeric.callbacks.length,2);assert.equal(r.numeric.energy.length,5);assert.equal(r.numeric.one.length,2);assert.equal(r.numeric.factored.length,3);assert.equal(r.numeric.direct.length,36);
 for(const values of Object.values(r.numeric))for(const z of values)assert.ok(Array.isArray(z)&&z.length===2&&z.every(v=>typeof v==='number'&&Number.isFinite(v)),'exact two finite complex components');
 assert.equal(r.discrete.counter.length,10);assert.ok(r.discrete.counter.every(v=>Number.isSafeInteger(v)&&v>=0));
 const o=r.observation;assert.deepEqual(Object.keys(o).sort(),['parallelCalls','serialCalls','parallelEntries','serialEntries','workerEntries','workerIds'].sort());for(const key of ['parallelCalls','serialCalls','parallelEntries','serialEntries','workerEntries'])assert.ok(Number.isSafeInteger(o[key])&&o[key]>=0);
 assert.ok(o.workerIds.every(id=>Number.isInteger(id)&&id>=0&&id<r.workers));assert.equal(new Set(o.workerIds).size,o.workerIds.length);
 if(r.workers===1){assert.equal(o.parallelEntries,0);assert.equal(o.workerEntries,0);assert.ok(o.serialEntries>=72);}else{assert.ok(o.parallelEntries>=72&&o.workerEntries>=72&&o.workerIds.length>0);}
}
const metadata=['world','rank','groupWidth','group','groupRank','seedOffset','threshold','frames','originalOne','originalDirect','dimensions'];
for(const r of rows.slice(1)){for(const k of metadata)assert.deepEqual(r[k],rows[0][k],'same world/group/rank '+k);}
const exact=JSON.stringify(rows[0].discrete);let divergence=null;
for(const r of rows.slice(1))if(JSON.stringify(r.discrete)!==exact){divergence={workers:r.workers,stage:'rank-local-discrete',diagnosis:'MissingEvidence for numerical interpretation'};break;}
let maximumError=0,compared=0;
const burnStorage=rows.map(r=>({workers:r.workers,completePacked32:r.discrete.burn.idx,activeIndices6:Array.from({length:6},(_,i)=>r.discrete.burn.idx[i]),layout:{active:[0,6],cfg:[6,18],num:[18,30],proj:[30,32],spin:[32,32]},separateCapturedInactiveBuffers:{cfg:r.discrete.burn.cfg,num:r.discrete.burn.num,proj:r.discrete.burn.proj,spin:r.discrete.burn.spin},padding:'no extra packed padding in this ordinary fixture; no captured content omitted'}));
function numbers(a,b,path){
 assert.equal(typeof a,typeof b,path);if(a===null){assert.equal(b,null,path);return;}
 if(Array.isArray(a)){assert.ok(Array.isArray(b));assert.equal(a.length,b.length,path);a.forEach((x,i)=>numbers(x,b[i],path+'/'+i));return;}
 if(typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path);for(const k of Object.keys(a))numbers(a[k],b[k],path+'/'+k);return;}
 assert.ok(typeof a==='number'&&Number.isFinite(a)&&Number.isFinite(b),path);
 const error=Math.abs(a-b);maximumError=Math.max(maximumError,error);compared++;
 // Inherited same-implementation output policy, physcal_callback_exact_contract.rs214–215;
 // not a new independent C/MPI budget. Failure requires first-cause review, never widening.
 assert.ok(error<=Math.max(1e-12,1e-12*Math.max(Math.abs(a),Math.abs(b))),path);
}
if(!divergence)for(const r of rows.slice(1))numbers(rows[0].numeric,r.numeric,'numeric');
fs.writeFileSync(output,JSON.stringify({scope:'same-world/group/rank workers1/2/4 only; initial/final raw boundaries not full per-proposal trace',metadata:Object.fromEntries(metadata.map(k=>[k,rows[0][k]])),burnStorage,discrete:divergence??'EXACT',numeric:divergence?[]:{compared,maximumError,policy:'existing same-implementation1e-12 absolute/relative'},observations:rows.map(r=>({workers:r.workers,...r.observation}))},null,2)+'\n',{flag:'wx'});
if(divergence)process.exitCode=1;
