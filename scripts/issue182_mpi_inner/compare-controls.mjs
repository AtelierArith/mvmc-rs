// SOURCE only, not executed. Synthetic transport controls are not a physics oracle.
import fs from 'node:fs';import cp from 'node:child_process';import assert from 'node:assert/strict';import path from 'node:path';
const root=process.argv[2];assert.ok(path.isAbsolute(root)&&!fs.existsSync(root));fs.mkdirSync(root);
const zeros=n=>Array(n).fill(0),pairs=n=>Array.from({length:n},()=>[0,0]);
const boundary=()=>({raw624:zeros(624),cursor:0,words:'0',future624:zeros(624)});
const plane=n=>({idx:zeros(n*6),cfg:Array(n*12).fill(-1),num:zeros(n*12),proj:zeros(n*2),spin:[]});
const base={schema:'issue182-mpi-inner-physcal-v1',world:2,rank:0,groupWidth:1,group:0,groupRank:0,seedOffset:0,workers:1,threshold:32,frames:2,originalOne:2,originalDirect:36,dimensions:{sites:6,electronsPerSpin:3,projection:2,samples:100,factored:3},result:{Ok:2},scope:'synthetic transport only',discrete:{initial:boundary(),final:boundary(),callbacks:[{frame:0,status:0},{frame:1,status:0}],saved:plane(100),burn:plane(1),scratch:plane(1),counter:zeros(10)},numeric:{callbacks:pairs(2),energy:pairs(5),one:pairs(2),factored:pairs(3),direct:pairs(36)},observation:{parallelCalls:0,serialCalls:1,parallelEntries:0,serialEntries:72,workerEntries:0,workerIds:[]}};
// Independent literal layout: six electron sites; twelve inverse labels;
// twelve binary occupations; two bounded projection counters. No Rust output.
base.discrete.burn.idx=[0,1,2,0,1,2, 0,1,2,-1,-1,-1,0,1,2,-1,-1,-1, 1,1,1,0,0,0,1,1,1,0,0,0, 3,-3];
const variants=[1,2,4].map(w=>{const r=structuredClone(base);r.workers=w;if(w>1)r.observation={parallelCalls:1,serialCalls:0,parallelEntries:72,serialEntries:0,workerEntries:72,workerIds:[0]};return r;});
const comparator=new URL('./compare-rank.mjs',import.meta.url).pathname;
function run(name,mutate,expected,diagnostic){
 const dir=root+'/'+name;fs.mkdirSync(dir);const rows=structuredClone(variants);mutate(rows);
 const files=rows.map((r,i)=>{const p=dir+'/'+i+'.json';fs.writeFileSync(p,JSON.stringify(r)+'\n',{flag:'wx'});return p;});
 const out=dir+'/result.json';const r=cp.spawnSync(process.execPath,[comparator,...files,out],{encoding:'utf8',timeout:30000,maxBuffer:1024*1024});
 fs.writeFileSync(dir+'/stdout',r.stdout??'',{flag:'wx'});fs.writeFileSync(dir+'/stderr',r.stderr??'',{flag:'wx'});
 assert.equal(r.error,undefined,name);assert.equal(r.signal,null,name);
 if(expected==='PASS'){assert.equal(r.status,0,name);const j=JSON.parse(fs.readFileSync(out));for(const [i,b] of j.burnStorage.entries()){assert.deepEqual(b.completePacked32,rows[i].discrete.burn.idx);assert.deepEqual(b.activeIndices6,Array.from({length:6},(_,k)=>rows[i].discrete.burn.idx[k]));assert.deepEqual(b.separateCapturedInactiveBuffers,{cfg:rows[i].discrete.burn.cfg,num:rows[i].discrete.burn.num,proj:rows[i].discrete.burn.proj,spin:rows[i].discrete.burn.spin});}}else {assert.notEqual(r.status,0,name);if(expected==='DISCRETE'){const j=JSON.parse(fs.readFileSync(out));assert.equal(j.discrete.stage,'rank-local-discrete');assert.deepEqual(j.numeric,[]);}else assert.match(r.stderr,/AssertionError|TypeError/,name);if(diagnostic)assert.match(r.stderr,diagnostic,name);}
 console.log(name+' '+expected);
}
run('valid',()=>{},'PASS');
run('world4-width2-rank3-valid',a=>a.forEach(r=>{r.world=4;r.groupWidth=2;r.rank=3;r.group=1;r.groupRank=1;r.seedOffset=1;}),'PASS');
for(const [name,change] of [
 ['raw',r=>r.discrete.final.raw624[0]=1],['cursor',r=>r.discrete.final.cursor=1],['count',r=>r.discrete.final.words='1'],['future',r=>r.discrete.final.future624[0]=1],['saved',r=>r.discrete.saved.idx[0]=1],['burn',r=>r.discrete.burn.idx[30]=2],['scratch',r=>r.discrete.scratch.idx[0]=1],['counter',r=>r.discrete.counter[0]=1],
])run(name,a=>change(a[1]),'DISCRETE');
for(const [name,change] of [
 ['width',r=>r.originalDirect=31],['raw-u32',r=>r.discrete.final.raw624[0]=2**32],['cursor-range',r=>r.discrete.final.cursor=625],['count-leading-zero',r=>r.discrete.final.words='00'],['count-u128',r=>r.discrete.final.words=(2n**128n).toString()],['missing-raw',r=>r.discrete.final.raw624.pop()],['config-width',r=>r.discrete.saved.cfg.pop()],['config-value',r=>r.discrete.saved.num[0]=2],['unknown-field',r=>r.unknown=true],['missing-callback',r=>r.discrete.callbacks.pop()],['wrong-result',r=>r.result={Err:'partial'}],['requested-only',r=>r.observation.workerIds=[]],['numeric-shape',r=>r.numeric.direct.pop()],['numeric-value',r=>r.numeric.direct[0][0]=1],['wrong-rank',r=>r.rank=1],
])run(name,a=>change(a[1]),'SCHEMA_OR_NUMERIC');
for(const [name,change] of [
 ['all-missing-component',r=>r.numeric.direct[0].pop()],
 ['all-extra-component',r=>r.numeric.direct[0].push(0)],
 ['all-object-component',r=>r.numeric.direct[0]={re:0,im:0}],
 ['all-unknown-boundary',r=>r.discrete.final.unknown=0],
 ['all-unknown-observation',r=>r.observation.unknown=0],
 ['all-group-formula',r=>r.group=1],
 ['all-group-rank-formula',r=>r.groupRank=1],
 ['all-seed-offset-formula',r=>r.seedOffset=1],
 ['all-negative-counter',r=>r.discrete.counter[0]=-1],
 ['all-gutz-range',r=>r.discrete.saved.proj[0]=4],
 ['all-jastrow-range',r=>r.discrete.saved.proj[1]=16],
])run(name,a=>a.forEach(change),'SCHEMA_OR_NUMERIC');
for(const [name,change,diagnostic] of [
 ['burn-truncated',r=>r.discrete.burn.idx.pop(),/storage shape/],
 ['burn-extra-padding',r=>r.discrete.burn.idx.push(0),/storage shape/],
 ['burn-active-range',r=>r.discrete.burn.idx[0]=6,/burn active index/],
 ['burn-cfg-range',r=>r.discrete.burn.idx[6]=3,/burn packed cfg/],
 ['burn-num-range',r=>r.discrete.burn.idx[18]=2,/burn packed num/],
 ['burn-cfg-num-conflict',r=>r.discrete.burn.idx[18]=0,/cfg\/num consistency/],
 ['burn-order-conflict',r=>{r.discrete.burn.idx[6]=1;r.discrete.burn.idx[7]=0;},/index\/cfg order/],
 ['burn-gutz-range',r=>r.discrete.burn.idx[30]=4,/burn packed gutz/],
 ['burn-jastrow-range',r=>r.discrete.burn.idx[31]=16,/burn packed jastrow/],
])run(name,a=>a.forEach(change),'SCHEMA_OR_NUMERIC',diagnostic);
for(const [name,change] of [
 ['burn-packed-tail-discrete',r=>r.discrete.burn.idx[31]=-2],
 ['burn-inactive-cfg-discrete',r=>r.discrete.burn.cfg[0]=0],
 ['burn-inactive-num-discrete',r=>r.discrete.burn.num[0]=1],
 ['burn-inactive-proj-discrete',r=>r.discrete.burn.proj[0]=1],
])run(name,a=>change(a[1]),'DISCRETE');
console.log('2positive/47negative transport controls PASS; not a physics/model oracle');
