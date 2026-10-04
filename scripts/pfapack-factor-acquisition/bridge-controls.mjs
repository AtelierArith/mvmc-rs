// Independent hand-authored multi-stage literals, not oracle or production calls.
import assert from 'node:assert/strict';
import {q,cmp} from './rational.mjs';
import {identity,product,congruence} from './matrix.mjs';
import {validateActualStage} from './stage-witness.mjs';
import {telescope} from './incremental-telescope.mjs';
const zero=q(0n),one=q(1n);
function upper(n,entries){const a=Array.from({length:n*n},()=>zero);for(const [i,j,v]of entries)a[i+j*n]=q(BigInt(v*4),4n);return a;}
export function handStages(n){
 assert(n===4||n===6);
 const originals=[[0,1,1],[0,2,2],[0,3,4],[1,2,3],[1,3,1],[2,3,2]];
 const firstSwap=[[0,1,-3],[0,2,-2],[0,3,2],[1,2,-1],[1,3,1],[2,3,4]];
 const firstAfter=[[0,1,-3],[0,2,-2],[0,3,.5],[1,2,-1],[1,3,.25],[2,3,4]];
 const secondSwap=[[0,1,3],[0,2,-1],[0,3,.25],[1,2,-2],[1,3,.5],[2,3,4]];
 const final=[[0,1,3],[0,2,.5],[0,3,.25],[1,2,-2],[1,3,.5],[2,3,4]];
 const make=e=>upper(n,n===6?[...e,[4,5,8]]:e);
 const original=make(originals),a=make(firstSwap),b=make(firstAfter),c=make(secondSwap),f=make(final);
 const stages=[];
 if(n===6){
  stages.push({n,k:5,kp:4,beforeUpper:original,afterSwapUpper:original,afterUpper:original,info:0});
  // Original zero-column INFO=4, despite nonsingular disjoint blocks.
  stages.push({n,k:4,kp:3,beforeUpper:original,afterSwapUpper:original,afterUpper:original,info:4});
 }
 stages.push({n,k:3,kp:0,beforeUpper:original,afterSwapUpper:a,afterUpper:b,info:n===6?4:0});
 stages.push({n,k:2,kp:0,beforeUpper:b,afterSwapUpper:c,afterUpper:f,info:n===6?4:0});
 stages.push({n,k:1,kp:0,beforeUpper:f,afterSwapUpper:f,afterUpper:f,info:n===6?4:0});
 return {n,original,factor:f,pivots:n===4?[1,1,1,4]:[1,1,1,4,5,6],stages};
}
export function bridgeControls(){
 const results=[];
 for(const n of [4,6]){
  const fixture=handStages(n),h=identity(n),u=identity(n),t=Array.from({length:n*n},()=>zero),at=(i,j)=>i+j*n;
  // Hand H columns: e1; .5e1+e2; e0+.25e1+.5e2; e3 (plus e4,e5).
  for(let j=0;j<3;j++)for(let i=0;i<n;i++)h[at(i,j)]=zero;
  h[at(1,0)]=one;h[at(1,1)]=q(1n,2n);h[at(2,1)]=one;
  h[at(0,2)]=one;h[at(1,2)]=q(1n,4n);h[at(2,2)]=q(1n,2n);
  const band=n===4?[3,-2,4]:[3,-2,4,0,8];
  for(let i=0;i<n-1;i++){t[at(i,i+1)]=q(BigInt(band[i]));t[at(i+1,i)]=q(BigInt(-band[i]));}
  const restored=congruence(h,t,n);
  for(let j=1;j<n;j++)for(let i=0;i<j;i++)assert.equal(cmp(restored[at(i,j)],fixture.original[at(i,j)]),0,'hand A=H T H-transpose');
  for(let j=0;j<n-1;j++)for(let i=0;i<j;i++)u[at(i,j)]=fixture.factor[at(i,j+1)];
  const p=Array.from({length:n*n},()=>zero),permutation=n===4?[1,2,0,3]:[1,2,0,3,4,5];
  for(let j=0;j<n;j++)p[at(permutation[j],j)]=one;
  const mapped=product(p,u,n);assert.deepEqual(mapped,h,'independent literal final packed U/P bridge');
  const proof=telescope(n);for(const s of fixture.stages)proof.stage(validateActualStage(s));
  const out=proof.finish(fixture.factor,fixture.pivots);
  results.push({name:`n${n}_two_pivots_two_shears_and_final_packed_inverse_U`,actual:'PASS',...out});
 }
 return results;
}
