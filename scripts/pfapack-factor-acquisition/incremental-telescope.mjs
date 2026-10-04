// SOURCE-only conditional factor backward ceiling. O(n^2) work per shear;
// no dense congruence products, retained stages, or observed residual fitting.
import assert from 'node:assert/strict';
import {q,add,mul,sub,abs,cmp,json} from './rational.mjs';
import {recenter,quantize} from './dyadic.mjs';
const z=q(0n),one=q(1n);
const up=x=>quantize(x,true);
export function telescope(n){
 assert(Number.isSafeInteger(n)&&n>=2&&n<=256&&n%2===0);
 const at=(i,j)=>i+j*n;
 let h=Array.from({length:n*n},(_,i)=>({v:i%n===Math.floor(i/n)?one:z,e:z}));
 let ceiling=z,count=0;
 function norms(){
  let row=z,col=z;
  for(let i=0;i<n;i++){
   let r=z,c=z;
   for(let j=0;j<n;j++){
    const a=h[at(i,j)],b=h[at(j,i)];
    r=up(add(r,add(abs(a.v),a.e)));c=up(add(c,add(abs(b.v),b.e)));
   }
   if(cmp(r,row)>0)row=r;if(cmp(c,col)>0)col=c;
  }return {row,col};
 }
 return {
  stage(s){
   assert.equal(s.n??n,n);assert.equal(s.k,n-1-count,'stage chronology');
   const kk=s.k-1;
   if(s.kp!==kk)for(let i=0;i<n;i++)[h[at(i,kk)],h[at(i,s.kp)]]=[h[at(i,s.kp)],h[at(i,kk)]];
   const norm=norms();let local=z;
   for(let i=0;i<n;i++){
    let sum=z;for(let j=0;j<n;j++){const e=s.local.defect[at(i,j)];assert(e[0]>=0n);sum=up(add(sum,e));}
    if(cmp(sum,local)>0)local=sum;
   }
   // ||H E H^T||inf <= ||H||inf ||E||inf ||H^T||inf.
   ceiling=up(add(ceiling,up(mul(up(mul(norm.row,local)),norm.col))));
   // H G differs only in column kk. Capsule radii cover proof-representation
   // rounding ONLY; multipliers are exact borrowed binary64 operands.
   const next=[];
   for(let i=0;i<n;i++){
    let a=h[at(i,kk)];
    for(let j=0;j<kk;j++){
     const l=s.storedMultipliers[j],b=h[at(i,j)];
     a=recenter(add(a.v,mul(b.v,l)),up(add(a.e,mul(b.e,abs(l)))));
    }next.push(a);
   }
   for(let i=0;i<n;i++)h[at(i,kk)]=next[i];count++;
  },finish(factor,pivots){
   assert.equal(count,n-1);assert.equal(factor.length,n*n);assert.equal(pivots.length,n);
   const permutation=Array.from({length:n},(_,i)=>i);
   for(let i=n-1;i>=0;i--){const j=pivots[i]-1;assert(Number.isSafeInteger(j)&&j>=0&&j<n);[permutation[i],permutation[j]]=[permutation[j],permutation[i]];}
   // Same packed unit-upper definition consumed by utu2inv Step2:
   // Uii=1; Uij=F[i,j+1] (i<j<n-1); final column is e_(n-1).
   // Check H=P*U against separately propagated proof-representation radii.
   // No tolerance chosen from this discrepancy; failure is a bridge failure.
   for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const expected=i===j?one:(i<j&&j<n-1?factor[at(i,j+1)]:z);
    const capsule=h[at(permutation[i],j)];
    assert(cmp(abs(sub(expected,capsule.v)),capsule.e)<=0,'chronological H versus final packed U/P bridge');
   }
   return {validatedStages:count,packedInverseFactorBridge:'H=P*U verified within independent proof-representation radius',factorBackwardInfinityCeiling:json(ceiling),scope:'conditional local-operation factor telescope; not inverse/Pf acceptance'};
  }
 };
}
