// SOURCE only; actual borrowed upper snapshots are required, not labels.
import assert from 'node:assert/strict';
import {q,add,sub,mul,div,abs,cmp} from './rational.mjs';
import {rank2,roundedError} from './local-defect.mjs';
import {localFactorStage} from './local-factor.mjs';
const zero=q(0n),one=q(1n);
export function logicalSkewFromUpper(physical,n){
 assert(Number.isSafeInteger(n)&&n>=2&&physical.length===n*n);
 const a=Array.from({length:n*n},()=>zero);
 for(let j=1;j<n;j++)for(let i=0;i<j;i++){const v=physical[i+j*n];assert(Array.isArray(v)&&v.length===2&&v[1]>0n);a[i+j*n]=v;a[j+i*n]=[-v[0],v[1]];}
 // Never read physical lower/diagonal; C UPLO='U' ignores stale cells.
 return a;
}
export function validateActualStage({n,k,kp,beforeUpper,afterSwapUpper,afterUpper}){
 assert(Number.isSafeInteger(k)&&k>=1&&k<n&&Number.isSafeInteger(kp)&&kp>=0&&kp<k);
 const before=logicalSkewFromUpper(beforeUpper,n),pre=logicalSkewFromUpper(afterSwapUpper,n),at=(i,j)=>i+j*n;
 let chosen=0;for(let i=1;i<k;i++)if(cmp(abs(before[at(i,k)]),abs(before[at(chosen,k)]))>0)chosen=i;
 const zeroColumn=before[at(chosen,k)][0]===0n;
 const expectedPivot=zeroColumn?k-1:chosen;
 assert.equal(kp,expectedPivot,'actual preBeforeSwap first-max pivot mismatch');
 const permutation=Array.from({length:n},(_,i)=>i);[permutation[k-1],permutation[kp]]=[permutation[kp],permutation[k-1]];
 for(let j=1;j<n;j++)for(let i=0;i<j;i++)assert.equal(cmp(pre[at(i,j)],before[at(permutation[i],permutation[j])]),0,'actual upper swap/sign mismatch');
 assert.equal(afterUpper.length,n*n);
 for(let j=1;j<n;j++)for(let i=0;i<j;i++){
  const updatedSchur=j<k-1,scaledMultiplier=j===k&&i<k-1;
  if(!updatedSchur&&!scaledMultiplier)assert.equal(cmp(afterUpper[at(i,j)],pre[at(i,j)]),0,'inactive upper changed outside source-defined update/scaling domain');
 }
 const multipliers=Array.from({length:k-1},(_,i)=>afterUpper[at(i,k)]);
 const local=localFactorStage(pre,multipliers,n,k);
 const p=pre[at(k-1,k)];
 if(!zeroColumn&&k>1){
  const alpha=div(one,p),alphaError=roundedError(alpha,zero);
  for(let j=0;j<k-1;j++)for(let i=0;i<j;i++){
   const x=pre[at(i,k)],y=pre[at(i,k-1)],xj=pre[at(j,k)],yj=pre[at(j,k-1)];
   const r=(xj[0]===0n&&yj[0]===0n)?{value:pre[at(i,j)],error:zero}:rank2(pre[at(i,j)],zero,x,zero,y,zero,alpha,alphaError,xj,zero,yj,zero);
   assert(cmp(abs(sub(afterUpper[at(i,j)],r.value)),r.error)<=0,'actual Schur output outside prospective C-left local ceiling');
  }
 }else{
  for(let j=1;j<=k;j++)for(let i=0;i<j;i++)assert.equal(cmp(afterUpper[at(i,j)],pre[at(i,j)]),0,'no-arithmetic stage changed upper storage');
 }
 assert.equal(cmp(afterUpper[at(k-1,k)],p),0,'factor stage altered pivot');
 return {preAfterSwap:pre,storedMultipliers:multipliers,local,k,kp,pivotAdmission:'actual supported first-max, including ties; no forced native-reference pivot',scope:'actual numeric stage containment, conditional on independently audited producer source DAG'};
}
