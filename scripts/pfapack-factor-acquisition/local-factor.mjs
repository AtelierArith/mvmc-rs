// Local full-N upper C-left factor reconstruction ceiling from actual operands.
// No Rust-C discrepancies or global pivot radii. SOURCE only.
import assert from 'node:assert/strict';
import {q,add,mul,div,sub,abs,cmp} from './rational.mjs';
import {rank2,roundedError} from './local-defect.mjs';
import {identity} from './matrix.mjs';
const zero=q(0n),one=q(1n),max=q(((1n<<53n)-1n)<<971n);
function finite(v,e=zero){assert(e[1]>0n&&e[0]>=0n);assert(cmp(add(abs(v),e),max)<0,'factor local neighbourhood overflow not excluded');}
export function localFactorStage(pre,multipliers,n,k){
 assert(Number.isSafeInteger(n)&&n>=2&&n%2===0&&pre.length===n*n);
 assert(Number.isSafeInteger(k)&&k>=1&&k<n&&multipliers.length===k-1);
 const kk=k-1,at=(i,j)=>i+j*n,p=pre[at(kk,k)],g=identity(n),defect=Array.from({length:n*n},()=>zero);
 for(const v of pre)finite(v);
 // Pre is the mathematical skew active matrix AFTER the prescribed interchange.
 for(let i=0;i<=k;i++)for(let j=0;j<=k;j++)assert.equal(cmp(add(pre[at(i,j)],pre[at(j,i)]),zero),0,'local active input must be skew');
 const colZero=Array.from({length:k},(_,i)=>pre[at(i,k)][0]===0n).every(Boolean);
 if(colZero){assert(multipliers.every(v=>v[0]===0n),'zero column cannot supply nonzero multipliers');return {transform:g,defect,zeroColumn:true};}
 assert(p[0]!==0n,'factor pivot denominator zero');
 // Actual Rust kk0==0 performs no reciprocal, rank2 or scaling.
 if(kk===0)return {transform:g,defect,zeroColumn:false,scope:'last pair, no arithmetic stage'};
 const alpha=div(one,p),alphaError=roundedError(alpha,zero);finite(alpha,alphaError);
 const scalingError=[];
 for(let i=0;i<kk;i++){
  const x=pre[at(i,k)],ideal=mul(alpha,x);
  // Exact zero DSCAL operand remains zero, otherwise reciprocal then multiply.
  const error=x[0]===0n?zero:roundedError(ideal,mul(abs(x),alphaError));finite(ideal,error);finite(multipliers[i]);
  // Verify only the theorem's predicted operation neighbourhood, not fit its radius.
  assert(cmp(abs(sub(multipliers[i],ideal)),error)<=0,'stored multiplier outside prospective local scaling ceiling');
  scalingError.push(error);g[at(i,kk)]=multipliers[i];
  const e=mul(abs(p),error);defect[at(i,k)]=e;defect[at(k,i)]=e;
 }
 for(let j=0;j<kk;j++)for(let i=0;i<j;i++){
  const x=pre[at(i,k)],y=pre[at(i,kk)],xj=pre[at(j,k)],yj=pre[at(j,kk)];
  let updateError=zero;
  if(xj[0]!==0n||yj[0]!==0n){
   const r=rank2(pre[at(i,j)],zero,x,zero,y,zero,alpha,alphaError,xj,zero,yj,zero);
   for(const node of Object.values(r.nodes))finite(node.value,node.error);
   updateError=r.error;
  }
  // Reverse reconstruction uses stored l, not ideal x/p:
  // Aij = Schur_ij - li*yj + yi*lj.
  const e=add(updateError,add(mul(scalingError[i],abs(yj)),mul(abs(y),scalingError[j])));
  defect[at(i,j)]=e;defect[at(j,i)]=e;
 }
 return {transform:g,defect,zeroColumn:false,scope:'local active pre - G*reduced*G^T ceiling, conditional on audited actual rank2 output and stored scaling path'};
}
