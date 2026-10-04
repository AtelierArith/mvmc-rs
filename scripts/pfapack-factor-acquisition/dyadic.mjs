// Exact-integer proof representation only. Neither production rounding nor oracle output.
import assert from 'node:assert/strict';
import {q,add,sub,abs,cmp} from './rational.mjs';
const zero=q(0n),max=q(((1n<<53n)-1n)<<971n);
export const PRECISION=128;
function exponent(a,b){
 let e=a.toString(2).length-b.toString(2).length;
 if(e>=0){if(a<(b<<BigInt(e)))e--;}
 else if((a<<BigInt(-e))<b)e--;
 assert(Number.isSafeInteger(e)&&Math.abs(e)<=16384,'dyadic exponent outside bounded proof representation');return e;
}
export function quantize(x,upward=false){
 assert(Array.isArray(x)&&x.length===2&&typeof x[0]==='bigint'&&typeof x[1]==='bigint'&&x[1]>0n);
 if(x[0]===0n)return zero;
 assert(!upward||x[0]>0n,'upward radius must be nonnegative');
 const negative=x[0]<0n,a=negative?-x[0]:x[0],b=x[1],shift=exponent(a,b)-(PRECISION-1);
 const numerator=shift<0?a<<BigInt(-shift):a,denominator=shift>0?b<<BigInt(shift):b;
 let m=numerator/denominator;const remainder=numerator%denominator;
 if(upward){if(remainder!==0n)m++;}
 else if(2n*remainder>denominator||(2n*remainder===denominator&&(m&1n)!==0n))m++;
 if(negative)m=-m;
 return shift>=0?q(m<<BigInt(shift)):q(m,1n<<BigInt(-shift));
}
export function recenter(v,e){
 assert(e[1]>0n&&e[0]>=0n,'negative or invalid capsule radius');
 assert(cmp(add(abs(v),e),max)<0,'finite neighbourhood not certified');
 const center=quantize(v),error=quantize(add(e,abs(sub(v,center))),true);
 assert(cmp(add(abs(center),error),max)<0,'recentered finite neighbourhood not certified');
 return {v:center,e:error};
}
