// Exact rational local operation bounds from independent reference operands.
import assert from 'node:assert/strict';
import {q,add,mul,sub,div,abs,cmp} from './rational.mjs';
export function nonnegativeRadius(e){assert(Array.isArray(e)&&e.length===2&&typeof e[0]==='bigint'&&typeof e[1]==='bigint'&&e[1]>0n,'invalid rational radius');assert(e[0]>=0n,'negative radius');return e;}
const u=q(1n,1n<<53n),tiny=q(1n,1n<<1074n);
export function roundedError(ideal,operandError){nonnegativeRadius(operandError);return add(operandError,add(mul(u,add(abs(ideal),operandError)),tiny));}
export function addition(a,ea,b,eb){nonnegativeRadius(ea);nonnegativeRadius(eb);const value=add(a,b);return {value,error:roundedError(value,add(ea,eb))};}
export function multiplication(a,ea,b,eb){nonnegativeRadius(ea);nonnegativeRadius(eb);const value=mul(a,b),uncertainty=add(add(mul(abs(a),eb),mul(abs(b),ea)),mul(ea,eb));return {value,error:roundedError(value,uncertainty)};}
export function reciprocal(a,ea){nonnegativeRadius(ea);const lower=sub(abs(a),ea);assert(cmp(lower,q(0n))>0,'reference neighbourhood denominator not separated');const value=div(q(1n),a),uncertainty=div(ea,mul(abs(a),lower));return {value,error:roundedError(value,uncertainty)};}
export function rank2(old,eold,x,ex,y,ey,alpha,ealpha,xj,exj,yj,eyj){
 const t1=multiplication(alpha,ealpha,yj,eyj),t2=multiplication(alpha,ealpha,xj,exj);
 const p1=multiplication(x,ex,t1.value,t1.error),s=addition(old,eold,p1.value,p1.error),p2=multiplication(y,ey,t2.value,t2.error);
 const end=addition(s.value,s.error,[-p2.value[0],p2.value[1]],p2.error);
 return {...end,nodes:{t1,t2,p1,sum:s,p2,after:end},scope:'normal/subnormal reference neighbourhood tree; overflow must be separately excluded'};
}
