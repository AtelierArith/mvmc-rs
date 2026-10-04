import {q,add,mul,abs,cmp} from './rational.mjs';
export const identity=n=>Array.from({length:n*n},(_,k)=>q(BigInt(k%n===Math.floor(k/n))));
export function product(a,b,n){const c=Array(n*n);for(let j=0;j<n;j++)for(let i=0;i<n;i++){let s=q(0n);for(let k=0;k<n;k++)s=add(s,mul(a[i+k*n],b[k+j*n]));c[i+j*n]=s;}return c;}
export const transpose=(a,n)=>Array.from({length:n*n},(_,k)=>a[Math.floor(k/n)+(k%n)*n]);
export const congruence=(h,d,n)=>product(product(h,d,n),transpose(h,n),n);
export function norm(a,n){let m=q(0n);for(let i=0;i<n;i++){let s=q(0n);for(let j=0;j<n;j++)s=add(s,abs(a[i+j*n]));if(cmp(s,m)>0)m=s;}return m;}
