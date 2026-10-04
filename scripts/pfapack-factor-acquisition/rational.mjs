import assert from 'node:assert/strict';
function gcd(a,b){a=a<0n?-a:a;while(b)[a,b]=[b,a%b];return a;}
export function q(a,b=1n){assert(b!==0n);if(b<0n){a=-a;b=-b;}const g=gcd(a,b);return [a/g,b/g];}
export const add=(a,b)=>q(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);
export const neg=a=>[-a[0],a[1]];
export const sub=(a,b)=>add(a,neg(b));
export const mul=(a,b)=>q(a[0]*b[0],a[1]*b[1]);
export const div=(a,b)=>q(a[0]*b[1],a[1]*b[0]);
export const abs=a=>[a[0]<0n?-a[0]:a[0],a[1]];
export const cmp=(a,b)=>{const d=a[0]*b[1]-b[0]*a[1];return d<0n?-1:d>0n?1:0;};
export const equal=(a,b)=>cmp(a,b)===0;
export function word(w){assert(/^[0-9a-f]{16}$/.test(w));const v=BigInt('0x'+w),e=Number(v>>52n&2047n);assert(e!==2047);let m=v&((1n<<52n)-1n);if(e)m|=1n<<52n;if(v>>63n)m=-m;const k=e?e-1075:-1074;return k>=0?q(m<<BigInt(k)):q(m,1n<<BigInt(-k));}
export const json=a=>a.map(String);
