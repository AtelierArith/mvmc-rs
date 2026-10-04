// SOURCE-only acquisition validator. Actual results retained as data, not oracle gates.
import assert from 'node:assert/strict';
import {word,cmp} from './rational.mjs';
import {validateActualStage} from './stage-witness.mjs';
import {loadFixture} from './fixture-loader.mjs';
import {frames} from './bounded-framer.mjs';
import {telescope} from './incremental-telescope.mjs';
import {parseFrame} from './closed-json.mjs';
function closed(r,keys){assert(r&&typeof r==='object'&&!Array.isArray(r));assert.deepEqual(Object.keys(r).sort(),keys.sort(),'closed event keys');}
export function createConsumer(onCase,loader=loadFixture,sizes=[32,64,128,256]){
 let active=null,index=0;
 function patches(base,rows,n){
  assert(Array.isArray(rows)&&rows.length<=n*(n-1)/2);const out=base.slice(),seen=new Set();
  for(const r of rows){assert(Array.isArray(r)&&r.length===2);const [i,w]=r;
   assert(Number.isSafeInteger(i)&&i>=0&&i<n*n&&i%n<Math.floor(i/n));assert(!seen.has(i),'duplicate patch');seen.add(i);out[i]=word(w);
  }return out;
 }
 return {accept(r){
  if(r.kind==='case-start'){
   closed(r,['kind','n']);assert.equal(active,null);assert.equal(r.n,sizes[index]);const f=loader(r.n);
   active={n:r.n,k:r.n-1,phase:'before',upper:f.inputWords.map(word),upperWords:f.inputWords.slice(),info:0,piv:Array(r.n).fill(null),planes:new Set(),proof:telescope(r.n)};
   active.piv[r.n-1]=r.n;return;
  }
  assert(active,'event outside case');const a=active;assert.equal(r.n,a.n);
  if(r.kind==='before-pivot'){
   closed(r,['kind','n','k','borrowedUpperMatchesPrevious']);assert.equal(a.phase,'before');assert(a.k>0);assert.equal(r.k,a.k);assert.equal(r.borrowedUpperMatchesPrevious,true);
   a.before=a.upper.slice();a.phase='swap';return;
  }
  if(r.kind==='after-swap'){
   closed(r,['kind','n','k','kp','info','patches']);assert.equal(a.phase,'swap');assert.equal(r.k,a.k);assert(Number.isSafeInteger(r.kp)&&r.kp>=0&&r.kp<a.k);
   const zero=Array.from({length:a.k},(_,i)=>a.before[i+a.k*a.n][0]===0n).every(Boolean);
   const expected=a.info||(zero?a.k:0);assert.equal(r.info,expected,'actual INFO zero-column chronology');a.info=expected;
   a.pre=patches(a.upper,r.patches,a.n);a.preWords=a.upperWords.slice();for(const [i,w]of r.patches)a.preWords[i]=w;
   a.kp=r.kp;a.piv[a.k-1]=r.kp+1;a.phase='update';return;
  }
  if(r.kind==='after-update'){
   closed(r,['kind','n','k','kp','info','patches']);assert.equal(a.phase,'update');assert.equal(r.k,a.k);assert.equal(r.kp,a.kp);assert.equal(r.info,a.info);
   const after=patches(a.pre,r.patches,a.n);
   const s=validateActualStage({n:a.n,k:a.k,kp:a.kp,beforeUpper:a.before,afterSwapUpper:a.pre,afterUpper:after});
   a.proof.stage(s);a.upper=after;a.upperWords=a.preWords;for(const [i,w]of r.patches)a.upperWords[i]=w;
   a.before=null;a.pre=null;a.preWords=null;a.k--;a.phase='before';return;
  }
  if(r.kind==='result-plane'){
   closed(r,['kind','n','name','words']);assert.equal(a.phase,'before');assert.equal(a.k,0);assert(['factor','inverse'].includes(r.name));assert(!a.planes.has(r.name),'duplicate result plane');assert.equal(r.name,a.planes.size===0?'factor':'inverse','actual result plane chronology');
   assert(Array.isArray(r.words)&&r.words.length===a.n*a.n);const plane=r.words.map(word);
   if(r.name==='factor')for(let j=1;j<a.n;j++)for(let i=0;i<j;i++)assert.equal(r.words[i+j*a.n],a.upperWords[i+j*a.n],'actual final factor storage association (including zero sign)');
   if(r.name==='factor')a.factor=plane;
   a.planes.add(r.name);return;
  }
  if(r.kind==='case-end'){
   closed(r,['kind','n','info','pfWord','pivots1based']);assert.equal(a.phase,'before');assert.equal(a.k,0);assert.equal(a.planes.size,2);assert.equal(r.info,a.info);word(r.pfWord);assert.deepEqual(r.pivots1based,a.piv,'actual pivot boundary');
   onCase({n:a.n,actualInfo:a.info,actualPfWord:r.pfWord,...a.proof.finish(a.factor,a.piv),inverseAdmission:'NOT_IMPLEMENTED',pfAcceptance:'NOT_IMPLEMENTED',providerAdmission:'NOT_IMPLEMENTED'});active=null;index++;return;
  }assert.fail('unknown event');
 },finish(){assert.equal(active,null,'incomplete case');assert.equal(index,sizes.length,'incomplete four-case stream');return {summaryComplete:true};}};
}
export async function consumeStream(input,onCase){const c=createConsumer(onCase);for await(const line of frames(input))c.accept(parseFrame(line));return c.finish();}
