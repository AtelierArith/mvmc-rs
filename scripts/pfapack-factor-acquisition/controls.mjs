// SOURCE controls only until a separately approved execution / CI run.
import assert from 'node:assert/strict';
import {Readable} from 'node:stream';
import {bridgeControls,handStages} from './bridge-controls.mjs';
import {createConsumer} from './consumer.mjs';
import {parseFrame} from './closed-json.mjs';
import {frames} from './bounded-framer.mjs';
const results=[];
function check(name,f){f();results.push({name,actual:'PASS'});}
function rejects(name,f,message){assert.throws(f,{code:'ERR_ASSERTION',message});results.push({name,actual:'EXPECTED_REJECTION'});}
function word(x){const b=Buffer.alloc(8);b.writeDoubleBE(x);return b.toString('hex');}
function words(a){return a.map(q=>word(Number(q[0])/Number(q[1])));}
function rows(){
 const fixture=handStages(4),records=[{kind:'case-start',n:4}],initial=words(fixture.original);let previous=initial;
 function patches(next){const w=words(next),out=[];for(let j=1;j<4;j++)for(let i=0;i<j;i++){const at=i+j*4;if(w[at]!==previous[at])out.push([at,w[at]]);}previous=w;return out;}
 for(const s of fixture.stages){records.push({kind:'before-pivot',n:4,k:s.k,borrowedUpperMatchesPrevious:true});records.push({kind:'after-swap',n:4,k:s.k,kp:s.kp,info:0,patches:patches(s.afterSwapUpper)});records.push({kind:'after-update',n:4,k:s.k,kp:s.kp,info:0,patches:patches(s.afterUpper)});}
 records.push({kind:'result-plane',n:4,name:'factor',words:words(fixture.factor)});
 // Finite schema-only synthetic inverse plane; not an inverse-acceptance test.
 records.push({kind:'result-plane',n:4,name:'inverse',words:Array(16).fill(word(0))});
 records.push({kind:'case-end',n:4,info:0,pfWord:word(12),pivots1based:fixture.pivots});
 return {records,initial};
}
function consume(records){const {initial}=rows(),c=createConsumer(()=>{},()=>({inputWords:initial}),[4]);for(const r of records)c.accept(r);return c.finish();}
results.push(...bridgeControls());
check('closed_complete_hand_stream',()=>assert.equal(consume(rows().records).summaryComplete,true));
rejects('duplicate_JSON_key',()=>parseFrame('{"kind":"case-start","kind":"case-end","n":4}'),/duplicate JSON key/);
check('malformed_JSON_grammar',()=>assert.throws(()=>parseFrame('{"n":4,}'),SyntaxError));
rejects('unknown_event_key',()=>{const r=rows().records;r[0].unexpected=true;consume(r);},/closed event keys/);
rejects('duplicate_upper_patch',()=>{const r=rows().records;r[2].patches.push(r[2].patches[0]);consume(r);},/duplicate patch/);
rejects('missing_stage',()=>{const r=rows().records;r.splice(3,1);consume(r);},/Expected values to be strictly equal/);
rejects('missing_case_end',()=>{const r=rows().records;r.pop();consume(r);},/incomplete case/);
rejects('missing_inverse_plane',()=>{const r=rows().records;r.splice(r.length-2,1);consume(r);},/Expected values to be strictly equal/);
rejects('duplicate_result_plane',()=>{const r=rows().records;r.splice(r.length-1,0,r[r.length-2]);consume(r);},/duplicate result plane/);
rejects('reversed_result_planes',()=>{const r=rows().records;[r[r.length-3],r[r.length-2]]=[r[r.length-2],r[r.length-3]];consume(r);},/actual result plane chronology/);
rejects('unknown_event_kind',()=>{const r=rows().records;r[1].kind='unknown';consume(r);},/unknown event/);
rejects('invented_INFO',()=>{const r=rows().records;r[2].info=2;consume(r);},/actual INFO zero-column chronology/);
rejects('wrong_final_pivot',()=>{const r=rows().records;r.at(-1).pivots1based=[1,2,3,4];consume(r);},/actual pivot boundary/);
rejects('nonfinite_result_word',()=>{const r=rows().records;r.at(-2).words[0]='7ff0000000000000';consume(r);},/The expression evaluated to a falsy value/);
rejects('short_result_plane',()=>{const r=rows().records;r.at(-2).words.pop();consume(r);},/The expression evaluated to a falsy value/);
// Async framer failures are checked directly, never caught by the control's
// own "unexpected acceptance" assertion.
const framed=[];for await(const s of frames(Readable.from([Buffer.from('{"n"'),Buffer.from(':4}\n')]),16,32))framed.push(s);
assert.deepEqual(framed,['{"n":4}']);results.push({name:'byte_framer_split_line',actual:'PASS'});
await assert.rejects(async()=>{for await(const _ of frames(Readable.from([Buffer.alloc(17,65)]),16,32)){assert.fail('oversized frame yielded');}},{code:'ERR_ASSERTION',message:/stage frame cap/});results.push({name:'byte_cap_before_complete_frame',actual:'EXPECTED_REJECTION'});
await assert.rejects(async()=>{for await(const _ of frames(Readable.from([Buffer.from('{}')]),16,32)){}},{code:'ERR_ASSERTION',message:/incomplete trailing frame/});results.push({name:'truncated_frame',actual:'EXPECTED_REJECTION'});
await assert.rejects(async()=>{for await(const _ of frames(Readable.from([Buffer.from('{}\n{}\n')]),16,5)){}},{code:'ERR_ASSERTION',message:/transport stream cap/});results.push({name:'total_transport_byte_cap',actual:'EXPECTED_REJECTION'});
console.log(JSON.stringify({scope:'synthetic schema/framer plus independent n4/n6 bridge literals; not large-case or inverse/Pf acceptance',results}));
