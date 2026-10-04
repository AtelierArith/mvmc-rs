// Check byte counts BEFORE retaining chunks or allocating a complete frame.
import assert from 'node:assert/strict';
export async function* frames(input, frameCap=2*1024*1024, totalCap=128*1024*1024) {
 let pieces=[], pending=0,total=0;
 for await(const raw of input){
  assert(Buffer.isBuffer(raw),'byte stream required');
  total+=raw.length;assert(total<=totalCap,'transport stream cap');
  let start=0;
  for(let i=0;i<raw.length;i++)if(raw[i]===10){
   const size=i-start;assert(pending+size<=frameCap,'stage frame cap');
   pieces.push(raw.subarray(start,i));
   const line=Buffer.concat(pieces,pending+size);
   assert(line.length>0,'empty frame');
   yield line.toString('utf8');pieces=[];pending=0;start=i+1;
  }
  const size=raw.length-start;assert(pending+size<=frameCap,'stage frame cap');
  if(size){pieces.push(Buffer.from(raw.subarray(start)));pending+=size;}
 }
 assert.equal(pending,0,'incomplete trailing frame');
}
