import assert from 'node:assert/strict';
// JSON.parse validates grammar; lexical pass rejects duplicate object keys first.
export function parseFrame(text){
 const stack=[];
 for(let i=0;i<text.length;i++){
  const c=text[i];
  if(c==='"'){
   const start=i;i++;
   for(;i<text.length;i++){if(text[i]==='\\'){i++;continue;}if(text[i]==='"')break;}
   assert(i<text.length,'unterminated JSON string');
   let j=i+1;while(j<text.length&&/\s/.test(text[j]))j++;
   if(text[j]===':'){
    assert(stack.length&&stack.at(-1)!==null,'key outside object');const key=JSON.parse(text.slice(start,i+1));
    assert(!stack.at(-1).has(key),'duplicate JSON key');stack.at(-1).add(key);
   }
  }else if(c==='{')stack.push(new Set());else if(c==='[')stack.push(null);
  else if(c==='}'||c===']'){assert(stack.length,'unbalanced JSON');stack.pop();}
 }
 assert.equal(stack.length,0,'incomplete JSON');return JSON.parse(text);
}
