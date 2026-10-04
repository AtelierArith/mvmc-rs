// Actual selected libtest producer -> bounded FD3 stream -> independent verifier.
// No detached processes, alternate numerical kernel, compiler, or runtime oracle.
import assert from 'node:assert/strict';
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {consumeStream} from './consumer.mjs';
import {hashes,loadFixture} from './fixture-loader.mjs';
const [bindingPath,out]=process.argv.slice(2);assert.equal(process.argv.length,4);
for(const key of ['NODE_OPTIONS','NODE_PATH'])assert(!(key in process.env),'unexpected Node environment '+key);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const bindingBytes=fs.readFileSync(bindingPath),binding=JSON.parse(bindingBytes);
assert.equal(binding.head,process.env.GITHUB_SHA);assert.equal(binding.profile,'ci');assert.deepEqual(binding.selected.features,['default']);
for(const n of hashes.keys())loadFixture(n); // All fixture hashes checked before the producer starts.
function checkArtifacts(){assert.equal(hash(bindingPath),crypto.createHash('sha256').update(bindingBytes).digest('hex'));for(const [p,h]of binding.artifacts)assert.equal(hash(p),h,'selected ELF/fingerprint/provider changed '+p);}
checkArtifacts();
const record=(name,data)=>fs.writeFileSync(path.join(out,name),JSON.stringify(data,null,2)+'\n',{flag:'wx'});
const streamFD=fs.openSync(path.join(out,'stream.ndjson'),'wx'),summaries=[];
let child=null,code=null,signal=null,primary=1,post=1,rawBytes=0,streamHash=crypto.createHash('sha256');
const logs=[];
function logPipe(input,name){
 const fd=fs.openSync(path.join(out,name),'wx');let bytes=0;
 const promise=(async()=>{try{for await(const chunk of input){bytes+=chunk.length;assert(bytes<=1024*1024,'child log cap '+name);fs.writeSync(fd,chunk);}return {name,bytes,status:0};}finally{fs.closeSync(fd);}})();
 logs.push(promise);return promise;
}
try{
 child=spawn(binding.selected.binary,['--ignored','--exact',binding.selected.name,'--nocapture','--test-threads=1'],{
  cwd:path.resolve('crates/pfapack'),detached:false,
  env:{...process.env,MVMC_FACTOR_ADMISSION_PIPE:'1',OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1',BLIS_NUM_THREADS:'1'},
  stdio:['ignore','pipe','pipe','pipe']
 });
 const completion=new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',(status,sig)=>{code=status;signal=sig;resolve();});});
 record('producer-process.json',{pid:child.pid,head:binding.head,binary:binding.selected.binary,name:binding.selected.name,fd3:'anonymous bounded pipe',noDetachedGroup:true});
 const stdout=logPipe(child.stdout,'producer.stdout'),stderr=logPipe(child.stderr,'producer.stderr');
 async function* stream(){for await(const chunk of child.stdio[3]){rawBytes+=chunk.length;assert(rawBytes<=128*1024*1024,'raw stream cap');streamHash.update(chunk);fs.writeSync(streamFD,chunk);yield chunk;}}
 const verified=consumeStream(stream(),summary=>{summaries.push(summary);record(`n${summary.n}.summary.json`,summary);console.log(JSON.stringify({n:summary.n,validatedStages:summary.validatedStages,scope:summary.scope}));});
 await Promise.all([verified,stdout,stderr,completion]);
 assert.equal(code,0,'actual ignored producer failed');assert.equal(signal,null,'actual producer signal');assert.equal(summaries.length,4,'complete four-case acquisition');
 record('acquisition.json',{head:binding.head,profile:'ci',features:['default'],rawBytes,streamSHA:streamHash.digest('hex'),producerCode:code,producerSignal:signal,summaries,scientificAcceptance:false,reason:'conditional factor source witness only; inverse/Pf/provider acceptance pending'});
 primary=0;
}catch(error){
 record('failure.json',{message:String(error),producerCode:code,producerSignal:signal,completeCases:summaries.length,rawBytes});
 // Only this exact ChildProcess handle is signalled, never a guessed PID/group.
 if(child&&code===null&&signal===null){child.kill('SIGTERM');await Promise.race([once(child,'exit'),new Promise(resolve=>setTimeout(resolve,5000))]);if(child.exitCode===null&&child.signalCode===null){child.kill('SIGKILL');await once(child,'exit');}}
 await Promise.allSettled(logs);
}finally{
 fs.closeSync(streamFD);
 try{checkArtifacts();post=0;}catch(error){record('post-failure.json',{message:String(error)});}
 record('acquisition-terminal.json',{primary,post,producerCode:code,producerSignal:signal,completeCases:summaries.length,scientificAcceptance:false});
}
process.exitCode=primary===0&&post===0?0:1;
