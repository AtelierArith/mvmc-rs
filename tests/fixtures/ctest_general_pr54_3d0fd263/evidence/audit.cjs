'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const [root, authority, destination] = process.argv.slice(2);
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const read = p => fs.readFileSync(p, 'utf8');
const check = (ok, message) => { if (!ok) throw Error(message); };
const numbers = p => { const a=read(p).trim().split(/\s+/).map(Number); check(a.every(Number.isFinite), 'nonfinite '+p); return a; };
const identity = path.join(root, 'published-source-complete.sha256');
check(sha(identity)==='33730954e5ed3b831369723fbff577d0571fbad5067c4697c665c420a4b69605', 'source manifest pin');
const project=path.join(root,'extern/Julia-mVMC');
const sourceRows=read(identity).split('\n').filter(x=>x&&!x.startsWith('#'));
check(sourceRows.length===63,'source closure');
const seen=new Set();
for(const line of sourceRows){const m=line.match(/^([a-f0-9]{64})  (.+)$/); check(m&&!seen.has(m[2]),'source row'); seen.add(m[2]); check(sha(path.join(project,m[2]))===m[1],'source hash '+m[2]);}
check(sha(path.join(project,'Manifest-v1.13.toml'))==='09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc','Manifest pin');
const stage=root+'-direct-v2';
check(read(path.join(stage,'provenance.txt')).includes('reference_commit=3d0fd2638fd34de2a8f9609fcfaac2504caf02d2'),'reference head');
const cases=[];
for(const steps of [1,2,3,20]){
 const dir=path.join(stage,'general_rbm_cmp','step-'+steps), settings=read(path.join(dir,'model-settings.txt'));
 for(const expected of ['RndSeed=12395','NSRCG=0 NStore=1','Julia_default_threads=1 Julia_interactive_threads=0 BLAS_threads=1 MPI=serial workers=1',`effective_NSROptItrStep=${steps} effective_NSROptItrSmp=${steps} override=both_no_clamp`,'initial_overlay=true']) check(settings.includes(expected),'settings '+expected);
 check(read(path.join(dir,'status.txt')).trim()==='0','runner status');
 const input=path.join(project,'test/integration/reference/general_rbm_cmp/inputs');
 const hashes=read(path.join(dir,'inputs.sha256')).trim().split('\n'), names=new Set();
 for(const row of hashes){const m=row.match(/^([a-f0-9]{64})  (.+)$/);check(m&&path.basename(m[2])===m[2]&&!names.has(m[2]),'input schema');names.add(m[2]);check(sha(path.join(input,m[2]))===m[1],'input identity');}
 for(const row of read(path.join(input,'namelist.def')).split('\n')){const t=row.trim().split(/\s+/);if(t.length===2&&!t[0].startsWith('#'))check(names.has(t[1]),'namelist closure '+t[1]);}
 check(names.has('initial.def')&&names.has('namelist.def'),'implicit initial closure');
 const raw=read(path.join(dir,'rng-state.txt')).trim().split('\n');check(raw.length===3,'RNG schema');
 const words=raw[0].trim().split(/\s+/).map(Number),index=Number(raw[1]),count=BigInt(raw[2]);
 check(words.length===624&&words.every(v=>Number.isInteger(v)&&v>=0&&v<=4294967295),'RAW624');check(Number.isInteger(index)&&index>=0&&index<=624&&count>=0n,'index/count');
 check(numbers(path.join(dir,'rng.txt')).length===624,'next624');
 const flags=read(path.join(dir,'direct-sr-flags.txt')).trim().split(/\s+/);check(flags.length===204&&flags.every(x=>x==='true'||x==='false'),'typed Boolean flags');
 const dimensions={'parameters.txt':204,'energy.txt':2,'sr_oo.txt':85696,'sr_ho.txt':412,'direct-sr-matrix.txt':194*194,'direct-sr-rhs.txt':194,'direct-sr-increment.txt':194,'direct-sr-active-indices.txt':194};
 for(const [file,length] of Object.entries(dimensions))check(numbers(path.join(dir,file)).length===length,'shape '+file);
 check(read(path.join(dir,'direct-sr-metrics.txt')).includes(`prefix=${steps} iteration=${steps} dimension=194 status=0`),'original solve status');
 const rows=read(path.join(dir,'c-window-input.txt')).trim().split('\n'),header=rows[0].split(/\s+/).map(Number);
 check(header.length===17&&header[0]===steps&&header[1]===102&&header.slice(2).reduce((a,b)=>a+b,0)===102&&rows.length===steps+1,'window schema');
 for(const row of rows.slice(1)){const a=row.split(/\s+/).map(Number);check(a.length===208&&a.every(Number.isFinite),'dense window');}
 const files={};for(const file of fs.readdirSync(dir).sort())if(fs.statSync(path.join(dir,file)).isFile())files[file]=sha(path.join(dir,file));
 cases.push({steps,seed:12395,NSRCG:0,NStore:1,window:steps,RAW624:words,index,draw_count:count.toString(),files});
}
const cSources={};for(const file of ['stcopt.c','stcopt_dposv.c']){const p=path.join(authority,'extern/mVMC-1.3.0/src/mVMC',file);cSources[p]=sha(p);}
const result={producer:{handle:78040,exit_code:1,failure:'post-numerical provenance tail: absent snapshot stcopt.c',all_four_numeric_captures_completed:true,on_disk_source_after_closure_reported:true},audit:{posthoc:true,does_not_relabel_producer_green:true,reference_root:project,C_authority_root:authority,C_sources:cSources,source_manifest_sha256:sha(identity),runtime_injection_identity_sha256:sha(path.join(root,'direct-v2-injection-identity.txt')),script_sha256:sha(__filename)},cases};
check(!fs.existsSync(destination),'refuse overwrite audit');fs.writeFileSync(destination,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log('POSTHOC AUDIT VERIFIED 4 cases/63 source files; producer78040 remains exit1; no fixture adoption');
