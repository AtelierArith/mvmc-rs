import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';import assert from 'node:assert/strict';import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
export function validateList({file,root,target,feature,mode,out,headExpected},deps={fs,spawnSync}){
 const fs=deps.fs,spawnSync=deps.spawnSync;
assert(['default','simd-backend','blas-backend','both'].includes(feature));assert(['ordinary','ignored'].includes(mode));
const head=spawnSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'});assert.equal(head.status,0);assert.equal(head.stdout.trim(),headExpected);
const expected=['default',...(feature==='both'?['blas-backend','simd-backend']:feature==='default'?[]:[feature])].sort(),golden=['real','complex'].flatMap(k=>[32,64,128,256].map(n=>k+'_n'+n+'_seed42_large')).sort();
const j=JSON.parse(fs.readFileSync(file)),meta=j['rust-build-meta'];assert.equal(meta['target-directory'],target);assert.deepEqual(meta['base-output-directories'],['test-fast']);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),selected=[],ignored=[],excludedSuites=[],providers=new Map(),artifacts=new Map();
for(const s of Object.values(j['rust-suites'])){assert.equal(s['package-name'],'pfapack');assert.equal(s['package-id'],'path+file://'+root+'/crates/pfapack#0.0.0');assert.equal(s.cwd,root+'/crates/pfapack');
assert(s.testcases&&typeof s.testcases==='object'&&!Array.isArray(s.testcases),'invalid testcase inventory');
if(s.status==='skipped'){
 assert.equal(mode,'ignored','ordinary suite cannot be skipped');
 assert.notEqual(s['binary-name'],'golden_vs_julia','target golden suite cannot be skipped');
 assert.equal(Object.keys(s.testcases).length,0,'skipped suite must have empty testcases');
 excludedSuites.push({binary:s['binary-name'],status:s.status,reason:'non-target binary excluded by binary(golden_vs_julia)'});
 continue;
}
assert.equal(s.status,'listed');
for(const [name,t]of Object.entries(s.testcases)){assert.equal(t.kind,'test');const m=t['filter-match'];assert(['matches','mismatch'].includes(m?.status));
if(t.ignored){assert.equal(s['binary-name'],'golden_vs_julia');ignored.push(name);}
if(m.status==='mismatch'){if(mode==='ordinary'){assert(t.ignored,'ordinary unignored mismatch');assert.equal(m.reason,'ignored','ordinary ignored reason');}continue;}
assert.equal(t.ignored,mode==='ignored');if(mode==='ignored'){assert.equal(s['binary-name'],'golden_vs_julia');assert(golden.includes(name));}
const binary=fs.realpathSync(s['binary-path']);assert.equal(binary,s['binary-path']);assert.equal(path.dirname(binary),target+'/test-fast/deps');
const suffix=path.basename(binary).match(/-([a-f0-9]{16})$/)?.[1];assert(suffix);const kind=s.kind==='lib'?'test-lib-pfapack':s.kind==='test'?'test-integration-test-'+s['binary-name']:null;assert(kind);
const fingerprint=target+'/test-fast/.fingerprint/pfapack-'+suffix+'/'+kind+'.json',fp=JSON.parse(fs.readFileSync(fingerprint));assert.deepEqual(JSON.parse(fp.features).sort(),expected);assert.deepEqual(fp.rustflags,[]);
artifacts.set(binary,hash(binary));artifacts.set(fingerprint,hash(fingerprint));selected.push({name,binary,fingerprint});}}
assert.deepEqual(ignored.sort(),golden);assert(selected.length>0);
if(mode==='ignored')assert.deepEqual(selected.map(s=>s.name).sort(),golden);
else{for(const name of ['real_rank2_c_order_preserves_storage','real_factor_hand_dyadic_first_pivot_has_strict_margin'])assert(selected.some(s=>s.name==='ltl::rank2_real_regression::'+name),'missing real rank2 control '+name);for(const name of ['public_real_inverse_dispatches_direct_numerators','solver_consumes_direct_division_operands'])assert(selected.some(s=>s.name==='utu2::c_direct_solver_contract::'+name),'missing direct dispatch control '+name);assert(selected.some(s=>s.name==='ordinary_real_public_boundary_and_pivot_contracts'&&path.basename(s.binary).startsWith('issue176_ordinary_panel-')),'missing public boundary control');}
for(const binary of new Set(selected.map(s=>s.binary))){const r=spawnSync('/usr/bin/ldd',[binary],{encoding:'utf8',timeout:10000,maxBuffer:1048576});assert.equal(r.status,0);assert.equal(r.stderr,'');for(const raw of r.stdout.trimEnd().split('\n')){const line=raw.trim();if(/^linux-vdso\.so\.\d+ \(0x[0-9a-f]+\)$/.test(line))continue;const m=line.match(/^(?:\S+ => )?(\/\S+) \(0x[0-9a-f]+\)$/);assert(m,'unknown/unresolved provider '+line);const p=fs.realpathSync(m[1]);providers.set(p,hash(p));}}
for(const [p,h]of providers)artifacts.set(p,h);
selected.sort((a,b)=>(a.binary+a.name).localeCompare(b.binary+b.name));const result={head:head.stdout.trim(),feature,mode,features:expected,selected,ignored:golden,excludedSuites,artifacts:[...artifacts].sort(([a],[b])=>a.localeCompare(b)),providerScope:'ldd-resolved canonical selected ELF providers; not dlopen/full OS closure'};
fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 return result;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const [file,root,target,feature,mode,out]=process.argv.slice(2);validateList({file,root,target,feature,mode,out,headExpected:process.env.GITHUB_SHA});}
