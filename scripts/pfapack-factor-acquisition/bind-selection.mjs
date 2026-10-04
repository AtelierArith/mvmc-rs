// Narrow reuse of the existing issue176 CI list/fingerprint/provider association.
import assert from 'node:assert/strict';
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
const [file,root,out]=process.argv.slice(2);assert.equal(process.argv.length,5);
const exact='ltl::factor_acquisition::independent_c59_four_factor_inverse_stage_stream';
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const head=spawnSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8',maxBuffer:1048576});assert.equal(head.status,0);assert.equal(head.stdout.trim(),process.env.GITHUB_SHA);
const j=JSON.parse(fs.readFileSync(file)),meta=j['rust-build-meta'];assert.deepEqual(meta['base-output-directories'],['ci']);
const target=fs.realpathSync(meta['target-directory']);assert.equal(target,meta['target-directory']);
const selected=[];
for(const s of Object.values(j['rust-suites'])){
 assert.equal(s['package-name'],'pfapack');assert.equal(s['package-id'],'path+file://'+root+'/crates/pfapack#0.0.0');assert.equal(s.cwd,root+'/crates/pfapack');
 assert.equal(s.status,'listed');assert.equal(s.kind,'lib');assert.equal(s['binary-name'],'pfapack');
 for(const [name,t]of Object.entries(s.testcases)){
  assert.equal(t.kind,'test');const m=t['filter-match'];assert(['matches','mismatch'].includes(m?.status));
  if(m.status!=='matches')continue;
  assert.equal(name,exact);assert.equal(t.ignored,true);
  const binary=fs.realpathSync(s['binary-path']);assert.equal(binary,s['binary-path']);assert.equal(path.dirname(binary),target+'/ci/deps');
  const suffix=path.basename(binary).match(/^pfapack-([a-f0-9]{16})$/)?.[1];assert(suffix);
  const fingerprint=target+'/ci/.fingerprint/pfapack-'+suffix+'/test-lib-pfapack.json',fp=JSON.parse(fs.readFileSync(fingerprint));
  assert.deepEqual(JSON.parse(fp.features).sort(),['default']);assert.deepEqual(fp.rustflags,[]);
  selected.push({name,binary,fingerprint,features:['default']});
 }
}
assert.equal(selected.length,1,'one exact ignored lib producer');
const artifacts=new Map();for(const p of [selected[0].binary,selected[0].fingerprint])artifacts.set(p,hash(p));
const node=fs.realpathSync(process.execPath);artifacts.set(node,hash(node));
for(const executable of [selected[0].binary,node]){
const ldd=spawnSync('/usr/bin/ldd',[executable],{encoding:'utf8',timeout:10000,maxBuffer:1048576});assert.equal(ldd.status,0);assert.equal(ldd.stderr,'');
for(const raw of ldd.stdout.trimEnd().split('\n')){
 const line=raw.trim();if(/^linux-vdso\.so\.\d+ \(0x[0-9a-f]+\)$/.test(line))continue;
 const m=line.match(/^(?:\S+ => )?(\/\S+) \(0x[0-9a-f]+\)$/);assert(m,'unknown/unresolved provider '+line);const p=fs.realpathSync(m[1]);artifacts.set(p,hash(p));
}}
fs.writeFileSync(out,JSON.stringify({head:head.stdout.trim(),profile:'ci',scope:'selected actual test ELF/fingerprint and actual Node plus ldd canonical providers; not full compiler/dlopen closure',selected:selected[0],artifacts:[...artifacts]},null,2)+'\n',{flag:'wx'});
fs.writeFileSync(out+'.sha256',[...artifacts].map(([p,h])=>`${h}  ${p}`).join('\n')+'\n',{flag:'wx'});
