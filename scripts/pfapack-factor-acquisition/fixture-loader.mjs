import assert from 'node:assert/strict';
import fs from 'node:fs';import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
export const hashes=new Map([
 [32,'c597cecb37b13b1023e604077d571d9f52d6dbf3b493efc0f1a0e8db9c38ff4c'],
 [64,'be137a1da53ba84423b7d27db14cccc58c901a92a51519835bd9639405259fc1'],
 [128,'74807b73091db8bffa6bb554cc3de3d0f3aafee28f6b8058dd654b1092b08607'],
 [256,'554339f1a479130add25e68a87b14ce707d2651ade9beb55e4c6d68c7e7fdd2f']]);
const directory=new URL('../../tests/fixtures/pfapack/c59_factor_admission/',import.meta.url);
export function fixturePath(n){assert(hashes.has(n),'closed fixture size');return fileURLToPath(new URL(`real_n${n}_seed42_c59.json`,directory));}
export function loadFixture(n){
 const p=fixturePath(n);assert(fs.lstatSync(p).isFile(),'fixture regular nonsymlink');
 const bytes=fs.readFileSync(p);
 assert.equal(createHash('sha256').update(bytes).digest('hex'),hashes.get(n),'fixture SHA before parse');
 const f=JSON.parse(bytes);assert.equal(f.n,n);assert.equal(f.kind,'retained-independent-c-kernel');assert.equal(f.layout,'column-major');assert.equal(f.inputWords.length,n*n);
 const text=fs.readFileSync(new URL(`input_n${n}.words`,directory),'utf8');
 assert.equal(text,`# C59 independent input words; column-major; n=${n}\n`+f.inputWords.join('\n')+'\n','derived input bytes match independent C data');
 return f;
}
