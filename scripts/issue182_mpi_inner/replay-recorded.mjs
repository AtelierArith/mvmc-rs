// Optional read-only transport replay. Not invoked by Cargo or physics tests.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import assert from 'node:assert/strict';

const [input, output] = process.argv.slice(2);
assert.ok(input && output && path.isAbsolute(output) && !fs.existsSync(output));
const root = fs.realpathSync(input);
assert.ok(!output.startsWith(root + '/'), 'never write inside retained evidence');
// Explicit relative inventory: no absolute paths or traversal; all evidence
// bytes verified before any rank comparison. Captures are not new goldens.
for (const line of fs.readFileSync(root + '/SHA256SUMS', 'utf8').trim().split('\n')) {
    const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
    assert.ok(match);
    const relative = match[2];
    assert.ok(!path.isAbsolute(relative) && !relative.split('/').includes('..'));
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(root + '/' + relative)).digest('hex'), match[1], relative);
}
fs.mkdirSync(output);
for (const world of [2, 4]) for (const width of [1, 2]) for (let rank = 0; rank < world; rank++) {
    const files = [1, 2, 4].map(workers => `${root}/captures/world${world}/width${width}/workers${workers}/rank-${rank}.json`);
    const name = `world${world}-width${width}-rank${rank}`;
    const result = cp.spawnSync(process.execPath, [new URL('./compare-rank.mjs', import.meta.url).pathname, ...files, `${output}/${name}.json`], { encoding: 'utf8', timeout: 20000, maxBuffer: 1048576 });
    fs.writeFileSync(`${output}/${name}.stdout`, result.stdout ?? '', { flag: 'wx' });
    fs.writeFileSync(`${output}/${name}.stderr`, result.stderr ?? '', { flag: 'wx' });
    assert.equal(result.error, undefined);
    assert.equal(result.signal, null);
    assert.equal(result.status, 0, name);
    console.log(`${name} PASS`);
}
