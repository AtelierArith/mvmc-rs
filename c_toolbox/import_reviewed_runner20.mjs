// Optional mechanical import of independently generated artifacts; never Cargo.
// CASE cg|direct STORE [EXPLICIT_CONTAINER_STAGE]. No Rust-generated expectations.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
const [model, method, storeText, explicitStage] = process.argv.slice(2);
const cases = new Set('rbm_reference_cmp real cmp hubbard pairhop_real dh2_real dh2_cmp dh4_real dh4_cmp dh24_real dh24_cmp rbm_real rbm_cmp rbm_general_cmp rbm_dh24_cmp opt_real opt_cmp opt_dh24_rbm_cmp fsz pairhop_fsz dh2_fsz dh4_fsz dh24_fsz rbm_fsz opt_fsz interall'.split(' '));
if (!cases.has(model) || !['cg', 'direct'].includes(method) || !['0', '1'].includes(storeText)) throw Error('known case, solver and store required');
const container = '73c57e563c61';
const root = '/tmp/mvmc-review62b-runner20.Ctp8Dh';
const stage = explicitStage ?? `${root}-${model}${method === 'direct' ? `-direct${storeText}` : model === 'rbm_fsz' ? '-julia' : ''}`;
if (!stage.startsWith(`${root}-`) || !/^[A-Za-z0-9/_.-]+$/.test(stage)) throw Error('explicit frozen20 stage required');
const docker = (...args) => execFileSync('docker', ['exec', container, ...args], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const log = docker('tail', '-3', `${stage}.log`);
if (!log.includes('deterministic prefix runs') || !log.includes('Test Summary:') || /\bFail\b|\bError\b/.test(log)) throw Error(`acquisition not terminal-successful: ${stage}`);
const prefixes = method === 'cg' && model !== 'rbm_reference_cmp' ? [1, 2, 3, 20] : [20];
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const before = new Map();
const files = [];
for (const prefix of prefixes) {
    for (const kind of ['parameters', 'configs', 'energy', 'rng', ...(method === 'cg' ? ['SRinfo'] : [])]) {
        docker('test', '-f', `${stage}/step-${prefix}-${kind}.txt`);
    }
    const dir = `${stage}/step-${prefix}`;
    docker('test', '-f', `${dir}/provenance.txt`);
    docker('test', '-f', `${dir}/zvo_var.dat`);
    if (/^(dh|rbm_|opt_)/.test(model)) docker('test', '-f', `${stage}/step-${prefix}-zvo_out.dat`);
    const successful = docker('bash', '-lc', `if test -f ${dir}/c-window-input.txt; then echo yes; fi`).trim() === 'yes';
    if (successful) {
        const input = docker('cat', `${dir}/c-window-input.txt`);
        before.set(`step-${prefix}/c-window-input.txt`, digest(input));
        const header = input.split('\n')[0].trim().split(/\s+/).map(Number);
        if (header.length !== 17 || header[0] !== prefix || !Number.isInteger(header[1]) || header[1] <= 0) throw Error('complete effective-window input required');
        const fsz = model === 'fsz' || model.endsWith('_fsz') || model === 'interall';
        // InterAll uses a single General section, not the AP36 + P15 layout.
        const generalOnly = model === 'interall' || model === 'fsz';
        const layout = fsz ? (generalOnly ? ['1', '0', '0'] : ['1', '36', '15']) : ['0', '0', '0'];
        if (fsz && header[15] !== (generalOnly ? 22 : 66)) throw Error('unverified Slater layout; refuse C out-of-bounds aggregation');
        docker('/tmp/mvmc-history-layoutprobe.3WpZwp/probe', `${dir}/c-window-input.txt`, `${dir}/c-window`, ...layout);
    } else {
        docker('test', '-f', `${dir}/successful-history-not-final-output.txt`);
    }
}
// Only this new fixture namespace is writable; old reference lineages excluded.
const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const target = path.join(repo, 'tests/fixtures/reviewed_cg_62b', model === 'rbm_reference_cmp' ? 'canonical_general_rbm' : model, ...(method === 'direct' ? [`direct-store${storeText}`] : []));
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'mvmc-reviewed20-import-'));
execFileSync('docker', ['cp', `${container}:${stage}/.`, temporary]);
for (const prefix of prefixes) {
    for (const name of fs.readdirSync(temporary).filter(name => name.startsWith(`step-${prefix}-`))) {
        files.push([name, name]);
    }
    const subdir = `step-${prefix}`;
    for (const name of fs.readdirSync(path.join(temporary, subdir))) {
        if (name.startsWith('c-window_') && name.endsWith('.dat')) files.push([`${subdir}/${name}`, `${subdir}/${name.replace('c-window_', 'zqp_')}`]);
        else files.push([`${subdir}/${name}`, `${subdir}/${name}`]);
    }
}
const hashes = [];
for (const [source, destination] of files) {
    const bytes = fs.readFileSync(path.join(temporary, source));
    const sourceDigest = digest(bytes);
    if (before.has(source) && before.get(source) !== sourceDigest) throw Error('source window changed during import');
    const final = path.join(target, destination);
    fs.mkdirSync(path.dirname(final), { recursive: true });
    fs.copyFileSync(path.join(temporary, source), final);
    if (digest(fs.readFileSync(final)) !== sourceDigest) throw Error('artifact copy hash mismatch');
    hashes.push(`${sourceDigest}  ${destination}`);
}
// Generated integrity metadata, after every byte-identical artifact is copied.
fs.writeFileSync(path.join(target, 'archive.sha256'), hashes.sort().join('\n') + '\n');
fs.writeFileSync(path.join(target, 'acquisition-complete.txt'),
    `Reviewed PR54 62b0f97f076fb55c71c3ab0caa041a9adff94e04 ${method}/store${storeText} prefixes=${prefixes.join(',')}.\nSource ${stage}\nMapped-term snapshots are not C-declared dense packs; complete history/window uses dense pack.\nStandalone C aggregation; Julia sampler/SR; not full native C/MPI execution.\nArchived sparse inputs are synthetic API regressions, not C input acceptance goldens.\nAll imported artifacts SHA-verified; numerical budgets unchanged.\n`);
console.log(JSON.stringify({ model, method, store: storeText, prefixes, imported: files.length, target, source: stage, temporary }));
// Temporary source copy retained, recoverable and independently inspectable.
