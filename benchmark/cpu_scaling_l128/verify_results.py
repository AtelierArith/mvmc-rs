"""Verify raw-archive checksums and original-file ledgers, without oracle runtimes."""
import argparse,hashlib,json,tarfile,tempfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('results',type=Path); a=p.parse_args()
manifest=json.loads((a.results/'archives.json').read_text())
for record in manifest:
    archive=a.results/record['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['sha256'], archive
    with tempfile.TemporaryDirectory(prefix='mvmc-586-verify-') as scratch:
        root=Path(scratch)
        with tarfile.open(archive,'r:gz') as tar: tar.extractall(root,filter='data')
        ledgers=list(root.glob('*/original-files.json')); assert len(ledgers)==1
        ledger=json.loads(ledgers[0].read_text()); cache={}
        assert len(ledger)==record['files']
        for name,expected in ledger.items():
            f=ledgers[0].parent/name; stat=f.stat(); key=(stat.st_dev,stat.st_ino)
            if key not in cache: cache[key]=hashlib.sha256(f.read_bytes()).hexdigest()
            assert stat.st_size==expected['size'] and cache[key]==expected['sha256'], f
    print(archive.name,'verified',flush=True)
