import hashlib, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
for path in source.iterdir():
    if path.name.endswith('.stdout'):
        with path.open('rb') as src, (destination / path.name).open('w') as out:
            for line in src:
                if line.startswith(b'D|'):
                    walker = line.split(b'|', 2)[1].decode('ascii')
                    out.write(f'H|{walker}|{hashlib.sha256(line.rstrip(bytes([10]))).hexdigest()}\n')
                elif line.startswith(b'N|'):
                    out.write(line.decode('ascii'))
    elif path.name.endswith(('.time', '.stderr')) or path.name in ('config.env', 'input-sha256.txt'):
        (destination / path.name).write_bytes(path.read_bytes())
