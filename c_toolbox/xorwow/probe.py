"""Optional CUDA-header oracle; ordinary Rust tests consume only the committed CSV."""
import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("header", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
raw = args.header.read_bytes()
expected = "4a37c07a1d77c9b5c8c627a4720733cee6b4da4200a844e8a49291858bc26adf"
assert hashlib.sha256(raw).hexdigest() == expected, "Review changed NVIDIA header first"
src = raw.decode()
init = src.split("QUALIFIERS void _curand_init_inplace(", 1)[1]
init = init[init.index("    unsigned int s0 ="):init.index("    _skipahead_sequence_inplace")]
step = src.split("QUALIFIERS unsigned int curand(curandStateXORWOW_t *state)", 1)[1]
step = step[step.index("{")+1:step.index("}")]
harness = '''#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
typedef struct { unsigned int v[5], d; } curandStateXORWOW_t;
void init(unsigned long long seed, curandStateXORWOW_t *state) { INIT }
unsigned int draw(curandStateXORWOW_t *state) { STEP }
int main(void) {
 uint64_t seeds[]={0,1,11272,UINT64_MAX};
 puts("seed,index,word");
 for(int j=0;j<4;j++){ curandStateXORWOW_t state; init(seeds[j],&state);
  for(int i=0;i<32;i++) printf("%" PRIu64 ",%d,%u\\n",seeds[j],i,draw(&state)); }
}
'''.replace("INIT", init).replace("STEP", step)
with tempfile.TemporaryDirectory(prefix="xorwow-oracle-") as tmp:
    c = Path(tmp)/"probe.c"
    exe = Path(tmp)/"probe"
    c.write_text(harness)
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", str(c), "-o", str(exe)], check=True)
    args.output.write_bytes(subprocess.check_output([str(exe)]))
