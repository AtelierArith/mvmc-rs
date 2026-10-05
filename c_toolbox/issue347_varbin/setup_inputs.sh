#!/bin/bash
# Prepare /work/opt (NVMCCalMode=0) and /work/phys (NVMCCalMode=1) for run_in_container.sh.
# Usage: S=<scratch dir mounted as /work> M=<physcal_181 heisenberg_chain_real fixture dir> bash setup_inputs.sh
set -e
S=${S:?scratch directory}
M=${M:?tests/fixtures/physcal_181/two-samples/heisenberg_chain_real}
for d in opt phys; do
  mkdir -p "$S/$d"
  cp "$M"/inputs/* "$S/$d/"
done
cp "$M/zqp_opt.dat" "$S/phys/"
sed -i 's/NVMCCalMode    1/NVMCCalMode    0/; s/NSROptItrStep  1000/NSROptItrStep  4/; s/NSROptItrSmp   100/NSROptItrSmp   2/; s/NVMCSample     100/NVMCSample     20/; s/NDataIdxStart 7/NDataIdxStart 1/' "$S/opt/modpara.def"
# The Rust optimizer rejects TwoBodyGEx and C ignores Green functions in mode 0.
grep -v "OneBodyG\|TwoBodyG" "$S/opt/namelist.def" > "$S/opt/n2"
mv "$S/opt/n2" "$S/opt/namelist.def"
rm -f "$S/opt/greenone.def" "$S/opt/greentwo.def" "$S/opt/greentwoex.def"
