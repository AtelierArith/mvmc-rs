"""Add an optimization boundary call to the existing nonconsuming state probe."""

from pathlib import Path
import sys

p = Path(sys.argv[1]) / "src/mVMC/vmcmain.c"
s = p.read_text()
prototype = (
    "int VMCParaOpt(MPI_Comm comm_parent, MPI_Comm comm_child1, MPI_Comm comm_child2);"
)
hook = "    StartTimer(5);"
assert s.count(prototype) == 1 and s.count(hook) == 1
s = s.replace(prototype, prototype + "\nstatic void mvmc_dump_state(int ismp);")
s = s.replace(
    hook,
    "    if(rank==0) mvmc_dump_state(step); /* #492 nonconsuming capture */\n" + hook,
)
p.write_text(s)
p = Path(sys.argv[1]) / "src/mVMC/stcopt_dposv.c"
s = p.read_text()
hook = "  stcOptInit(S, g, nSmat, smatToParaIdx);"
assert s.count(hook) == 1
s = s.replace(
    hook,
    hook
    + """
  /* #492 read-only original solver inputs, column-major S. */
  for(int k=0;k<nSmat*nSmat;k++) fprintf(stderr,"DEBUG: direct_S[%d]=%.17e\\n",k,S[k]);
  for(int k=0;k<nSmat;k++) {
    fprintf(stderr,"DEBUG: direct_g[%d]=%.17e\\n",k,g[k]);
    fprintf(stderr,"DEBUG: direct_map[%d]=%d\\n",k,smatToParaIdx[k]);
  }
""",
)
p.write_text(s)
