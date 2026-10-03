"""Reviewed original-source anchors; retained execution claims are historical."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args()
revision='8bb1b9e8ae47b1512c00b321be05664ddcac0fd1'
opt='MVMCOptimizers.jl/src/'
mpi='test/mpi/'
specs={
 'S445':([(opt+'c_timer.jl',31,105),(opt+'c_timer.jl',174,182),(opt+'c_timer.jl',222,230)],'Enabled/disabled type dispatch, reset both arrays, env absent/literal0 disabled, other strings enabled; timer output labels. Worker merge source not covered by these timer anchors.', 'Bounded timer evidence retained; full diagnostic/caller/merge equivalence UNVERIFIED'),
 'S446':([(opt+'vmc_para_opt.jl',77,105),(opt+'vmc_para_opt.jl',305,367)],'kwargs callback/rng/output_dir/skip_sr/c_timer/ctx; absent RNG seeded, supplied RNG unchanged; skip_sr returns after first output/callback; SR status bcast then sync then final-window history then callback. Original data-only final writer differs C.', 'Current independent all-overload/output-history equivalence UNVERIFIED; writer repair owned Wegener'),
 'S447':([(opt+'vmc_phys_cal.jl',41,92),(opt+'vmc_phys_cal.jl',171,171),(opt+'vmc_phys_cal.jl',254,273)],'kwargs callback/rng/output_dir/ctx; validate before RNG, supplied RNG not reseeded; init draws then restore fixed params; sample output before callback. Rust final_rng field is diagnostic extension, not Julia returned field.', 'Historical twelve callback cases retain their source pin; every overload/MPI failure not proved'),
 'S448':([(mpi+'mpi_failure_modes.jl',35,45),(mpi+'run_mpi_smoke.jl',201,207)],'NSRCG2 reject before MPI.Init, two worker markers and empty output directory.', 'Original rejection contract source-reviewed; exact Rust before-MPI.Init replay UNVERIFIED'),
 'S449':([(mpi+'mpi_failure_modes.jl',47,58),(mpi+'run_mpi_smoke.jl',209,215)],'NSplitSize2+NSRCG1 reject before MPI.Init, two worker markers and empty outputs.', 'Exact original worker replay UNVERIFIED'),
 'S450':([(mpi+'mpi_failure_modes.jl',60,73),(mpi+'run_mpi_smoke.jl',217,223)],'NSplitSize2+two OptTrans sectors reject before MPI.Init; synthetic maps retained as original policy test, not valid C model.', 'Julia unsupported split policy is not proof of C input rejection; authority/runtime equivalence UNVERIFIED'),
 'S451':([(mpi+'mpi_failure_modes.jl',75,89),(mpi+'run_mpi_smoke.jl',225,231)],'FSZ NSplitSize2/NMPTrans2/NSPGaussLeg1 reject before MPI.Init; original minimal constructed data is policy test, not model.', 'Julia unsupported projection split policy, C input rejection not established; exact runtime UNVERIFIED'),
 'S452':([(opt+'parallel.jl',328,339),(mpi+'mpi_failure_modes.jl',91,109)],'Negative seed root time/bcast+group offset and abort/rethrow worker boundary. These original anchors DO NOT implement Rust asymmetric parse/output/callback failure protocol or injected clock.', 'RelatedOriginalSourceOnly; Rust safety-extension proof remains historical unpinned 81618, latest-source replay UNVERIFIED'),
 'S453':([(opt+'parallel.jl',328,339),(mpi+'mpi_failure_modes.jl',91,109)],'Same related seed/worker-abort anchors; original four-rank group protocol/communicator recovery claims absent from this worker.', 'RelatedOriginalSourceOnly; historical four-rank protocol not independent numerical parity; current proof UNVERIFIED'),
 'S454':([(mpi+'run_mpi_smoke.jl',108,129),(mpi+'run_mpi_smoke.jl',315,333),(mpi+'run_mpi_smoke.jl',345,355)],'Original grouped PhysCal real/cmp split1 world2 vs split2 world4 five-file self-consistency; root-only start/end count in world2. Not Rust CLI exactly14 coefficients contract.', 'Related bounded original stdout/self-consistency source; current Rust grouped CLI exact-discrete/independent numeric replay UNVERIFIED'),
 'S455':([(mpi+'run_mpi_smoke.jl',373,379),(mpi+'mpi_failure_modes.jl',91,109)],'Original launched MPI=0 must nonzero-exit and failure worker aborts if initialized. No original MPMD asymmetric parse/fixed/blocked-output test exists here.', 'NoExactOriginalCounterpart; Rust additional failure safety contract, historical LC5 evidence separate/unpinned'),
 'S456':([(mpi+'run_mpi_smoke.jl',359,387)],'Original rank0 stdout and MPI policy0/1 controls. No original valid-but-different steps/modes/quantity collective consensus test exists here.', 'NoExactOriginalCounterpart; Rust added control agreement contract, C/Julia parity not claimed')}
path=args.repo/'docs/reference/c-to-julia/verification/issue-184-scenarios.tsv'
with path.open(newline='') as stream: rows={r['id']:r for r in csv.DictReader(stream,delimiter='\t')}
out=[]
for identity,(anchors,conditions,classification) in specs.items():
    bound=[]
    for source,start,end in anchors:
        blob=subprocess.check_output(['git','-C',str(args.repo/'extern/Julia-mVMC'),'show',revision+':'+source])
        lines=blob.decode().splitlines()
        assert 1<=start<=end<=len(lines)
        bound.append(dict(source=source,start=start,end=end,sha256=hashlib.sha256(blob).hexdigest(),
                          source_excerpt='\n'.join(lines[start-1:end])))
    original=rows[identity]
    out.append(dict(id=identity,revision=revision,anchors=bound,conditions=conditions,
        proof_classification=classification,owner=original['owner'],
        original_result_retained=original['result'],command=original['command'],settings=original['settings'],
        actual_new_execution='NotRun',parent_review='Pending',
        proposed_julia_source='; '.join(a['source']+':'+str(a['start'])+'-'+str(a['end']) for a in bound)))
print(json.dumps(out,indent=2,sort_keys=True))
