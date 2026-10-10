# Optional diagnostic: install a typed real-QP budget selector before all warmups.
# Both variants use the production API and identical arithmetic; no timed eval/JIT.
using MPI, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra, Libdl, PfaPack, SHA
const O=MVMCOptimizers
BLAS.set_num_threads(1)
namelist, parameters, groups_s, reps_s, output, ranks_s = ARGS
groups,reps,ranks=parse.(Int,(groups_s,reps_s,ranks_s))
@assert groups==100 && reps==3
modpara=MVMCExpertModeParsers.parse_expert_mode_files(namelist).modpara
@assert modpara.n_data_qty_smp==groups && modpara.nsplit_size==1
@assert modpara.nvmc_sample*ranks==320
provided=MPI.Init_thread(MPI.THREAD_FUNNELED)
@assert provided>=MPI.THREAD_FUNNELED && MPI.Is_thread_main()
comm=MPI.COMM_WORLD
rank=MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm)==ranks
@assert BLAS.get_num_threads()==1
native=Libdl.dlopen(PfaPack.libltl2inv)
@assert ccall(Libdl.dlsym(native,:openblas_get_num_threads),Cint,())==1
@eval O const real_qp_budget_candidate=Ref{Bool}(false)
kernel_source=read(joinpath(dirname(pathof(O)),"calculate_m_all.jl"),String)
method_start=first(findlast("function calculate_m_all_real!(",kernel_source))
method=strip(kernel_source[method_start:end])
@assert occursin("threaded_workspace::ThreadedPfaPackWorkspace",method)
@assert count("qp_num < 2",method)==1
Core.eval(O, Meta.parse(replace(method,"qp_num < 2"=>
    "qp_num < (real_qp_budget_candidate[] ? 2 : n_threads)")))
if rank==0
    mkpath(output)
    write(joinpath(output,"gate-override.txt"),"Typed Ref selector installed before warmups; baseline requires QP count >= thread count, candidate requires at least two QPs and sufficient partial-pool matrix work; full-pool paths unchanged.\n")
end
println("WORLD $rank $ranks\nTHREADS $rank $(Threads.nthreads())\nBLAS_THREADS $rank $(BLAS.get_num_threads())\nNATIVE_BLAS_THREADS $rank 1")
flush(stdout)
function production(label)
    @assert MPI.Is_thread_main()
    O.run_phys_cal_from_namelist(namelist;opt_para=parameters,mode=:real,
                                output_dir=joinpath(output,label))
end
function validate_output(result)
    for group in 0:(groups-1)
        index=lpad(string(modpara.n_data_idx_start+group),3,'0')
        path=joinpath(result.output_dir,"$(modpara.c_data_file_head)_out_$index.dat")
        rows=readlines(path)
        @assert length(rows)==1
        values=parse.(Float64,split(only(rows)))
        @assert length(values)==6 && all(isfinite,values)
    end
end
try
    for candidate in (false,true)
        O.real_qp_budget_candidate[]=candidate
        MPI.Barrier(comm)
        result=production(candidate ? "warm-candidate" : "warm-baseline")
        @assert result.status==0 && result.n_para_consumed>0
        rank==0 && validate_output(result)
        MPI.Barrier(comm)
    end
    for rep in 1:reps
        for candidate in (isodd(rep) ? (false,true) : (true,false))
            O.real_qp_budget_candidate[]=candidate
            name=candidate ? "candidate" : "baseline"
            MPI.Barrier(comm)
            start=time_ns()
            result=production("$name-$rep")
            MPI.Barrier(comm)
            elapsed=(time_ns()-start)/1e9
            seconds=MPI.Allreduce(elapsed,max,comm)
            @assert result.status==0 && result.n_para_consumed>0
            if rank==0
                validate_output(result)
                println("PAIRED $name $rep $seconds $groups")
                flush(stdout)
            end
        end
    end
catch err
    showerror(stderr,err,catch_backtrace())
    MPI.Abort(comm,1)
finally
    MPI.Finalize()
end
