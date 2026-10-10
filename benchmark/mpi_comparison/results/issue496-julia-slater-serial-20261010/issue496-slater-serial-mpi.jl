using MVMCOptimizers, MPI, LinearAlgebra
const diag_serial_checked=Ref{Bool}(get(ENV,"MVMC_SERIAL_CHECKED","0")=="1")
let source=read(joinpath(pkgdir(MVMCOptimizers),"src","slater_update.jl"),String)
    first= findfirst("function update_slater_elm_fcmp!(",source)
    last=findnext("\nend",source,first.start)
    original=source[first.start:last.stop]
    Core.eval(MVMCOptimizers,Meta.parse(replace(original,"function update_slater_elm_fcmp!"=>"function _issue496_original_slater!";count=1)))
end
candidate=read("/home/vscode/.cache/mvmc/issue496-slater-serial-checked.jl",String)
candidate=replace(candidate,"return update_slater_elm_fcmp!("=>"return _issue496_original_slater!(")
Base.include_string(Main,candidate,"issue496-slater-serial-private.jl")
Core.eval(MVMCOptimizers,quote
    function update_slater_elm_fcmp!(data::ExpertModeData,state::VMCOptimizationState)
        if Main.diag_serial_checked[]
            return _issue496_update_slater_serial_checked!(data,state)
        end
        return _issue496_original_slater!(data,state)
    end
end)
println("SERIAL_CHECKED_INSTALLED ",diag_serial_checked[])
if haskey(ENV,"MVMC_SERIAL_AUDIT")
    include("/home/vscode/.cache/mvmc/issue496-julia-state-audit.jl")
else
    BLAS.set_num_threads(1)
    MPI.Init_thread(MPI.THREAD_FUNNELED)
    input,out,steps_s=ARGS; steps=parse(Int,steps_s)
    rank=MPI.Comm_rank(MPI.COMM_WORLD)
    variants=(("baseline",false),("checked",true))
    function production(label)
        MVMCOptimizers.run_para_opt_from_namelist(input;nsteps=steps,nsmp=steps,mode=:real,output_dir=joinpath(out,label))
    end
    try
        for (label,flag) in variants
            diag_serial_checked[]=flag
            MPI.Barrier(MPI.COMM_WORLD)
            result=production("warm-$label"); @assert result.status==0
        end
        for rep in 1:3
            for (label,flag) in (isodd(rep) ? variants : reverse(variants))
                diag_serial_checked[]=flag; MPI.Barrier(MPI.COMM_WORLD)
                begin_ns=time_ns(); result=production("$label-$rep")
                MPI.Barrier(MPI.COMM_WORLD)
                elapsed=MPI.Allreduce((time_ns()-begin_ns)/1e9,max,MPI.COMM_WORLD)
                @assert result.status==0
                rank==0 && println("BENCH $label $rep $elapsed ",result.final_energy_per_site)
            end
        end
    finally
        MPI.Finalize()
    end
end
