# Production Julia runner; MPI ownership remains live across warmup/repetitions.
using MPI, MVMCOptimizers, LinearAlgebra, Profile, PfaPack, Libdl
BLAS.set_num_threads(1)
namelist, steps_s, warmups_s, reps_s, output, ranks_s = ARGS
steps, warmups, reps, ranks = parse.(Int, (steps_s, warmups_s, reps_s, ranks_s))
provided = MPI.Init_thread(MPI.THREAD_FUNNELED)
@assert provided >= MPI.THREAD_FUNNELED
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
try
    @assert MPI.Comm_size(comm) == ranks
    write(stdout, "WORLD $rank $ranks\nTHREADS $rank $(Threads.nthreads())\nBLAS_THREADS $rank $(BLAS.get_num_threads())\n")
    @assert BLAS.get_num_threads() == 1
    # Julia's ILP64 BLAS and PfaPack's native LP64 helper are separate libraries.
    native = Libdl.dlopen(PfaPack.libltl2inv)
    native_threads = ccall(Libdl.dlsym(native, :openblas_get_num_threads), Cint, ())
    @assert native_threads == 1
    write(stdout, "NATIVE_BLAS_THREADS $rank $native_threads\n")
    flush(stdout)
    function production(iteration; diagnostic=false)
        run_steps = diagnostic ? min(steps, 20) : steps
        MVMCOptimizers.run_para_opt_from_namelist(namelist;
            nsteps=run_steps, nsmp=run_steps, mode=:real,
            output_dir=joinpath(output, "run-$iteration"))
    end
    for iteration in 0:(warmups + reps - 1)
        MPI.Barrier(comm)
        start = time_ns()
        if iteration == 0
            # Compile in the ordinary full warmup. A separate short diagnostic
            # observes real warmed kernels, rather than mostly recording JIT.
            result = production(iteration)
            Profile.clear()
            Profile.init(n=10^7, delay=0.0002)
            Profile.@profile production("observe"; diagnostic=true)
            data, frames = Profile.retrieve()
            counts = Dict{Int,Int}()
            first_ip = 1
            # Julia 1.13 profile blocks end with four metadata fields and two zeros.
            for i in eachindex(data)
                if Profile.is_block_end(data, i)
                    tid = Int(data[i - Profile.META_OFFSET_THREADID])
                    ips = data[first_ip:(i - Profile.nmeta - 2)]
                    if any(ip -> any(frame -> occursin("MVMCOptimizers", string(frame.file)), get(frames, ip, [])), ips)
                        counts[tid] = get(counts, tid, 0) + 1
                    end
                    first_ip = i + 1
                end
            end
            write(stdout, "EXECUTION $rank $(sum(values(counts); init=0)) $(length(counts))\nEXECUTION_DETAIL $rank $counts\n")
            flush(stdout)
            mkpath(output)
            open(joinpath(output, "profile-rank-$rank.txt"), "w") do io
                Profile.print(io, data, frames; groupby=:thread, C=true)
            end
        else
            result = production(iteration)
        end
        MPI.Barrier(comm)
        seconds = MPI.Allreduce((time_ns() - start) / 1e9, max, comm)
        @assert result.status == 0
        if rank == 0
            @assert length(result.zvo_first_n) == steps
            @assert isfinite(result.final_energy_per_site)
            if iteration >= warmups
                println("BENCH $(iteration - warmups + 1) $seconds $(result.final_energy_per_site)")
            end
        end
    end
catch err
    showerror(stderr, err, catch_backtrace())
    MPI.Abort(comm, 1)
end
