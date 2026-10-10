# Production PhysCal benchmark. MPI remains initialized across repetitions.
using MPI, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra, PfaPack, Libdl, Profile

length(ARGS) == 7 || error("usage: NAMELIST PARAMS GROUPS WARMUPS REPS OUTPUT RANKS")
namelist, parameters, groups_s, warmups_s, reps_s, output, ranks_s = ARGS
groups, warmups, reps, ranks = parse.(Int, (groups_s, warmups_s, reps_s, ranks_s))
groups > 0 && warmups >= 1 && reps > 0 && ranks > 0 || error("invalid benchmark counts")
BLAS.set_num_threads(1)
provided = MPI.Init_thread(MPI.THREAD_FUNNELED)
provided >= MPI.THREAD_FUNNELED || error("MPI_THREAD_FUNNELED unavailable")
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

try
    MPI.Comm_size(comm) == ranks || error("MPI rank count mismatch")
    parsed = MVMCExpertModeParsers.parse_expert_mode_files(namelist)
    modpara = parsed.modpara
    modpara.n_data_qty_smp == groups || error("NDataQtySmp must equal GROUPS")
    modpara.nsplit_size == 1 || error("benchmark requires NSplitSize=1")
    BLAS.get_num_threads() == 1 || error("Julia BLAS must use one thread")
    native = Libdl.dlopen(PfaPack.libltl2inv)
    native_threads = ccall(Libdl.dlsym(native, :openblas_get_num_threads), Cint, ())
    native_threads == 1 || error("native BLAS must use one thread")
    write(stdout, "WORLD $rank $ranks\nTHREADS $rank $(Threads.nthreads())\nBLAS_THREADS $rank $(BLAS.get_num_threads())\nNATIVE_BLAS_THREADS $rank $native_threads\n")
    write(stdout, "SAMPLES $rank $(modpara.nvmc_sample) $(modpara.nvmc_sample * ranks)\nGROUPS $rank $groups\n")
    flush(stdout)

    function production(iteration)
        MVMCOptimizers.run_phys_cal_from_namelist(namelist;
            opt_para=parameters, mode=:real,
            output_dir=joinpath(output, "run-$iteration"))
    end

    for iteration in 0:(warmups + reps - 1)
        MPI.Barrier(comm)
        started = time_ns()
        result = production(iteration)
        MPI.Barrier(comm)
        seconds = MPI.Allreduce((time_ns() - started) / 1e9, max, comm)
        result.status == 0 || error("PhysCal failed")
        result.n_para_consumed > 0 || error("no fixed parameters loaded")
        if rank == 0
            for group in 0:(groups - 1)
                index = lpad(string(modpara.n_data_idx_start + group), 3, '0')
                path = joinpath(result.output_dir, "$(modpara.c_data_file_head)_out_$index.dat")
                rows = readlines(path)
                length(rows) == 1 || error("expected one energy row per PhysCal group: $path")
                columns = split(only(rows))
                length(columns) == 6 || error("invalid PhysCal energy output: $path")
                isfinite(parse(Float64, first(columns))) || error("nonfinite PhysCal energy: $path")
            end
        end
        if iteration == 0
            # Observe warmed production kernels in a separate, untimed run.
            Profile.clear()
            Profile.init(n=10^7, delay=0.0002)
            Profile.@profile production("observe")
            data, frames = Profile.retrieve()
            counts = Dict{Int,Int}()
            first_ip = 1
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
        end
        if rank == 0 && iteration >= warmups
            println("BENCH $(iteration - warmups + 1) $seconds $groups")
            flush(stdout)
        end
    end
    MPI.Finalize()
catch err
    showerror(stderr, err, catch_backtrace())
    flush(stderr)
    MPI.Abort(comm, 1)
end
