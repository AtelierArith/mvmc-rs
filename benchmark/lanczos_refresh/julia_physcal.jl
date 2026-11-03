using MVMCOptimizers, LinearAlgebra
BLAS.set_num_threads(1)
input, output = abspath.(ARGS)
println("Julia=", VERSION, " threads=", Threads.nthreads(), " BLAS=", BLAS.get_config(), " BLAS_threads=", BLAS.get_num_threads())
flush(stdout)
for rep in 0:3
    dest = joinpath(output, "julia-$(rep)")
    @assert !ispath(dest)
    mkpath(dest)
    GC.gc()
    start = time_ns()
    result = MVMCOptimizers.run_phys_cal_from_namelist(joinpath(input, "namelist.def"); opt_para=joinpath(input, "zqp_opt.dat"), mode=:real, output_dir=dest)
    seconds = (time_ns() - start) / 1e9
    @assert result.status == 0
    files = filter(name -> startswith(name, "zvo_ls_qqqq_"), readdir(dest))
    @assert length(files) == 20 && all(name -> !isempty(strip(read(joinpath(dest, name), String))), files)
    println("BENCH rep=", rep, " seconds=", seconds)
    flush(stdout)
end
