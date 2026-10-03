# Observation only: hash the code/binaries actually loaded by reference workers.
using SHA
function issue179_reference_provenance(output, input, packages)
    mkpath(output)
    label = MPI.Initialized() ? string(MPI.Comm_rank(MPI.COMM_WORLD)) : get(ENV,"PMI_RANK","uninitialized")
    open(joinpath(output,"reference-provenance-$label.txt"),"w") do io
        println(io,"Julia=",VERSION," BLAS=",BLAS.get_config()," BLAS_threads=",BLAS.get_num_threads())
        println(io,"MPI=",MPI.Get_library_version())
        function digest(path)
            println(io,bytes2hex(SHA.sha256(read(path))),"  ",path)
        end
        for path in sort(readdir(dirname(input);join=true))
            isfile(path) && digest(path)
        end
        for package in packages
            root = pkgdir(package)
            println(io,"package=",nameof(package)," path=",root)
            for subtree in ("src","deps")
                base = joinpath(root,subtree)
                isdir(base) || continue
                for (dir, _, files) in walkdir(base)
                    for file in sort(files)
                        any(ext -> endswith(file,ext),(".jl",".c",".h",".so",".toml")) && digest(joinpath(dir,file))
                    end
                end
            end
        end
        manifest = joinpath(@__DIR__,"..","extern","Julia-mVMC","Manifest-v1.13.toml")
        digest(manifest)
        digest(joinpath(@__DIR__,"..","c_toolbox","ctest_direct_sr_capture.jl"))
        for script in ("mpi_issue179_reference_provenance.jl", "verify_mpi_issue179_state.jl", "verify_mpi_issue179_julia_launch.jl")
            digest(joinpath(@__DIR__,script))
        end
    end
end
