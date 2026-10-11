using Pkg, Libdl
@assert VERSION == v"1.13.1"
Pkg.instantiate()
using MPIPreferences
MPIPreferences.use_system_binary(; library_names=[ARGS[1]], mpiexec=ARGS[2])
root = dirname(Base.active_project())
for (pkg, path) in (("SFMT", joinpath(root, "SFMT.jl", "deps", "sfmt", "libsfmt." * Libdl.dlext)),
                    ("PfaPack", joinpath(root, "PfaPack.jl", "deps", "libltl2inv." * Libdl.dlext)))
    isfile(path) || Pkg.build(pkg)
end
