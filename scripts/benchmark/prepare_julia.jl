# Prepare only the selected pinned workspace. Never resolve/update its lock.
using Pkg, SHA
lock = joinpath(dirname(Base.active_project()), "Manifest-v1.13.toml")
@assert isfile(lock) "Julia 1.13.1 manifest is required"
before = sha256(read(lock))
Pkg.instantiate()
Pkg.build("PfaPack")
Pkg.build("SFMT")
@assert sha256(read(lock)) == before "Pinned manifest changed during preparation"
