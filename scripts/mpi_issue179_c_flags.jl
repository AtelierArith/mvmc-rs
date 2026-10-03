# Optional independent definition-record contract, not a Rust/Julia state repair.
using SHA
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: <input-directory> <contract-file>")
repo = normpath(joinpath(@__DIR__,".."))
generator = joinpath(repo,"scripts","generate_physcal_181.jl")
source = read(generator,String)
start = findfirst("function c_definition_flags(inputs)",source)
start === nothing && error("C flag enumeration boundary missing")
stop = findnext("function checkpoint(",source,last(start)+1)
stop === nothing && error("C flag enumeration end missing")
Base.include_string(Main,source[first(start):prevind(source,first(stop))],"mpi179_C_definition_flags")
flags, written = c_definition_flags(ARGS[1])
length(flags) == length(written) > 0 || error("empty/incomplete C flags contract")
reader = joinpath(repo,"extern","mVMC-1.3.0","src","mVMC","readdef.c")
open(ARGS[2],"w") do io
    println(io,"flags ",join(flags," "))
    println(io,"written ",join(written," "))
    println(io,"C_source_sha256 ",bytes2hex(sha256(read(reader))))
    println(io,"enumerator_sha256 ",bytes2hex(sha256(read(generator))))
    println(io,"contract_sha256 ",bytes2hex(sha256(read(@__FILE__))))
end
# Unwritten placeholder zeros are NOT native allocation expectations. Compare
# only written cells. GetInfoOptOrbitalParalell writes imaginary slots explicitly;
# the reused enumeration includes that family exception and native OptTrans offsets.
