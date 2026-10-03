module C_API

using CEnum: CEnum, @cenum

using Libdl: dlext

const libsfmt = joinpath(pkgdir(@__MODULE__), "deps", "sfmt", "libsfmt.$(dlext)")

function gen_rand32()
    ccall((:gen_rand32, libsfmt), UInt32, ())
end

function gen_rand64()
    ccall((:gen_rand64, libsfmt), UInt64, ())
end

function sfmt_dump_rand32(array, size)
    ccall((:sfmt_dump_rand32, libsfmt), Cvoid, (Ptr{UInt32}, Cint), array, size)
end

function fill_array32(array, size)
    ccall((:fill_array32, libsfmt), Cvoid, (Ptr{UInt32}, Cint), array, size)
end

function fill_array64(array, size)
    ccall((:fill_array64, libsfmt), Cvoid, (Ptr{UInt64}, Cint), array, size)
end

function init_gen_rand(seed)
    ccall((:init_gen_rand, libsfmt), Cvoid, (UInt32,), seed)
end

function init_by_array(init_key, key_length)
    ccall((:init_by_array, libsfmt), Cvoid, (Ptr{UInt32}, Cint), init_key, key_length)
end

function get_idstring()
    ccall((:get_idstring, libsfmt), Ptr{Cchar}, ())
end

function get_min_array_size32()
    ccall((:get_min_array_size32, libsfmt), Cint, ())
end

function get_min_array_size64()
    ccall((:get_min_array_size64, libsfmt), Cint, ())
end

"""
    to_real1(v)

generates a random number on [0,1]-real-interval
"""
function to_real1(v)
    ccall((:to_real1, libsfmt), Cdouble, (UInt32,), v)
end

"""
    genrand_real1()

generates a random number on [0,1]-real-interval
"""
function genrand_real1()
    ccall((:genrand_real1, libsfmt), Cdouble, ())
end

"""
    to_real2(v)

generates a random number on [0,1)-real-interval
"""
function to_real2(v)
    ccall((:to_real2, libsfmt), Cdouble, (UInt32,), v)
end

"""
    genrand_real2()

generates a random number on [0,1)-real-interval
"""
function genrand_real2()
    ccall((:genrand_real2, libsfmt), Cdouble, ())
end

"""
    to_real3(v)

generates a random number on (0,1)-real-interval
"""
function to_real3(v)
    ccall((:to_real3, libsfmt), Cdouble, (UInt32,), v)
end

"""
    genrand_real3()

generates a random number on (0,1)-real-interval
"""
function genrand_real3()
    ccall((:genrand_real3, libsfmt), Cdouble, ())
end

"""
    to_res53(v)

generates a random number on [0,1) with 53-bit resolution
"""
function to_res53(v)
    ccall((:to_res53, libsfmt), Cdouble, (UInt64,), v)
end

"""
    to_res53_mix(x, y)

generates a random number on [0,1) with 53-bit resolution from two 32 bit integers
"""
function to_res53_mix(x, y)
    ccall((:to_res53_mix, libsfmt), Cdouble, (UInt32, UInt32), x, y)
end

"""
    genrand_res53()

generates a random number on [0,1) with 53-bit resolution
"""
function genrand_res53()
    ccall((:genrand_res53, libsfmt), Cdouble, ())
end

"""
    genrand_res53_mix()

generates a random number on [0,1) with 53-bit resolution using 32bit integer.
"""
function genrand_res53_mix()
    ccall((:genrand_res53_mix, libsfmt), Cdouble, ())
end

# Skipping MacroDefinition: ALWAYSINLINE __attribute__ ( ( always_inline ) )

# Skipping MacroDefinition: PRE_ALWAYS inline

const MEXP = 19937

const N = MEXP ÷ 128 + 1

const N32 = N * 4

const N64 = N * 2

const POS1 = 122

const SL1 = 18

const SL2 = 1

const SR1 = 11

const SR2 = 1

const MSK1 = Cuint(0xdfffffef)

const MSK2 = Cuint(0xddfecb7f)

const MSK3 = Cuint(0xbffaffff)

const MSK4 = Cuint(0xbffffff6)

const PARITY1 = Cuint(0x00000001)

const PARITY2 = Cuint(0x00000000)

const PARITY3 = Cuint(0x00000000)

const PARITY4 = Cuint(0x13c9e684)

# Skipping MacroDefinition: ALTI_SL1 ( vector unsigned int ) ( SL1 , SL1 , SL1 , SL1 )

# Skipping MacroDefinition: ALTI_SR1 ( vector unsigned int ) ( SR1 , SR1 , SR1 , SR1 )

# Skipping MacroDefinition: ALTI_MSK ( vector unsigned int ) ( MSK1 , MSK2 , MSK3 , MSK4 )

# Skipping MacroDefinition: ALTI_MSK64 ( vector unsigned int ) ( MSK2 , MSK1 , MSK4 , MSK3 )

# Skipping MacroDefinition: ALTI_SL2_PERM ( vector unsigned char ) ( 1 , 2 , 3 , 23 , 5 , 6 , 7 , 0 , 9 , 10 , 11 , 4 , 13 , 14 , 15 , 8 )

# Skipping MacroDefinition: ALTI_SL2_PERM64 ( vector unsigned char ) ( 1 , 2 , 3 , 4 , 5 , 6 , 7 , 31 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 0 )

# Skipping MacroDefinition: ALTI_SR2_PERM ( vector unsigned char ) ( 7 , 0 , 1 , 2 , 11 , 4 , 5 , 6 , 15 , 8 , 9 , 10 , 17 , 12 , 13 , 14 )

# Skipping MacroDefinition: ALTI_SR2_PERM64 ( vector unsigned char ) ( 15 , 0 , 1 , 2 , 3 , 4 , 5 , 6 , 17 , 8 , 9 , 10 , 11 , 12 , 13 , 14 )

const IDSTR = "SFMT-19937:122-18-1-11-1:dfffffef-ddfecb7f-bffaffff-bffffff6"

end # module
