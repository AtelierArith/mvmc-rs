module SFMT

import Random

include("C_API.jl")
using .C_API: init_gen_rand, genrand_real2, gen_rand32
using .C_API: sfmt_dump_rand32 as c_sfmt_dump_rand32

export init_gen_rand, genrand_real2, gen_rand32
export sfmt_dump_rand32
export SFMT19937RNG

const IS_INITIALIZED = Ref{Bool}(false)

struct SFMT19937RNG <: Random.AbstractRNG end

function Random.seed!(rng::SFMT19937RNG, seed)
	init_gen_rand(UInt32(seed))
	IS_INITIALIZED[] = true
	return rng
end

function Random.rand(rng::SFMT19937RNG)
	IS_INITIALIZED[] || throw(ArgumentError("SFMT19937 not initialized. Please call Random.seed!(rng, seed) before generating random numbers."))
	genrand_real2()
end

function Random.rand(rng::SFMT19937RNG, ::Type{UInt32})
	IS_INITIALIZED[] || throw(ArgumentError("SFMT19937 not initialized. Please call Random.seed!(rng, seed) before generating random numbers."))
	gen_rand32()
end

# Support for range arguments: rand(rng, 0:n-1)
# NOTE: This uses simple modulo (like C implementation's gen_rand32()%N)
# instead of rejection sampling for exact compatibility with mVMC C implementation.
# This may have slight bias for ranges that don't evenly divide 2^32,
# but matches C implementation behavior exactly.
function Random.rand(rng::SFMT19937RNG, r::AbstractUnitRange{<:Integer})
	IS_INITIALIZED[] || throw(ArgumentError("SFMT19937 not initialized. Please call Random.seed!(rng, seed) before generating random numbers."))
	# Use gen_rand32() to generate random integer in range
	# Convert range to length and offset
	len = length(r)
	if len == 0
		throw(ArgumentError("Cannot sample from empty range"))
	end
	offset = first(r)
	# Generate random value in [0, len) using simple modulo (like C implementation)
	# C implementation: gen_rand32()%Ne
	val = gen_rand32()
	return offset + Int(val % len)
end

# Support for Int type: rand(rng, Int)
function Random.rand(rng::SFMT19937RNG, ::Type{Int})
	IS_INITIALIZED[] || throw(ArgumentError("SFMT19937 not initialized. Please call Random.seed!(rng, seed) before generating random numbers."))
	# Generate random Int using two UInt32 values
	# For simplicity, use one UInt32 and sign-extend if needed
	val = gen_rand32()
	# Convert to signed Int (assuming 64-bit Int)
	Int(val)
end

function sfmt_dump_rand32(n::Integer)
	if n <= 0
		return UInt32[]
	end
	buf = Vector{UInt32}(undef, n)
	c_sfmt_dump_rand32(buf, n)
	return buf
end

end # module SFMT
