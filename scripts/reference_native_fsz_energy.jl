# Optional mixed reference. Only local FSZ energy calls native C; the Julia
# runner, parameter initialization, RNG, sampler, SR solvers and output remain
# the original reference. Neither Cargo nor the Rust executable loads C here.
module NativeFSZEnergyReference
using Libdl, SHA
import MVMCOptimizers, MVMCExpertModeParsers

const CALL = Ref{Ptr{Cvoid}}(C_NULL)
const HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const PROVENANCE = Ref("")

function install!(directory)
    Threads.nthreads() == 1 || error("Native FSZ reference globals require Julia threads=1")
    suffix = Sys.isapple() ? ".dylib" : ".so"
    library = joinpath(directory, "libmvmc_fsz_reference" * suffix)
    HANDLE[] = Libdl.dlopen(library)
    CALL[] = Libdl.dlsym(HANDLE[], :mvmc_reference_fsz_energy)
    metadata = strip(read(joinpath(directory, "provenance.txt"), String))
    PROVENANCE[] = "# Mixed reference: original Julia 1.13.1 runner/RNG/sampling/SR with actual serial native C FSZ local energy including DH2/DH4; every saved projection counter verified by C MakeProjCnt; no RBM; not full C executable or MPI parity.\n# " * metadata * "\n# reference_native_fsz_energy.jl sha256=" * bytes2hex(sha256(read(@__FILE__))) * "\n"
    body = """
    function calculate_hamiltonian_fsz(
        ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
        ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_spn::Vector{Int},
        data::ExpertModeData, state::VMCOptimizationState;
        all_complex::Bool = true, c_timer::CTimer = CTIMER_DISABLED,
    )::ComplexF64
        return Main.NativeFSZEnergyReference.energy(ip, ele_idx, ele_cfg, ele_num,
            ele_proj_cnt, ele_spn, data, state; all_complex)
    end
    """
    Base.include_string(MVMCOptimizers, body, @__FILE__)
end

function energy(ip, idx, cfg, num, cnt, spins, data, state; all_complex)
    all_complex || error("This runner adapter is scoped to complex FSZ models")
    MVMCOptimizers.has_rbm_terms(data) && error("Native FSZ energy has no RBM extension")
    layout = MVMCExpertModeParsers.projection_layout(data)
    ns, nsize = data.modpara.nsite, length(idx)
    weights = data.qp_weights.qp_full_weight
    @assert length(spins) == nsize && length(cfg) == length(num) == 2*ns
    @assert length(cnt) == layout.n_proj == layout.n_gutzwiller + layout.n_jastrow + 6*layout.n_dh2 + 10*layout.n_dh4
    @assert length(data.doublon_holon_2site_indices) == layout.n_dh2
    @assert length(data.doublon_holon_4site_indices) == layout.n_dh4
    @assert length(data.doublon_holon_2site_params) == 6*layout.n_dh2
    @assert length(data.doublon_holon_4site_params) == 10*layout.n_dh4
    groups = (data.coulomb_intra_terms, data.coulomb_inter_terms, data.hund_terms,
              data.transfer_terms, data.pair_hop_terms, data.exchange_terms, data.inter_all_terms)
    dims = Cint[ns, nsize, length(weights), layout.n_gutzwiller, layout.n_jastrow, layout.n_dh2, layout.n_dh4,
                length.(groups)...]
    tables = Cint[]
    if layout.n_gutzwiller == 0
        append!(tables, zeros(Cint, ns))
    else
        @assert length(data.gutzwiller_idx) >= ns
        append!(tables, data.gutzwiller_idx[1:ns])
    end
    if layout.n_jastrow == 0
        append!(tables, fill(Cint(-1), ns*ns))
    else
        @assert size(data.jastrow_idx, 1) >= ns && size(data.jastrow_idx, 2) >= ns
        append!(tables, [data.jastrow_idx[i, j] for i in 1:ns for j in 1:ns])
    end
    for (indices, width) in ((data.doublon_holon_2site_indices, 2), (data.doublon_holon_4site_indices, 4))
        for item in indices
            @assert size(item.neighbors) == (ns, width)
            append!(tables, [item.neighbors[i, j] for i in 1:ns for j in 1:width])
        end
    end
    for term in groups[1]
        push!(tables, term.site)
    end
    for section in (2, 3)
        for term in groups[section]
            append!(tables, (term.site1, term.site2))
        end
    end
    for term in groups[4]
        append!(tables, (term.site1, term.spin1, term.site2, term.spin2))
    end
    for section in (5, 6)
        for term in groups[section]
            append!(tables, (term.site1, term.site2))
        end
    end
    for term in groups[7]
        append!(tables, (term.site0, term.spin0, term.site1, term.spin1,
                         term.site2, term.spin2, term.site3, term.spin3))
    end
    values = ComplexF64[t.value for t in data.gutzwiller_terms[1:layout.n_gutzwiller]]
    append!(values, [t.value for t in data.jastrow_terms[1:layout.n_jastrow]])
    append!(values, data.doublon_holon_2site_params)
    append!(values, data.doublon_holon_4site_params)
    append!(values, weights)
    for terms in groups
        append!(values, [t.value for t in terms])
    end
    configuration = Cint[idx; spins; cfg; num; cnt]
    mat = state.slater_matrix
    @assert length(mat.slater_elm) == length(weights)*(2*ns)^2
    @assert length(mat.pf_m) == length(weights)
    @assert length(mat.inv_m) >= length(weights)*nsize^2
    ip_arg, output = ComplexF64[ip], zeros(Float64, 2)
    ccall(CALL[], Cvoid,
          (Ptr{Cint}, Ptr{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64},
           Ptr{ComplexF64}, Ptr{Cint}, Ptr{ComplexF64}, Ptr{Float64}),
          dims, tables, values, mat.slater_elm, mat.pf_m, mat.inv_m, configuration, ip_arg, output)
    @assert configuration == Cint[idx; spins; cfg; num; cnt] "Native C mutated a caller configuration"
    return ComplexF64(output[1], output[2])
end
end
