# Optional Julia test harness for two explicitly different C kernels.
# This is a mixed reference, not an unmodified Julia or full-C executable oracle.
# Native C MakeRBMCnt expectations validate the translated summation below.
# Rust tests consume the generated fixtures and never run this script.
using SHA

function source_function(file, signature)
    source = read(file, String)
    a = first(findfirst(signature, source))
    b = last(findnext("\nend\n", source, a))
    return source[a:b]
end

const C_RBM_SHA = "e35af053ce06e127309b1636dd2776650b4f1b2abd1ea0eaca10cd5257c3d742"
const C_SLATER_SHA = "cba5ddb9b8f42894dfa808aa867b05b358caab17175fb39931caf4625037db70"
const C_KERNEL_PROVENANCE = "# Mixed reference: Julia 1.13.1 runner with C MakeRBMCnt grouping and C Slater coefficient retention; not full C executable parity.\n# rbm.c SHA-256 $C_RBM_SHA; native expectations: rbm/c_counters.txt.\n# slater.c SHA-256 $C_SLATER_SHA; UpdateSlaterElm reads every mapped coefficient.\n"

function install_c_kernel_order!()
    root = joinpath(@__DIR__, "..")
    @assert bytes2hex(sha256(read(joinpath(root, "extern/mVMC-1.3.0/src/mVMC/rbm.c")))) == C_RBM_SHA
    @assert bytes2hex(sha256(read(joinpath(root, "extern/mVMC-1.3.0/src/mVMC/slater.c")))) == C_SLATER_SHA
    julia_source = joinpath(root, "extern/Julia-mVMC/MVMCOptimizers.jl/src")
    body = source_function(joinpath(julia_source, "vmc_sampling.jl"), "function make_rbm_cnt(")
    needle = "rbm_cnt[idx] += term.value * xi"
    @assert length(findall(needle, body)) == 3
    body = replace(body, "    # Coupling between physical and hidden layers." =>
        "    coupling = zeros(ComplexF64, length(rbm_cnt))\n    # Coupling between physical and hidden layers.")
    body = replace(body, needle => "coupling[idx] += term.value * xi")
    body = replace(body, "    return rbm_cnt" =>
        "    for idx in hidden_offset+1:length(rbm_cnt)\n        rbm_cnt[idx] += coupling[idx]\n    end\n    return rbm_cnt")
    Base.include_string(MVMCOptimizers, body)

    body = source_function(joinpath(julia_source, "slater_update.jl"), "function build_orbital_idx_sgn_matrices(")
    @assert occursin(" && abs(term.value) > 1e-14", body)
    body = replace(body, " && abs(term.value) > 1e-14" => "")
    Base.include_string(MVMCOptimizers, body)
    Base.invokelatest(verify_c_counter_translation!)
end

function c_kernel_input_order!(data)
    # Complete canonical tables use C's spin-major physical arrays. Sparse
    # archived internal models retain their explicitly supplied row order.
    if CASE == "rbm_reference_cmp"
        sort!(data.general_rbm_phys_layer_terms; by=t -> (t.spin, t.site))
        sort!(data.general_rbm_phys_hidden_terms; by=t -> (t.spin, t.site1, t.site2))
    end
    return data
end

function verify_c_counter_translation!()
    fixture = joinpath(@__DIR__, "../tests/fixtures/rbm/c_counters.txt")
    lines = filter(line -> !startswith(line, "#"), readlines(fixture))
    fields = (:charge_rbm_phys_layer_terms, :spin_rbm_phys_layer_terms,
        :general_rbm_phys_layer_terms, :charge_rbm_hidden_layer_terms,
        :spin_rbm_hidden_layer_terms, :general_rbm_hidden_layer_terms,
        :charge_rbm_phys_hidden_terms, :spin_rbm_phys_hidden_terms,
        :general_rbm_phys_hidden_terms)
    coords = (1, 1, 2, 1, 1, 1, 2, 2, 3)
    at = 1
    checked = 0
    while at <= length(lines)
        header = split(lines[at]); at += 1
        @assert header[1] == "MODEL"
        name = header[2]
        nsite, hidden, cases = parse.(Int, header[3:5])
        widths = parse.(Int, header[6:end])
        payloads = split(lines[at], '~'); at += 1
        words = parse.(UInt64, split(lines[at]); base=16); at += 1
        values = reinterpret(ComplexF64, words)
        data = MVMCExpertModeParsers.ExpertModeData()
        data.modpara.nsite = nsite
        data.modpara.nneuron_charge = hidden
        data.modpara.nneuron_spin = hidden
        data.modpara.nneuron_general = hidden
        offset = 0
        for section in 1:9
            terms = getfield(data, fields[section])
            if payloads[section] != "-"
                count = section in (1, 2) ? nsite : section == 3 ? 2*nsite :
                    section in (4, 5, 6) ? hidden : section in (7, 8) ? nsite*hidden : 2*nsite*hidden
                rows = split(payloads[section], '|')[6:5+count]
                # C keeps the final assignment for duplicate coordinates.
                mappings = Dict{Tuple,Int}()
                for row in rows
                    v = parse.(Int, split(row))
                    mappings[Tuple(v[1:end-1])] = v[end]
                end
                order = sort!(collect(keys(mappings)); by=c -> section in (3, 9) ? (c[2], c[1], section == 9 ? c[3] : 0) : c)
                for c in order
                    idx = mappings[c]
                    push!(terms, eltype(terms)(c..., values[offset+idx+1], true, idx))
                end
                # Julia infers width from mappings; a skipped ghost row retains
                # C's unused declared slots without changing any spatial cell.
                push!(terms, eltype(terms)(fill(-1, coords[section])..., 0.0+0.0im, true, widths[section]-1))
            end
            offset += widths[section]
        end
        for _ in 1:cases
            occupation = parse(Int, split(lines[at])[1]); at += 1
            ele_num = [(occupation >> bit) & 1 for bit in 0:2*nsite-1]
            expected = parse.(UInt64, split(lines[at]); base=16); at += 1
            actual = MVMCOptimizers.make_rbm_cnt(ele_num, data)
            actual_values = collect(reinterpret(Float64,actual))
            expected_values = reinterpret.(Float64,expected)
            @assert length(actual_values) == length(expected) "C counter width $name"
            @assert all(isapprox(a,e;atol=1e-13,rtol=1e-13,nans=true) for (a,e) in zip(actual_values,expected_values)) "C MakeRBMCnt $name occupation=$occupation"
            at += 1 # Incremental-hop expectations are tested independently in Rust.
            checked += 1
        end
    end
    println("Validated mixed-reference MakeRBMCnt against $checked native C cases")
end
