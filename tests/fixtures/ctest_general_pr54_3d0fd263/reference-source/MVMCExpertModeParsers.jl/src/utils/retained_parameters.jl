# C family order. A width change resets only that family's Julia API storage;
# accepted definition-file layouts are immutable throughout a C run.
const RETAINED_RBM_FAMILIES = (
    :charge_rbm_phys_layer_terms, :spin_rbm_phys_layer_terms,
    :general_rbm_phys_layer_terms, :charge_rbm_hidden_layer_terms,
    :spin_rbm_hidden_layer_terms, :general_rbm_hidden_layer_terms,
    :charge_rbm_phys_hidden_terms, :spin_rbm_phys_hidden_terms,
    :general_rbm_phys_hidden_terms,
)

function declared_parameter_families(data::ExpertModeData)
    layout = projection_layout(data)
    rbm = map(RETAINED_RBM_FAMILIES) do name
        terms = getfield(data, name)
        name => (isempty(terms) ? 0 : maximum(t.idx for t in terms) + 1)
    end
    return (
        :gutzwiller => layout.n_gutzwiller, :jastrow => layout.n_jastrow,
        :dh2 => 6 * layout.n_dh2, :dh4 => 10 * layout.n_dh4,
        rbm..., :slater => count_orbital_parameters(data),
        :opttrans => count_opt_trans_parameters(data),
    )
end

function retained_family!(data::ExpertModeData, name::Symbol, width::Int)
    values = get(data.retained_parameters, name, nothing)
    if values === nothing || length(values) != width
        values = zeros(ComplexF64, width)
        data.retained_parameters[name] = values
    end
    return values
end

function retained_flat_parameters!(data::ExpertModeData)
    values = ComplexF64[]
    for (name, width) in declared_parameter_families(data)
        append!(values, retained_family!(data, name, width))
    end
    return values
end

function retained_parameter_slot!(data::ExpertModeData, index::Int)
    offset = 0
    for (name, width) in declared_parameter_families(data)
        if offset < index <= offset + width
            return retained_family!(data, name, width), index - offset
        end
        offset += width
    end
    return nothing
end

function retain_dense_prefix!(data::ExpertModeData, name::Symbol, source)
    width = only(last(p) for p in declared_parameter_families(data) if first(p) == name)
    target = retained_family!(data, name, width)
    for i in 1:min(width, length(source))
        target[i] = source[i]
    end
    return nothing
end

function retain_flat_parameters!(data::ExpertModeData, values)
    families = declared_parameter_families(data)
    length(values) == sum(last, families) || throw(ArgumentError("declared Para length mismatch"))
    offset = 0
    for (name, width) in families
        copyto!(retained_family!(data, name, width), 1, values, offset + 1, width)
        offset += width
    end
    return nothing
end

function gather_retained_parameters!(data::ExpertModeData)
    for (name, width) in declared_parameter_families(data)
        values = retained_family!(data, name, width)
        if name in RETAINED_RBM_FAMILIES || name == :slater
            terms = name == :slater ? data.orbital_terms : getfield(data, name)
            for term in terms
                0 <= term.idx < width && (values[term.idx + 1] = term.value)
            end
        else
            source = name == :gutzwiller ? data.gutzwiller_terms :
                name == :jastrow ? data.jastrow_terms :
                name == :dh2 ? data.doublon_holon_2site_params :
                name == :dh4 ? data.doublon_holon_4site_params : data.opt_trans
            for i in 1:min(width, length(source))
                values[i] = name in (:gutzwiller, :jastrow) ? source[i].value : source[i]
            end
        end
    end
    return data
end

function scatter_retained_parameters!(data::ExpertModeData)
    for (name, width) in declared_parameter_families(data)
        values = retained_family!(data, name, width)
        if name in RETAINED_RBM_FAMILIES || name == :slater
            terms = name == :slater ? data.orbital_terms : getfield(data, name)
            for term in terms
                0 <= term.idx < width && (term.value = values[term.idx + 1])
            end
        else
            target = name == :gutzwiller ? data.gutzwiller_terms :
                name == :jastrow ? data.jastrow_terms :
                name == :dh2 ? data.doublon_holon_2site_params :
                name == :dh4 ? data.doublon_holon_4site_params : data.opt_trans
            for i in 1:min(width, length(target))
                if name in (:gutzwiller, :jastrow)
                    target[i].value = values[i]
                else
                    target[i] = values[i]
                end
            end
        end
    end
    return data
end

const RETAINED_INPUT_FAMILIES = Dict(
    "InGutzwiller" => :gutzwiller, "InJastrow" => :jastrow,
    "InOrbital" => :slater, "InOrbitalAntiParallel" => :slater,
    "InOrbitalGeneral" => :slater,
    ("In" * replace(String(name),
        "charge_rbm" => "ChargeRBM", "spin_rbm" => "SpinRBM",
        "general_rbm" => "GeneralRBM", "phys_layer_terms" => "PhysLayer",
        "hidden_layer_terms" => "HiddenLayer", "phys_hidden_terms" => "PhysHidden") => name
        for name in RETAINED_RBM_FAMILIES)...,
)

function retain_input_overlay!(data::ExpertModeData, file_type, params)
    family = get(RETAINED_INPUT_FAMILIES, file_type, nothing)
    offset = 0
    if file_type == "InOrbitalParallel"
        family = :slater
        offset = _orbital_parallel_offset(data)
    end
    family === nothing && return nothing
    width = only(last(p) for p in declared_parameter_families(data) if first(p) == family)
    values = retained_family!(data, family, width)
    entries = params isa AbstractDict ? pairs(params) : ((i - 1, v) for (i, v) in enumerate(params))
    for (idx, value) in entries
        0 <= idx + offset < width && (values[idx + offset + 1] = value)
    end
    return nothing
end
