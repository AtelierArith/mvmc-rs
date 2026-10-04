# Parsed nonzero all-nine-section model for exhaustive original Green fixtures.
function add_rbm_green_model!(data)
    file=joinpath(@__DIR__,"..","tests","fixtures","rbm","production","namelist_all.def")
    parsed=MVMCExpertModeParsers.parse_expert_mode_files(file)
    MVMCExpertModeParsers.read_input_parameters!(parsed,file)
    for field in fieldnames(typeof(data))
        occursin("_rbm_",String(field)) && endswith(String(field),"_terms") &&
            setfield!(data,field,getfield(parsed,field))
    end
    data.modpara.nneuron_charge=2
    data.modpara.nneuron_spin=3
    data.modpara.nneuron_general=4
    return data
end
