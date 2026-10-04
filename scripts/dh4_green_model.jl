# Nonzero four-neighbor data for exhaustive original Green fixtures.
function add_dh4_green_model!(data)
    data.doublon_holon_4site_indices = [
        MVMCExpertModeParsers.DoublonHolon4SiteIndex([1 2 3 0; 0 2 3 1; 0 1 3 2; 0 1 2 3]),
        MVMCExpertModeParsers.DoublonHolon4SiteIndex([1 1 1 1; 1 1 3 3; 3 3 0 0; 2 2 2 2]),
    ]
    data.doublon_holon_4site_params = ComplexF64[ComplexF64(i/128,-i/256) for i=1:20]
    return data
end
