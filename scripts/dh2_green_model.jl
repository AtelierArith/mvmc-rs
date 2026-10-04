# Shared nonzero DH2 data for exhaustive normal and FSZ operator fixtures.
function add_dh2_green_model!(data)
    data.doublon_holon_2site_indices = [
        MVMCExpertModeParsers.DoublonHolon2SiteIndex([1 2; 0 2; 0 1; 0 1]),
        MVMCExpertModeParsers.DoublonHolon2SiteIndex([0 0; 0 0; 3 3; 2 2]),
    ]
    data.doublon_holon_2site_params = ComplexF64[ComplexF64(i/128,-i/256) for i=1:12]
    return data
end
