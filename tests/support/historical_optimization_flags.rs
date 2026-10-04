// Convert only the documented Julia/C orbital imaginary representation.
// Historical fixtures stay byte-for-byte unchanged. Native C flag tests prove
// these rules separately; this helper does not establish C input acceptance.
use mvmc_expert_parsers::ExpertModeData;

pub fn c_orbital_representation(data: &ExpertModeData, mut flags: Vec<i64>) -> Vec<i64> {
    let complex = data.orbital_terms.iter().any(|term| term.is_complex);
    let offset = data.projection_layout().n_proj + data.count_rbm_parameters();
    for index in 0..data.modpara.n_orbital_idx.max(0) as usize {
        if let Some(imaginary) = flags.get_mut(2 * (offset + index) + 1) {
            if !complex {
                *imaginary = 0;
            } else if data.i_flg_orbital_parallel == 1
                && index >= data.n_orbital_anti_parallel as usize
            {
                *imaginary = 1;
            }
        }
    }
    flags
}
