//! Word-source seam for sampling. SFMT remains the default C-parity generator.
//! All sources use the C modulo and exact `[0,1)` one-word conversion.

/// Sampling consumes words in control-flow order; rejected candidates consume their draws too.
pub trait SamplingRng {
    /// Consume one raw 32-bit word.
    fn gen_rand32(&mut self) -> u32;
    /// Words consumed by callers (prefetched words are excluded).
    fn words_consumed(&self) -> u128;
    /// Nonconsuming preview for diagnostic traces and device-resident sampling.
    /// The following `gen_rand32` calls must return this same prefix.
    fn dump_rand32(&self, out: &mut [u32]);
    /// SFMT raw-checkpoint schema is unavailable for other algorithms.
    fn sfmt_snapshot(&self) -> Option<([u32; 624], usize)> {
        None
    }
    /// Exact one-word mapping to `[0, 1)`, matching SFMT real2.
    fn genrand_real2(&mut self) -> f64 {
        f64::from(self.gen_rand32()) * (1.0 / 4294967296.0)
    }
}

impl SamplingRng for sfmt19937::Sfmt19937Rng {
    fn gen_rand32(&mut self) -> u32 {
        self.gen_rand32()
    }
    fn words_consumed(&self) -> u128 {
        self.words_consumed()
    }
    fn dump_rand32(&self, out: &mut [u32]) {
        self.dump_rand32(out)
    }
    fn sfmt_snapshot(&self) -> Option<([u32; 624], usize)> {
        Some(self.state_snapshot())
    }
}

/// Native XORWOW word generator. Seed initialization follows NVIDIA's
/// `curand_init(seed, 0, 0)`; host-library interleaved output ordering is not implied.
#[derive(Clone, Debug)]
pub struct XorwowRng {
    state: [u32; 6],
    consumed: u128,
}

impl XorwowRng {
    /// Initialize subsequence zero at offset zero for this 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let t0 = (seed as u32 ^ 0xaad2_6b49).wrapping_mul(1_099_087_573);
        let t1 = ((seed >> 32) as u32 ^ 0xf7dc_efdd).wrapping_mul(2_591_861_531);
        Self {
            state: [
                123_456_789_u32.wrapping_add(t0),
                362_436_069 ^ t0,
                521_288_629_u32.wrapping_add(t1),
                88_675_123 ^ t1,
                5_783_321_u32.wrapping_add(t0),
                6_615_241_u32.wrapping_add(t0).wrapping_add(t1),
            ],
            consumed: 0,
        }
    }
    /// Resume raw `[x, y, z, w, v, d]` state with a fresh consumed counter.
    pub fn from_state(state: [u32; 6]) -> Self {
        Self { state, consumed: 0 }
    }
    /// Raw `[x, y, z, w, v, d]` state.
    pub fn state(&self) -> [u32; 6] {
        self.state
    }
}

impl SamplingRng for XorwowRng {
    fn gen_rand32(&mut self) -> u32 {
        let t = self.state[0] ^ (self.state[0] >> 2);
        self.state.copy_within(1..5, 0);
        self.state[4] = self.state[4] ^ (self.state[4] << 4) ^ t ^ (t << 1);
        self.state[5] = self.state[5].wrapping_add(362_437);
        self.consumed += 1;
        self.state[4].wrapping_add(self.state[5])
    }
    fn words_consumed(&self) -> u128 {
        self.consumed
    }
    fn dump_rand32(&self, out: &mut [u32]) {
        let mut copy = self.clone();
        for word in out {
            *word = copy.gen_rand32();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SamplingRng, XorwowRng};
    #[test]
    fn xorwow_matches_independent_nvidia_header_fixture() {
        let fixture = include_str!("../../../../tests/fixtures/xorwow/nvidia_seed_words.csv");
        let mut seed = None;
        let mut rng = XorwowRng::new(0);
        for line in fixture.lines().skip(1) {
            let fields: Vec<u64> = line.split(',').map(|s| s.parse().unwrap()).collect();
            if seed != Some(fields[0]) {
                seed = Some(fields[0]);
                rng = XorwowRng::new(fields[0]);
            }
            assert_eq!(rng.words_consumed(), u128::from(fields[1]));
            assert_eq!(rng.gen_rand32(), fields[2] as u32);
        }
    }
}
