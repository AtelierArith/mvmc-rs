use mvmc_core::state::SlaterElmFlat;

pub struct Case {
    pub name: String,
    pub hops: usize,
    pub words: Vec<u32>,
    // C rows: consumed, mi, spin, ri, rj, rejected, accepted, up, down, total, window.
    pub steps: Vec<Vec<i32>>,
}

pub fn cases() -> Vec<Case> {
    let mut lines = include_str!("../../../../tests/fixtures/interval_c_rng_557.txt").lines();
    let mut out = Vec::new();
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        assert_eq!(fields[0], "case");
        let hops = fields[2].parse().unwrap();
        let words = lines
            .next()
            .unwrap()
            .split_whitespace()
            .skip(1)
            .map(|s| s.parse().unwrap())
            .collect();
        let steps = (0..hops)
            .map(|_| {
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .skip(1)
                    .map(|s| s.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(lines.next(), Some("end"));
        out.push(Case {
            name: fields[1].into(),
            hops,
            words,
            steps,
        });
    }
    out
}

pub fn slater() -> SlaterElmFlat<f64> {
    let mut slater = SlaterElmFlat::zeros(1, 4);
    for up in 0..4 {
        for down in 0..4 {
            let value = 2f64.powi((up + down) as i32);
            slater.set(0, up, down + 4, value);
            slater.set(0, down + 4, up, -value);
        }
    }
    slater
}
