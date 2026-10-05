//! Pure-Rust fixture-input SHA256 checks; no oracle or external process.
use std::fs;
use std::path::Path;

fn sha256(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut bytes = input.to_vec();
    let bits = u64::try_from(bytes.len()).unwrap().checked_mul(8).unwrap();
    bytes.push(0x80);
    while bytes.len() % 64 != 56 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&bits.to_be_bytes());
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let (blocks, remainder) = bytes.as_chunks::<64>();
    assert!(remainder.is_empty());
    for block in blocks {
        let mut w = [0u32; 64];
        for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*chunk);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (value, add) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *value = value.wrapping_add(add);
        }
    }
    h.iter().map(|value| format!("{value:08x}")).collect()
}

pub fn verify_inputs(manifest: &Path, inputs: &Path) {
    let text = fs::read_to_string(manifest).expect("independent input hashes required");
    let mut names = std::collections::BTreeSet::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 2, "input manifest row");
        let name = fields[1];
        assert!(!name.contains('/') && !name.contains('\\') && name != "." && name != "..");
        assert!(names.insert(name), "duplicate input hash");
        assert_eq!(
            sha256(&fs::read(inputs.join(name)).expect("hashed input missing")),
            fields[0],
            "actual input {name}"
        );
    }
    assert!(names.contains("namelist.def") && names.contains("modpara.def"));
    // These canonical runner gates use InitialDef::Auto: a neighboring file is
    // consumed before In* overlays even if the namelist does not mention it.
    if inputs.join("initial.def").exists() {
        assert!(names.contains("initial.def"), "unhashed auto initial.def");
    }
    // Use the runtime parser's ordered metadata, including every initial overlay.
    // Fixture inputs are deliberately flat: reject references outside this bundle.
    let namelist = fs::read_to_string(inputs.join("namelist.def")).unwrap();
    for (_, name) in mvmc_expert_parsers::utils::file::parse_namelist_content(&namelist) {
        assert!(
            !name.contains('/') && !name.contains('\\') && name != "." && name != "..",
            "referenced input outside flat fixture bundle: {name}"
        );
        assert!(
            names.contains(name.as_str()),
            "unhashed referenced input: {name}"
        );
    }
}

#[test]
fn input_manifest_requires_every_definition_and_initial_overlay() {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("ctest-input-closure-{}-{id}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    struct OwnedDirectory(std::path::PathBuf);
    impl Drop for OwnedDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove exclusively owned test directory");
        }
    }
    let owned = OwnedDirectory(directory);
    let root = &owned.0;
    let namelist = b"# input bundle\nModPara modpara.def\nOrbital orbital.def\nInGutzwiller initial.def\nInOrbital initial.def // shared overlay source\n";
    let files: [(&str, &[u8]); 4] = [
        ("namelist.def", namelist),
        ("modpara.def", b"--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNSite 2\n"),
        ("orbital.def", b"mapping\n"),
        ("initial.def", b"0 0.25 0\n1 0.5 0\n"),
    ];
    let mut rows = Vec::new();
    for (name, bytes) in files {
        fs::write(root.join(name), bytes).unwrap();
        rows.push(format!("{} {name}\n", sha256(bytes)));
    }
    let manifest = root.join("manifest.txt");
    fs::write(&manifest, rows.concat()).unwrap();
    verify_inputs(&manifest, root);
    // Also require the implicit Auto source, independent of explicit overlays.
    let auto_namelist = b"ModPara modpara.def\nOrbital orbital.def\n";
    fs::write(root.join("namelist.def"), auto_namelist).unwrap();
    let mut auto_rows = rows.clone();
    auto_rows[0] = format!("{} namelist.def\n", sha256(auto_namelist));
    fs::write(&manifest, auto_rows[..3].concat()).unwrap();
    assert!(std::panic::catch_unwind(|| verify_inputs(&manifest, root)).is_err());
    fs::write(&manifest, auto_rows.concat()).unwrap();
    verify_inputs(&manifest, root);
    fs::write(root.join("namelist.def"), namelist).unwrap();
    for omitted in [2, 3] {
        let incomplete: String = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != omitted)
            .map(|(_, row)| row.as_str())
            .collect();
        fs::write(&manifest, incomplete).unwrap();
        assert!(std::panic::catch_unwind(|| verify_inputs(&manifest, root)).is_err());
    }
    fs::write(&manifest, rows.concat()).unwrap();
    fs::write(root.join("initial.def"), b"0 0.75 0\n").unwrap();
    assert!(std::panic::catch_unwind(|| verify_inputs(&manifest, root)).is_err());
    fs::write(root.join("initial.def"), files[3].1).unwrap();
    fs::remove_file(root.join("orbital.def")).unwrap();
    assert!(std::panic::catch_unwind(|| verify_inputs(&manifest, root)).is_err());
}

#[test]
fn sha256_standard_vectors() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn input_manifest_rejects_tampering_and_malformed_rows() {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("ctest-provenance-{}-{id}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    struct OwnedDirectory(std::path::PathBuf);
    impl Drop for OwnedDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove exclusively owned test directory");
        }
    }
    let owned = OwnedDirectory(directory);
    let root = &owned.0;
    let manifest = root.join("manifest.txt");
    fs::write(root.join("namelist.def"), b"abc").unwrap();
    fs::write(root.join("modpara.def"), b"").unwrap();
    let row = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  namelist.def\n";
    let good = format!(
        "{row}e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  modpara.def\n"
    );
    fs::write(&manifest, &good).unwrap();
    verify_inputs(&manifest, root);
    fs::write(root.join("namelist.def"), b"changed").unwrap();
    assert!(std::panic::catch_unwind(|| verify_inputs(&manifest, root)).is_err());
    fs::write(root.join("namelist.def"), b"abc").unwrap();
    for invalid in [
        String::new(),
        "hashonly\n".into(),
        "invalid namelist.def\n".into(),
        format!("{good}{row}"),
        good.replace("namelist.def", "../namelist.def"),
        row.into(),
        good.replace("modpara.def", "missing.def"),
    ] {
        fs::write(&manifest, invalid).unwrap();
        assert!(std::panic::catch_unwind(|| verify_inputs(&manifest, root)).is_err());
    }
}
