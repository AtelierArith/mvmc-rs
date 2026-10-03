//! Original Julia8bb GreenTwoEx public API controls, M0075–M0080.
//! C readdef.c GetInfoTwoBodyGEx reverses the second pair and checks row count.
//! Rust typed errors/malformed-input safety are not native C execution proof.
//! Direct-file parsing is not Julia's mutable parse_file_by_type! API.

use mvmc_expert_parsers::{
    parse_expert_mode_files, parsers::green::parse_green_two_ex_def, GreenTwoExTerm, ParseError,
    Spin,
};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

struct Input(PathBuf);

impl Input {
    fn new() -> Self {
        let path = loop {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir()
                .join(format!("issue184-green-public-{}-{id}", std::process::id()));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive input directory: {error}"),
            }
        };
        let input = Self(path);
        fs::write(
            input.0.join("modpara.def"),
            "Nsite 4\nNElec 1\nNMPTrans 1\n",
        )
        .unwrap();
        fs::write(
            input.0.join("namelist.def"),
            "ModPara modpara.def\nTwoBodyGEx greentwoex.def\n",
        )
        .unwrap();
        input
    }

    fn write_green(&self, count: usize) {
        fs::write(
            self.0.join("greentwoex.def"),
            format!(
                "===\nNCisAjsCktAlt {count}\n===\nFactored two-body Green\n===\n\
                 0 0 1 0 2 1 3 1\n"
            ),
        )
        .unwrap();
    }

    fn public_error(&self) -> String {
        match parse_expert_mode_files(self.0.join("namelist.def")).unwrap_err() {
            ParseError::InvalidInput { message } => message,
            other => panic!("expected required-definition InvalidInput, got {other}"),
        }
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively owned input directory");
    }
}

#[test]
fn original_valid_factored_row_reaches_public_loader_with_c_reordering() {
    let input = Input::new();
    input.write_green(1);
    let expected = GreenTwoExTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Up,
        site3: 3,
        spin3: Spin::Down,
        site4: 2,
        spin4: Spin::Down,
    };
    let data = parse_expert_mode_files(input.0.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    assert_eq!(data.green_two_ex_terms, [expected]);
    assert_eq!(
        parse_green_two_ex_def(input.0.join("greentwoex.def")).unwrap(),
        [expected]
    );
}

#[test]
fn original_count_two_one_row_is_fatal_on_public_loader() {
    let input = Input::new();
    input.write_green(2);
    let message = input.public_error();
    assert!(
        message.contains("Error parsing required TwoBodyGEx file"),
        "{message}"
    );
    assert!(message.contains("declared row count 2, got 1"), "{message}");
    let direct = parse_green_two_ex_def(input.0.join("greentwoex.def")).unwrap_err();
    assert_eq!(direct.kind(), io::ErrorKind::InvalidData);
    assert!(direct.to_string().contains("header count 2"), "{direct}");
}

#[test]
fn original_missing_factored_file_is_fatal_on_public_loader() {
    let input = Input::new();
    assert!(!input.0.join("greentwoex.def").exists());
    let message = input.public_error();
    assert!(
        message.contains("Required TwoBodyGEx file not found"),
        "{message}"
    );
    assert!(message.contains("greentwoex.def"), "{message}");
}
