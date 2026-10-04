//! Optional developer decoder; build/run separately only after SOURCE approval.
//! Exact transported bits are not a floating-point parity acceptance criterion.
#[path = "../crates/mvmc-core/src/issue180_binary_diagnostics.rs"]
mod transport;

use std::io::{self, Write};
use transport::Kind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: decoder KIND EXPECTED_LOGICAL_COUNT INPUT.f64bin".into());
    }
    let kind = match args[1].as_str() {
        "oo" => Kind::Oo,
        "ho" => Kind::Ho,
        "parameters" => Kind::Parameters,
        "energy" => Kind::Energy,
        "matrix" => Kind::Matrix,
        "rhs" => Kind::Rhs,
        "increment" => Kind::Increment,
        _ => return Err("unknown record kind".into()),
    };
    if args[2].is_empty() || !args[2].bytes().all(|b| b.is_ascii_digit()) {
        return Err("invalid expected count".into());
    }
    let count = args[2].parse::<u64>()?;
    let input = std::fs::File::open(&args[3])?;
    // Caller supplies schema-derived expected kind/count; header cannot define
    // its own expectation. Validate complete record before printing anything.
    let bits = transport::decode(input, kind, count)?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    for (index, word) in bits.iter().enumerate() {
        if index != 0 {
            write!(output, " ")?;
        }
        // Hexadecimal bits avoid rounding ambiguity in transport inspection.
        write!(output, "{word:016x}")?;
    }
    writeln!(output)?;
    Ok(())
}
