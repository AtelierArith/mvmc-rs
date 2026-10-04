//! Lossless test-only transport, not computed floating-point parity policy.
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: [u8; 8] = *b"MVMCF64\0";
const HEADER: usize = 32;
const MAX_ELEMENTS: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(super) enum Kind {
    Oo = 1,
    Ho = 2,
    Parameters = 3,
    Energy = 4,
    Matrix = 5,
    Rhs = 6,
    Increment = 7,
}

impl Kind {
    fn width(self) -> u64 {
        match self {
            Self::Oo | Self::Ho | Self::Parameters | Self::Energy => 2,
            Self::Matrix | Self::Rhs | Self::Increment => 1,
        }
    }
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid diagnostic transport")
}

fn header(kind: Kind, count: u64) -> io::Result<[u8; HEADER]> {
    if count > MAX_ELEMENTS {
        return Err(invalid());
    }
    let payload = count
        .checked_mul(kind.width())
        .and_then(|n| n.checked_mul(8));
    let mut bytes = [0; HEADER];
    bytes[..8].copy_from_slice(&MAGIC);
    bytes[8..10].copy_from_slice(&1_u16.to_le_bytes()); // version
    bytes[10] = 1; // explicitly little endian
    bytes[11] = if kind.width() == 2 { 2 } else { 1 }; // f64 / complex f64 re,im
    bytes[12..14].copy_from_slice(&(kind as u16).to_le_bytes());
    // bytes14..16 reserved, must remain zero
    bytes[16..24].copy_from_slice(&count.to_le_bytes()); // logical elements
    bytes[24..32].copy_from_slice(&payload.ok_or_else(invalid)?.to_le_bytes());
    Ok(bytes)
}

fn encode<W: Write, I: IntoIterator<Item = f64>>(
    mut writer: W,
    kind: Kind,
    count: u64,
    values: I,
) -> io::Result<()> {
    let bytes = header(kind, count)?;
    writer.write_all(&bytes)?;
    let expected = count * kind.width();
    let mut actual = 0_u64;
    for value in values {
        if actual == expected {
            return Err(invalid());
        }
        writer.write_all(&value.to_bits().to_le_bytes())?;
        actual += 1;
    }
    if actual != expected {
        return Err(invalid());
    }
    writer.flush()
}

pub(super) fn write_record<I: IntoIterator<Item = f64>>(
    path: &Path,
    kind: Kind,
    count: usize,
    values: I,
) -> io::Result<()> {
    // Invalid declaration rejected before file creation; failures retain partial files.
    header(kind, u64::try_from(count).map_err(|_| invalid())?)?;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    // Fixed32KiB batch buffer: avoids one filesystem syscall per8-byte word.
    // No change to payload bytes, order, or declared cardinality.
    encode(
        io::BufWriter::with_capacity(32 * 1024, file),
        kind,
        count as u64,
        values,
    )
}

/// Returns transported IEEE bits, never declares numerical equivalence.
pub(super) fn decode<R: Read>(mut reader: R, kind: Kind, expected: u64) -> io::Result<Vec<u64>> {
    let wanted = header(kind, expected)?;
    let mut actual = [0; HEADER];
    reader.read_exact(&mut actual)?;
    if actual != wanted {
        return Err(invalid());
    }
    // Header/count validated before any allocation; no untrusted count is used.
    let scalars = expected * kind.width();
    let mut result = Vec::with_capacity(scalars as usize);
    for _ in 0..scalars {
        let mut word = [0; 8];
        reader.read_exact(&mut word)?;
        result.push(u64::from_le_bytes(word));
    }
    let mut extra = [0];
    if reader.read(&mut extra)? != 0 {
        return Err(invalid());
    }
    Ok(result)
}

#[test]
fn literal_header_and_boundary_bits_roundtrip() {
    // Independent literal ABI: two real elements, +0 and -0.
    let literal = [
        77, 86, 77, 67, 70, 54, 52, 0, 1, 0, 1, 1, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
    ];
    let mut encoded = Vec::new();
    encode(&mut encoded, Kind::Matrix, 2, [0.0, -0.0]).unwrap();
    assert_eq!(encoded, literal);
    assert_eq!(
        decode(&literal[..], Kind::Matrix, 2).unwrap(),
        [0, 1_u64 << 63]
    );
    let bits = [
        0x0000000000000000,
        0x8000000000000000, // both zero signs, including complex transport
        0x0000000000000001,
        0x8000000000000001, // smallest subnormals
        0x000fffffffffffff,
        0x0010000000000000, // subnormal/normal boundary
        0x7fefffffffffffff,
        0xffefffffffffffff, // finite extremes
        0x3ff0000000000000,
        0xbff0000000000000, // independently known +/-1
    ];
    for kind in [
        Kind::Oo,
        Kind::Ho,
        Kind::Parameters,
        Kind::Energy,
        Kind::Matrix,
        Kind::Rhs,
        Kind::Increment,
    ] {
        let mut encoded = Vec::new();
        encode(
            &mut encoded,
            kind,
            bits.len() as u64 / kind.width(),
            bits.map(f64::from_bits),
        )
        .unwrap();
        assert_eq!(
            decode(&encoded[..], kind, bits.len() as u64 / kind.width()).unwrap(),
            bits
        );
    }
}

#[test]
fn malformed_truncated_oversized_and_wrong_record_rejected() {
    let mut valid = Vec::new();
    encode(&mut valid, Kind::Matrix, 1, [1.0]).unwrap();
    assert_eq!(
        decode(&valid[..], Kind::Matrix, 1).unwrap(),
        [0x3ff0000000000000]
    );
    for index in [0, 8, 10, 11, 12, 14, 16, 24] {
        let mut changed = valid.clone();
        changed[index] ^= 1;
        assert!(
            decode(&changed[..], Kind::Matrix, 1).is_err(),
            "byte {index}"
        );
    }
    for length in 0..valid.len() {
        assert!(
            decode(&valid[..length], Kind::Matrix, 1).is_err(),
            "length {length}"
        );
    }
    let mut extra = valid.clone();
    extra.push(0);
    assert!(decode(&extra[..], Kind::Matrix, 1).is_err());
    assert!(decode(&valid[..], Kind::Rhs, 1).is_err());
    assert!(decode(&valid[..], Kind::Matrix, MAX_ELEMENTS + 1).is_err());
    let mut oversized = valid.clone();
    oversized[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(decode(&oversized[..], Kind::Matrix, 1).is_err());
    assert!(encode(Vec::new(), Kind::Matrix, 1, []).is_err());
    assert!(encode(Vec::new(), Kind::Matrix, 1, [1.0, 2.0]).is_err());
    assert_eq!(
        header(Kind::Matrix, MAX_ELEMENTS).unwrap()[24..32],
        (8388608_u64).to_le_bytes()
    );
    assert!(header(Kind::Matrix, MAX_ELEMENTS + 1).is_err());
    let mut empty = Vec::new();
    encode(&mut empty, Kind::Rhs, 0, []).unwrap();
    assert!(decode(&empty[..], Kind::Rhs, 0).unwrap().is_empty());
}

#[test]
fn file_creation_is_exclusive_and_preserves_literal_bits() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("issue180-transport-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("literal.f64bin");
    write_record(&path, Kind::Rhs, 2, [0.0, -0.0]).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert_eq!(before.len(), 48);
    assert_eq!(
        decode(&before[..], Kind::Rhs, 2).unwrap(),
        [0, 0x8000000000000000]
    );
    assert_eq!(
        write_record(&path, Kind::Rhs, 2, [1.0, 2.0])
            .unwrap_err()
            .kind(),
        io::ErrorKind::AlreadyExists
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
}
