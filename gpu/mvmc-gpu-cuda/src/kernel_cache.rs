//! Cache only compiled PTX, never device/context-owned modules or numerical state.
//! Exact input bytes are checked on every hit; unreadable/corrupt entries recompile.
use cudarc::{
    driver::{CudaContext, CudaModule},
    nvrtc::{self, CompileOptions, Ptx},
};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

const VARIABLE: &str = "MVMC_RS_CUDA_KERNEL_CACHE";
const MAGIC: &[u8] = b"mvmc-ptx-cache-v1\0";
static NEXT: AtomicU64 = AtomicU64::new(0);

fn digest(bytes: &[u8]) -> u64 {
    let mut hash = DefaultHasher::new();
    bytes.hash(&mut hash);
    hash.finish()
}

fn directory() -> Option<PathBuf> {
    match std::env::var_os(VARIABLE) {
        Some(value) if value == "off" => None,
        Some(value) if !value.is_empty() => Some(value.into()),
        Some(_) => None,
        None => std::env::var_os("XDG_CACHE_HOME")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
            .map(|p| p.join("mvmc-rs/ptx-v1")),
    }
}

fn identity(source: &str, options: &CompileOptions) -> Option<Vec<u8>> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both output pointers address live c_int values.
    unsafe { nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .ok()?;
    // Linux library path/size/mtime also invalidates a same-major/minor library replacement.
    let mut libraries = std::collections::BTreeSet::new();
    if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
        for line in maps.lines() {
            if let Some(path) = line
                .split_whitespace()
                .last()
                .filter(|p| p.contains("/libnvrtc.so"))
            {
                if let Ok(meta) = std::fs::metadata(path) {
                    libraries.insert(format!("{path}:{}:{:?}", meta.len(), meta.modified().ok()));
                }
            }
        }
    }
    Some(
        format!("{MAGIC:?}\nNVRTC={major}.{minor}\n{libraries:?}\n{options:?}\n{source}")
            .into_bytes(),
    )
}

fn encode(input: &[u8], ptx: &str) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend((input.len() as u64).to_le_bytes());
    bytes.extend(digest(ptx.as_bytes()).to_le_bytes());
    bytes.extend(input);
    bytes.extend(ptx.as_bytes());
    bytes
}

fn decode<'a>(input: &[u8], bytes: &'a [u8]) -> Option<&'a str> {
    let bytes = bytes.strip_prefix(MAGIC)?;
    let length = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?);
    if length != input.len() as u64 {
        return None;
    }
    let checksum = u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?);
    let bytes = bytes.get(16..)?;
    // The filename hash is just a lookup aid; even a collision cannot select other inputs.
    if bytes.get(..input.len())? != input {
        return None;
    }
    let ptx = bytes.get(input.len()..)?;
    if ptx.is_empty() || digest(ptx) != checksum {
        return None;
    }
    std::str::from_utf8(ptx).ok()
}

fn publish(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().expect("cache path has parent");
    std::fs::create_dir_all(parent)?;
    let temp = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    // Do not remove a pre-existing temporary file when create_new fails.
    let mut file = options.open(&temp)?;
    let result = (|| {
        file.write_all(bytes)?;
        drop(file);
        std::fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}

pub(crate) fn load(
    context: &Arc<CudaContext>,
    source: &str,
    options: CompileOptions,
    component: &str,
) -> Result<Arc<CudaModule>, String> {
    load_in(context, source, options, component, directory()).map(|(module, _)| module)
}

fn load_in(
    context: &Arc<CudaContext>,
    source: &str,
    options: CompileOptions,
    component: &str,
    directory: Option<PathBuf>,
) -> Result<(Arc<CudaModule>, bool), String> {
    let start = Instant::now();
    let input = identity(source, &options);
    let path = directory
        .zip(input.as_ref())
        .map(|(dir, key)| dir.join(format!("{:016x}.ptx-cache", digest(key))));
    if let (Some(path), Some(input)) = (&path, &input) {
        // Bound reads of malformed entries. A normal module is far below 64 MiB.
        if std::fs::metadata(path).is_ok_and(|m| m.len() < 64 * 1024 * 1024) {
            if let Ok(bytes) = std::fs::read(path) {
                if let Some(ptx) = decode(input, &bytes) {
                    if let Ok(module) = context.load_module(Ptx::from_src(ptx)) {
                        if std::env::var_os("MVMC_RS_CUDA_INIT_PROFILE").is_some() {
                            eprintln!(
                                "cuda-kernel component={component} cache=hit setup_s={:.6}",
                                start.elapsed().as_secs_f64()
                            );
                        }
                        return Ok((module, true));
                    }
                }
            }
        }
    }
    let compile_start = Instant::now();
    let ptx = nvrtc::compile_ptx_with_opts(source, options).map_err(|e| format!("NVRTC: {e:?}"))?;
    let compile_seconds = compile_start.elapsed().as_secs_f64();
    let text = ptx.to_src();
    let module = context
        .load_module(ptx)
        .map_err(|e| format!("CUDA module: {e:?}"))?;
    if let (Some(path), Some(input)) = (&path, &input) {
        // Cache IO never changes whether a supported kernel can run.
        let _ = publish(path, &encode(input, &text));
    }
    if std::env::var_os("MVMC_RS_CUDA_INIT_PROFILE").is_some() {
        eprintln!(
            "cuda-kernel component={component} cache={} nvrtc_s={compile_seconds:.6} setup_s={:.6}",
            if path.is_some() { "miss" } else { "disabled" },
            start.elapsed().as_secs_f64()
        );
    }
    Ok((module, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_changed_input_truncation_and_corruption() {
        let encoded = encode(b"source and options", "PTX payload");
        assert_eq!(decode(b"source and options", &encoded), Some("PTX payload"));
        assert!(decode(b"different input", &encoded).is_none());
        for length in 0..encoded.len() {
            assert!(decode(b"source and options", &encoded[..length]).is_none());
        }
        for position in 0..encoded.len() {
            let mut corrupt = encoded.clone();
            corrupt[position] ^= 1;
            assert!(decode(b"source and options", &corrupt).is_none());
        }
    }

    #[test]
    fn concurrent_publish_produces_one_complete_entry() {
        let dir = std::env::temp_dir().join(format!(
            "mvmc-cache-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = dir.join("entry");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let path = &path;
                scope.spawn(move || publish(path, &encode(b"same source", "same PTX")).unwrap());
            }
        });
        assert_eq!(
            decode(b"same source", &std::fs::read(&path).unwrap()),
            Some("same PTX")
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    #[ignore = "requires an NVIDIA device and NVRTC; run the explicit CUDA gate"]
    fn cuda_cache_hit_rejection_and_unwritable_location() {
        let dir = std::env::temp_dir().join(format!(
            "mvmc-cuda-cache-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let context = CudaContext::new(0).unwrap();
        let source = "extern \"C\" __global__ void cache_probe() {}";
        let options = CompileOptions {
            fmad: Some(false),
            arch: Some("compute_80"),
            ..Default::default()
        };
        let open =
            |directory| load_in(&context, source, options.clone(), "test", directory).unwrap();
        let (module, hit) = open(Some(dir.clone()));
        assert!(!hit);
        module.load_function("cache_probe").unwrap();
        assert!(open(Some(dir.clone())).1);
        let path = std::fs::read_dir(&dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let input = identity(source, &options).unwrap();
        // A well-formed cache envelope whose PTX the driver rejects must recompile.
        publish(&path, &encode(&input, "invalid PTX")).unwrap();
        assert!(!open(Some(dir.clone())).1);
        assert!(open(Some(dir.clone())).1);
        std::fs::write(&path, b"truncated entry").unwrap();
        assert!(!open(Some(dir.clone())).1);
        assert!(!open(None).1);
        // A file used as the directory makes cache writes impossible, even as root.
        let unavailable = dir.join("not-a-directory");
        std::fs::write(&unavailable, b"file").unwrap();
        assert!(!open(Some(unavailable)).1);
        let changed = "extern \"C\" __global__ void changed_probe() {}";
        let (_, hit) = load_in(
            &context,
            changed,
            options.clone(),
            "test",
            Some(dir.clone()),
        )
        .unwrap();
        assert!(!hit);
        let options_changed = CompileOptions {
            fmad: Some(true),
            ..options
        };
        assert!(
            !load_in(&context, source, options_changed, "test", Some(dir.clone()))
                .unwrap()
                .1
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
