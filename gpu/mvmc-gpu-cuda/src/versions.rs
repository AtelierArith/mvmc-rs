//! Driver and CUDA library version queries (no CUDA toolkit needed at build time).
//!
//! Every library is opened with `dlopen`; a library that cannot be loaded yields `None`,
//! which the report renders as `unavailable` (never silently omitted).

use libloading::Library;
use std::ffi::{c_char, c_int, c_uint};

/// Versions of the NVIDIA stack visible at run time.
#[derive(Debug, Clone, Default)]
pub struct Versions {
    /// Kernel driver version from NVML (for example `580.178.04`).
    pub driver_version: Option<String>,
    /// `cuDriverGetVersion` formatted as `major.minor`.
    pub cuda_driver_api: Option<String>,
    /// NVRTC `major.minor`.
    pub nvrtc: Option<String>,
    /// cuBLAS `major.minor.patch`.
    pub cublas: Option<String>,
    /// cuSOLVER `major.minor.patch`.
    pub cusolver: Option<String>,
}

fn open(names: &[&str]) -> Option<Library> {
    names.iter().find_map(|name| {
        // SAFETY: loading a system CUDA library only runs its initializers.
        unsafe { Library::new(name).ok() }
    })
}

fn driver_api() -> Option<String> {
    let lib = open(&["libcuda.so.1", "libcuda.so"])?;
    // SAFETY: signature `CUresult cuDriverGetVersion(int*)`.
    unsafe {
        let f = lib
            .get::<unsafe extern "C" fn(*mut c_int) -> c_int>(b"cuDriverGetVersion\0")
            .ok()?;
        let mut v: c_int = 0;
        (f(&mut v) == 0).then(|| format!("{}.{}", v / 1000, (v % 1000) / 10))
    }
}

fn nvml_driver() -> Option<String> {
    let lib = open(&["libnvidia-ml.so.1", "libnvidia-ml.so"])?;
    // SAFETY: NVML signatures `nvmlInit_v2()`, `nvmlSystemGetDriverVersion(char*, uint)`.
    unsafe {
        let init = lib
            .get::<unsafe extern "C" fn() -> c_int>(b"nvmlInit_v2\0")
            .ok()?;
        let get = lib
            .get::<unsafe extern "C" fn(*mut c_char, c_uint) -> c_int>(
                b"nvmlSystemGetDriverVersion\0",
            )
            .ok()?;
        let shutdown = lib
            .get::<unsafe extern "C" fn() -> c_int>(b"nvmlShutdown\0")
            .ok()?;
        if init() != 0 {
            return None;
        }
        let mut buf = [0 as c_char; 96];
        let status = get(buf.as_mut_ptr(), buf.len() as c_uint);
        shutdown();
        (status == 0).then(|| {
            let bytes: Vec<u8> = buf
                .iter()
                .take_while(|c| **c != 0)
                .map(|c| *c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }
}

fn nvrtc() -> Option<String> {
    let lib = open(&[
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11.2",
    ])?;
    // SAFETY: signature `nvrtcResult nvrtcVersion(int*, int*)`.
    unsafe {
        let f = lib
            .get::<unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int>(b"nvrtcVersion\0")
            .ok()?;
        let (mut major, mut minor) = (0, 0);
        (f(&mut major, &mut minor) == 0).then(|| format!("{major}.{minor}"))
    }
}

/// `libraryPropertyType` MAJOR_VERSION / MINOR_VERSION / PATCH_LEVEL.
fn property_version(lib: &Library, symbol: &[u8]) -> Option<String> {
    // SAFETY: signature `status get*Property(libraryPropertyType, int*)`.
    unsafe {
        let f = lib
            .get::<unsafe extern "C" fn(c_int, *mut c_int) -> c_int>(symbol)
            .ok()?;
        let mut parts = [0 as c_int; 3];
        for (kind, slot) in parts.iter_mut().enumerate() {
            if f(kind as c_int, slot) != 0 {
                return None;
            }
        }
        Some(format!("{}.{}.{}", parts[0], parts[1], parts[2]))
    }
}

fn cublas() -> Option<String> {
    let lib = open(&["libcublas.so", "libcublas.so.13", "libcublas.so.12"])?;
    property_version(&lib, b"cublasGetProperty\0")
}

fn cusolver() -> Option<String> {
    let lib = open(&["libcusolver.so", "libcusolver.so.12", "libcusolver.so.11"])?;
    property_version(&lib, b"cusolverGetProperty\0")
}

/// Query all versions; unavailable components are `None`.
pub fn query() -> Versions {
    Versions {
        driver_version: nvml_driver(),
        cuda_driver_api: driver_api(),
        nvrtc: nvrtc(),
        cublas: cublas(),
        cusolver: cusolver(),
    }
}
