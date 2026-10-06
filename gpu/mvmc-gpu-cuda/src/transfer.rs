//! Pinned (page-locked) host buffers and stream-ordered asynchronous transfers (issue #432).
//!
//! tenferro 0.7.1 offers synchronous, pageable `upload_tensor` / `download_tensor` only (see
//! `docs/design/gpu-readiness.md` section 10.7 for the measurements). This module is the
//! in-repo replacement for the hot, small per-step transfers of the device-resident sampler
//! (#434): walker configurations in; Pfaffians, energies and O vectors out.
//!
//! * [`PinnedBuf`]: page-locked host memory, [`PinnedKind::Cached`] (default) or write-combined
//!   for large upload staging ([`PinnedKind::for_upload`]). Pinned memory makes the copy
//!   asynchronous with respect to the host (a 16 MB enqueue returns in about 4 us while the
//!   pageable copy blocks the host for 2.3 ms).
//! * [`PinnedPool`]: size-class pool; `cudaHostAlloc` costs about a millisecond and must not
//!   happen per step.
//! * [`TransferStream`]: one CUDA stream used for copies. `upload_async` / `download_async`
//!   enqueue a copy and return a [`Pending`] that borrows the host buffer, so the host cannot
//!   touch a buffer the device may still read or write (enforced by the borrow checker;
//!   dropping a `Pending` waits for the copy). Cross-stream dependencies use events:
//!   [`TransferStream::record`] and [`TransferStream::wait_event`], so a compute stream waits
//!   for an upload without a device-wide synchronize.
//!
//! Device memory is a cudarc `CudaSlice`, or a raw device pointer through the `unsafe`
//! `*_raw` variants (for buffers owned by another allocator, for example a tenferro raw
//! session). The context must have event tracking disabled
//! ([`disable_event_tracking`]) so the safe cudarc wrappers do not insert hidden
//! synchronization; this module orders everything with explicit events.

use std::sync::{Arc, Mutex};

use cudarc::driver::{
    result, sys, CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError,
};

/// Disable cudarc's implicit cross-stream event tracking on `ctx`.
///
/// # Safety
///
/// After this call the caller (this module) is responsible for every cross-stream dependency.
pub unsafe fn disable_event_tracking(ctx: &Arc<CudaContext>) {
    ctx.disable_event_tracking();
}

/// Kind of pinned allocation.
///
/// Measured on an RTX 3060 (`docs/design/gpu-readiness.md` section 10.7): write-combined memory
/// reaches the same bandwidth as cached pinned memory for large copies but is much slower for
/// small ones (a 4 KB host-to-device copy took 51 us from write-combined memory against 7 us
/// from cached pinned memory), and host reads of it are very slow. Use
/// [`PinnedKind::for_upload`] to choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnedKind {
    /// Write-combined (`CU_MEMHOSTALLOC_WRITECOMBINED`): host writes are fast, host reads are
    /// very slow, small copies are slow. Only for large staging buffers that the host writes
    /// once per use.
    WriteCombined,
    /// Ordinary pinned memory: fast host reads and writes, fast small copies. The default
    /// choice, and the landing zone for device-to-host copies.
    Cached,
}

impl PinnedKind {
    /// Copies of at least this many bytes may use write-combined memory.
    pub const WRITE_COMBINED_MIN_BYTES: usize = 32 << 10;

    /// The kind to use for a host-to-device staging buffer of `bytes`: cached pinned memory
    /// below [`WRITE_COMBINED_MIN_BYTES`](Self::WRITE_COMBINED_MIN_BYTES) (and always when the
    /// host also reads the buffer), write-combined above.
    pub fn for_upload(bytes: usize) -> Self {
        if bytes >= Self::WRITE_COMBINED_MIN_BYTES {
            Self::WriteCombined
        } else {
            Self::Cached
        }
    }
}

/// A page-locked host buffer of raw bytes.
pub struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
    kind: PinnedKind,
    ctx: Arc<CudaContext>,
}

// SAFETY: the buffer owns its allocation; access is mediated by `&`/`&mut`.
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// Allocate `bytes` of pinned memory (zero length is rounded up to one byte).
    pub fn new(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        kind: PinnedKind,
    ) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let bytes = bytes.max(1);
        let flags = match kind {
            PinnedKind::WriteCombined => sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            PinnedKind::Cached => 0,
        };
        // SAFETY: allocation of `bytes` bytes; freed in `Drop` with `free_host`.
        let ptr = unsafe { result::malloc_host(bytes, flags) }?.cast::<u8>();
        assert!(!ptr.is_null());
        Ok(Self {
            ptr,
            bytes,
            kind,
            ctx: ctx.clone(),
        })
    }

    /// Size in bytes.
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Allocation kind.
    pub fn kind(&self) -> PinnedKind {
        self.kind
    }

    /// The buffer as `f64` values (`len_bytes / 8` of them).
    ///
    /// Reading an [`PinnedKind::WriteCombined`] buffer on the host is very slow; write it only.
    pub fn as_f64(&self) -> &[f64] {
        // SAFETY: the allocation is live, 8-aligned (page aligned) and `bytes / 8` long.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<f64>(), self.bytes / 8) }
    }

    /// Mutable `f64` view.
    pub fn as_f64_mut(&mut self) -> &mut [f64] {
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<f64>(), self.bytes / 8) }
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: `ptr` came from `malloc_host`; any copy using it is waited by `Pending`.
        let _ = unsafe { result::free_host(self.ptr.cast()) };
    }
}

impl std::fmt::Debug for PinnedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PinnedBuf({} bytes, {:?})", self.bytes, self.kind)
    }
}

/// Size-class pool of pinned buffers (power-of-two classes, one free list per kind).
pub struct PinnedPool {
    ctx: Arc<CudaContext>,
    free: Mutex<Vec<PinnedBuf>>,
}

impl PinnedPool {
    /// Empty pool for `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Self {
        Self {
            ctx: ctx.clone(),
            free: Mutex::new(Vec::new()),
        }
    }

    /// A buffer of at least `bytes` of `kind`, reused when possible.
    pub fn take(&self, bytes: usize, kind: PinnedKind) -> Result<PinnedBuf, DriverError> {
        let class = bytes.max(64).next_power_of_two();
        {
            let mut free = self.free.lock().expect("pinned pool lock");
            if let Some(i) = free.iter().position(|b| b.kind == kind && b.bytes == class) {
                return Ok(free.swap_remove(i));
            }
        }
        PinnedBuf::new(&self.ctx, class, kind)
    }

    /// Return a buffer for reuse.
    pub fn put(&self, buf: PinnedBuf) {
        self.free.lock().expect("pinned pool lock").push(buf);
    }

    /// Buffers currently idle in the pool.
    pub fn idle(&self) -> usize {
        self.free.lock().expect("pinned pool lock").len()
    }
}

/// A recorded point on a stream (CUDA event).
pub struct TransferEvent(CudaEvent);

impl TransferEvent {
    /// Block the host until the event completed.
    pub fn wait_host(&self) -> Result<(), DriverError> {
        self.0.synchronize()
    }

    /// Whether the event already completed (non-blocking).
    pub fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    /// Milliseconds from `self` to `later` (both on completed events).
    pub fn elapsed_ms(&self, later: &TransferEvent) -> Result<f32, DriverError> {
        self.0.elapsed_ms(&later.0)
    }
}

/// An in-flight copy. Borrows the host buffer until the copy completed; dropping waits.
#[must_use = "a Pending waits for the copy when dropped; call wait() or record() explicitly"]
pub struct Pending<'a> {
    event: Option<CudaEvent>,
    done: bool,
    _host: std::marker::PhantomData<&'a ()>,
}

impl Pending<'_> {
    /// Block until the copy completed.
    pub fn wait(mut self) -> Result<(), DriverError> {
        self.done = true;
        self.event.as_ref().expect("event present").synchronize()
    }

    /// Non-blocking completion check.
    pub fn is_complete(&self) -> bool {
        self.event.as_ref().is_none_or(|e| e.is_complete())
    }

    /// Release the host borrow and return the completion event without waiting.
    ///
    /// # Safety
    ///
    /// The caller must not modify (upload) or read (download) the host buffer, or free it,
    /// until the returned event completed.
    pub unsafe fn detach(mut self) -> TransferEvent {
        self.done = true;
        TransferEvent(self.event.take().expect("event present"))
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if !self.done {
            if let Some(e) = self.event.as_ref() {
                let _ = e.synchronize();
            }
        }
    }
}

/// One stream used for host-device copies.
pub struct TransferStream {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl TransferStream {
    /// A new non-blocking stream on `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        Ok(Self {
            ctx: ctx.clone(),
            stream: ctx.new_stream()?,
        })
    }

    /// The underlying cudarc stream (launch kernels on it to order them after copies).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Enqueue `host -> dev` (copies `host.len_bytes()` bytes; `dev` must hold at least that).
    ///
    /// The copy is asynchronous only for pinned `host`, which this type guarantees.
    pub fn upload_async<'a>(
        &self,
        host: &'a PinnedBuf,
        dev: &'a mut CudaSlice<f64>,
    ) -> Result<Pending<'a>, DriverError> {
        let bytes = host.bytes.min(dev.len() * 8);
        let (ptr, _guard) = dev.device_ptr_mut(&self.stream);
        // SAFETY: `ptr` is a live device allocation of at least `bytes`; `host` is pinned and
        // stays borrowed by the returned `Pending` until the copy completed.
        unsafe { self.upload_raw(host.as_ptr(), ptr, bytes) }
    }

    /// Enqueue `dev -> host` (`host` must hold at least `dev.len() * 8` bytes).
    pub fn download_async<'a>(
        &self,
        dev: &'a CudaSlice<f64>,
        host: &'a mut PinnedBuf,
    ) -> Result<Pending<'a>, DriverError> {
        let bytes = host.bytes.min(dev.len() * 8);
        let (ptr, _guard) = dev.device_ptr(&self.stream);
        // SAFETY: as above; `host` is exclusively borrowed until the copy completed.
        unsafe { self.download_raw(ptr, host.ptr, bytes) }
    }

    /// Raw host-to-device copy.
    ///
    /// # Safety
    ///
    /// `host` must be pinned memory valid for `bytes` bytes and `dev` a device allocation valid
    /// for `bytes` bytes; neither may be freed or (for `host`) modified before the returned
    /// `Pending` completes.
    pub unsafe fn upload_raw<'a>(
        &self,
        host: *const u8,
        dev: sys::CUdeviceptr,
        bytes: usize,
    ) -> Result<Pending<'a>, DriverError> {
        self.ctx.bind_to_thread()?;
        let src = std::slice::from_raw_parts(host, bytes);
        result::memcpy_htod_async(dev, src, self.stream.cu_stream())?;
        self.pending()
    }

    /// Raw device-to-host copy.
    ///
    /// # Safety
    ///
    /// As [`upload_raw`](Self::upload_raw), with `host` writable.
    pub unsafe fn download_raw<'a>(
        &self,
        dev: sys::CUdeviceptr,
        host: *mut u8,
        bytes: usize,
    ) -> Result<Pending<'a>, DriverError> {
        self.ctx.bind_to_thread()?;
        let dst = std::slice::from_raw_parts_mut(host, bytes);
        result::memcpy_dtoh_async(dst, dev, self.stream.cu_stream())?;
        self.pending()
    }

    fn pending<'a>(&self) -> Result<Pending<'a>, DriverError> {
        let event = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        event.record(&self.stream)?;
        Ok(Pending {
            event: Some(event),
            done: false,
            _host: std::marker::PhantomData,
        })
    }

    /// Record an event at the current end of this stream.
    pub fn record(&self) -> Result<TransferEvent, DriverError> {
        let event = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        event.record(&self.stream)?;
        Ok(TransferEvent(event))
    }

    /// Make this stream wait (on the device, not the host) for `event`.
    pub fn wait_event(&self, event: &TransferEvent) -> Result<(), DriverError> {
        self.stream.wait(&event.0)
    }

    /// Make this stream wait for the completion event of an in-flight copy.
    pub fn wait_pending(&self, pending: &Pending<'_>) -> Result<(), DriverError> {
        self.stream
            .wait(pending.event.as_ref().expect("event present"))
    }

    /// Block the host until everything enqueued so far completed.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }
}
