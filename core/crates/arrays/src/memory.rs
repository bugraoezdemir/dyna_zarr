//! The engine's heap, counted (Workplan decision DR-5): bytes in use and the most in
//! use since the last reset, so tests and writes can compare a read's real peak with
//! its [`cost`](crate::Source::cost). The binary that uses the engine installs
//! [`Counting`] as its global allocator (the Python module does, over mimalloc); without
//! it the counters stay at 0.
//!
//! The counts are process-wide: every thread's allocations, the engine's and anything
//! else in the same binary. A destination buffer the caller owns (a NumPy array) is not
//! allocated here and is not counted, as a cost does not include it.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

static IN_USE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// A global allocator that counts the bytes `A` hands out.
pub struct Counting<A>(pub A);

fn grew(by: usize) {
    let now = IN_USE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

fn shrank(by: usize) {
    IN_USE.fetch_sub(by, Ordering::Relaxed);
}

// SAFETY: every call is forwarded to `A` unchanged; only counters are added.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(ptr, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                shrank(layout.size() - new_size);
            }
        }
        p
    }
}

/// Bytes allocated now.
pub fn in_use() -> usize {
    IN_USE.load(Ordering::Relaxed)
}

/// The most bytes allocated at once since the last [`reset_peak`].
pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Start a new peak from what is allocated now; returns that.
pub fn reset_peak() -> usize {
    let now = IN_USE.load(Ordering::Relaxed);
    PEAK.store(now, Ordering::Relaxed);
    now
}
