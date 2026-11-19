//! Threads (Workplan decision DR-5): every read runs on ONE pool, and its `threads` cap
//! how many of its pieces hold memory at once.
//!
//! - A pool is a fixed set of threads. A write (or a direct read) runs all its work on
//!   one pool: the region loop, zarrs' chunk decoding and the per-row computing, balanced
//!   by work stealing. Nested work never starts a second pool: on a pool's thread it
//!   stays in that pool ([`install`]). So the threads busy at once never exceed the
//!   pool's size.
//! - A read's `threads` caps its memory: pieces that hold buffers run in at most
//!   `threads` groups ([`pieces`]), each group one piece after another, whatever the
//!   scheduling. Work that holds no buffer of its own (output rows) uses whatever threads
//!   of the pool are free.
//! - A read starts ([`enter`]) on the pool given, else on the shared pool of `threads`,
//!   else, with one thread, right on the calling thread. zarrs then decodes one chunk at
//!   a time (no minimum of chunks at once). On sharded arrays some of its paths hand
//!   even that one task to rayon's global pool while the caller waits, so a one-thread
//!   read still keeps one thread busy.
//! - No pool belongs to a thread: a one-thread read needs no thread of its own, and
//!   callers that come and go (io.write's region workers) leave no pools behind.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
pub use rayon::ThreadPool;

use crate::ReadError;

static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

/// A new pool of `threads` threads (at least 1), not shared with anyone.
pub fn new_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, ReadError> {
    let threads = threads.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("dyna-zarr-core-{threads}-{i}"))
        .build()
        .map(Arc::new)
        .map_err(|e| ReadError::new(format!("cannot start {threads} threads: {e}")))
}

/// The shared pool of `threads` threads (at least 1), made once and kept: direct reads
/// that ask for the same number share it.
pub fn pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, ReadError> {
    let threads = threads.max(1);
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| ReadError::new("the thread pools' lock is poisoned".to_string()))?;
    if let Some(p) = pools.get(&threads) {
        return Ok(p.clone());
    }
    let p = new_pool(threads)?;
    pools.insert(threads, p.clone());
    Ok(p)
}

/// Run a read of `threads` from outside the engine: on `given`; else on the shared
/// pool of `threads`; else (one thread) right here. Already on a pool's thread: right
/// here.
pub fn enter<R: Send>(
    threads: usize,
    given: Option<&rayon::ThreadPool>,
    f: impl FnOnce() -> R + Send,
) -> Result<R, ReadError> {
    if rayon::current_thread_index().is_some() {
        return Ok(f());
    }
    Ok(match given {
        Some(p) => p.install(f),
        None if threads > 1 => pool(threads)?.install(f),
        None => f(),
    })
}

/// Run `f` where parallel work of `threads` may spread: in place on a pool's thread (the
/// caller's pool), on the shared pool of `threads` otherwise.
pub fn install<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> Result<R, ReadError> {
    if rayon::current_thread_index().is_some() {
        Ok(f())
    } else {
        Ok(pool(threads)?.install(f))
    }
}

/// How `n` pieces run with `threads`: (pieces open at once, threads of each piece).
pub fn piece_plan(n: usize, threads: usize) -> (usize, usize) {
    if threads <= 1 || n <= 1 {
        return (1.min(n), threads.max(1));
    }
    let open = threads.min(n);
    (open, (threads / open).max(1))
}

/// Run `f(k, threads_of_piece)` for every piece `k` of `n`. At most `open` pieces run
/// at once ([`piece_plan`]): `open` workers each take the next piece until none is left
/// (a slow piece does not hold up the others), so no scheduling can open more. One
/// piece, or one thread: one after another with all threads.
pub fn pieces(
    n: usize,
    threads: usize,
    f: impl Fn(usize, usize) -> Result<(), ReadError> + Sync,
) -> Result<(), ReadError> {
    let (open, each) = piece_plan(n, threads);
    if open <= 1 {
        return (0..n).try_for_each(|k| f(k, each));
    }
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    install(threads, || {
        (0..open).into_par_iter().try_for_each(|_| loop {
            if failed.load(Ordering::Relaxed) {
                return Ok(());
            }
            let k = next.fetch_add(1, Ordering::Relaxed);
            if k >= n {
                return Ok(());
            }
            if let Err(e) = f(k, each) {
                failed.store(true, Ordering::Relaxed);
                return Err(e);
            }
        })
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_most_threads_pieces_open() {
        let open = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        pieces(64, 3, |_, each| {
            assert_eq!(each, 1);
            let now = open.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(2));
            open.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert!(most.load(Ordering::SeqCst) <= 3);
    }

    #[test]
    fn plans() {
        assert_eq!(piece_plan(0, 8), (0, 8));
        assert_eq!(piece_plan(1, 8), (1, 8));
        assert_eq!(piece_plan(2, 8), (2, 4));
        assert_eq!(piece_plan(100, 8), (8, 1));
        assert_eq!(piece_plan(100, 1), (1, 1));
    }

    #[test]
    fn errors_stop_the_rest() {
        let ran = AtomicUsize::new(0);
        let r = pieces(1000, 4, |i, _| {
            ran.fetch_add(1, Ordering::SeqCst);
            if i == 3 {
                Err(ReadError::new("boom".to_string()))
            } else {
                Ok(())
            }
        });
        assert!(r.is_err());
        assert!(ran.load(Ordering::SeqCst) < 1000);
    }
}
