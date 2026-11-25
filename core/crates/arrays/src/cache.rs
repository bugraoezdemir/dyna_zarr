//! A write's cache of decoded chunks (Workplan DR6, 2026-11-25).
//!
//! A filter or a resampling reads a halo around every region, and the halo's elements
//! lie in the neighbours' chunks: without a cache, every chunk at a region's border is
//! decoded again for each region that touches it (2.5 to 3.5 decodes per chunk for a
//! Gaussian downscale of real pyramids). With one, each is decoded about once.
//!
//! - A cache belongs to one pool (one write, see [`crate::pool::new_pool_cached`]): its
//!   threads see it ([`current`]), reads elsewhere do not, so a read outside a write
//!   behaves as without a cache, and the cache goes with the write's pool.
//! - It holds at most `limit` bytes of decoded chunks, least recently used dropped
//!   first. A write counts the limit once in its memory budget (decision DR-5: it is
//!   shared by all of the write's regions, so no region's cost holds it).
//! - Two reads that need the same chunk at once decode it once: the second waits.
//! - Only arrays that ask for it use the cache ([`ChunkCache::get_or_decode`]): Zarr
//!   arrays, per read unit (the inner chunk when sharded).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::ReadError;

/// One decoded unit: its bytes, once some read has decoded them.
type Slot = Arc<Mutex<Option<Arc<Vec<u8>>>>>;

/// (the array's id, the unit's index on the unit grid)
type Key = (u64, Vec<u64>);

struct Entry {
    slot: Slot,
    bytes: u64,
    last: u64,
}

#[derive(Default)]
struct State {
    map: HashMap<Key, Entry>,
    /// bytes of the decoded units held
    used: u64,
    tick: u64,
    hits: u64,
    decodes: u64,
}

/// Decoded chunks of one write, at most `limit` bytes.
pub struct ChunkCache {
    limit: u64,
    state: Mutex<State>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A new id for an array whose chunks may be cached (unique in the process).
pub(crate) fn new_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<ChunkCache>>> = const { RefCell::new(None) };
}

/// The cache of the pool this thread belongs to, if it has one.
pub(crate) fn current() -> Option<Arc<ChunkCache>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Make `cache` this thread's (a pool's threads call it when they start).
pub(crate) fn set_current(cache: Option<Arc<ChunkCache>>) {
    CURRENT.with(|c| *c.borrow_mut() = cache);
}

impl ChunkCache {
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit,
            state: Mutex::new(State::default()),
        })
    }

    /// The most bytes of decoded units it holds.
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// (units served from the cache, units decoded) so far.
    pub fn counts(&self) -> (u64, u64) {
        self.state
            .lock()
            .map(|s| (s.hits, s.decodes))
            .unwrap_or((0, 0))
    }

    /// The unit `key` of `bytes` bytes: from the cache, or `decode`d now and kept. A
    /// unit larger than the whole limit is decoded and not kept.
    pub(crate) fn get_or_decode(
        &self,
        key: Key,
        bytes: u64,
        decode: impl FnOnce() -> Result<Vec<u8>, ReadError>,
    ) -> Result<Arc<Vec<u8>>, ReadError> {
        if bytes > self.limit {
            self.count(false);
            return decode().map(Arc::new);
        }
        let slot = {
            let mut s = self.lock()?;
            s.tick += 1;
            let tick = s.tick;
            let e = s.map.entry(key).or_insert_with(|| Entry {
                slot: Arc::new(Mutex::new(None)),
                bytes,
                last: tick,
            });
            e.last = tick;
            e.slot.clone()
        };
        // the unit's own lock: a second read of it waits here while the first decodes
        let mut held = slot
            .lock()
            .map_err(|_| ReadError::new("a cached chunk's lock is poisoned".to_string()))?;
        if let Some(d) = held.as_ref() {
            let d = d.clone();
            drop(held);
            self.count(true);
            return Ok(d);
        }
        let d = Arc::new(decode()?);
        *held = Some(d.clone());
        drop(held);
        self.count(false);
        let mut s = self.lock()?;
        s.used += bytes;
        self.evict(&mut s, &slot);
        Ok(d)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>, ReadError> {
        self.state
            .lock()
            .map_err(|_| ReadError::new("the chunk cache's lock is poisoned".to_string()))
    }

    fn count(&self, hit: bool) {
        if let Ok(mut s) = self.state.lock() {
            if hit {
                s.hits += 1;
            } else {
                s.decodes += 1;
            }
        }
    }

    /// Drop the least recently used decoded units (never `keep`, just decoded) until
    /// the decoded bytes fit the limit. Units still being decoded are not counted yet
    /// and stay.
    fn evict(&self, s: &mut State, keep: &Slot) {
        while s.used > self.limit {
            let oldest = s
                .map
                .iter()
                .filter(|(_, e)| !Arc::ptr_eq(&e.slot, keep))
                .filter(|(_, e)| e.slot.try_lock().map(|g| g.is_some()).unwrap_or(false))
                .min_by_key(|(_, e)| e.last)
                .map(|(k, e)| (k.clone(), e.bytes));
            match oldest {
                Some((k, b)) => {
                    s.map.remove(&k);
                    s.used -= b;
                }
                None => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn decodes_once_and_drops_the_oldest() {
        let c = ChunkCache::new(20);
        let n = AtomicUsize::new(0);
        let dec = |v: u8| {
            n.fetch_add(1, Ordering::SeqCst);
            Ok(vec![v; 10])
        };
        let a = c.get_or_decode((1, vec![0]), 10, || dec(1)).unwrap();
        let b = c.get_or_decode((1, vec![0]), 10, || dec(9)).unwrap();
        assert_eq!((a[0], b[0], n.load(Ordering::SeqCst)), (1, 1, 1));
        c.get_or_decode((1, vec![1]), 10, || dec(2)).unwrap();
        c.get_or_decode((1, vec![2]), 10, || dec(3)).unwrap(); // drops [0]
        assert_eq!(n.load(Ordering::SeqCst), 3);
        c.get_or_decode((1, vec![0]), 10, || dec(4)).unwrap();
        assert_eq!(n.load(Ordering::SeqCst), 4);
        assert!(c.lock().unwrap().used <= 20);
        assert_eq!(c.counts(), (1, 4));
    }

    #[test]
    fn too_large_is_not_kept_and_errors_are_retried() {
        let c = ChunkCache::new(8);
        c.get_or_decode((1, vec![0]), 16, || Ok(vec![0; 16])).unwrap();
        assert_eq!(c.lock().unwrap().used, 0);
        let e = c.get_or_decode((1, vec![1]), 4, || Err(ReadError::new("x".into())));
        assert!(e.is_err());
        let ok = c.get_or_decode((1, vec![1]), 4, || Ok(vec![7; 4])).unwrap();
        assert_eq!(ok[0], 7);
    }

    #[test]
    fn concurrent_reads_of_one_unit_decode_it_once() {
        let c = ChunkCache::new(1 << 20);
        let n = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    c.get_or_decode((3, vec![5, 5]), 100, || {
                        n.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        Ok(vec![1; 100])
                    })
                    .unwrap();
                });
            }
        });
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }
}
