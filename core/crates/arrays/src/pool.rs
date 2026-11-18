//! Thread pools by size, made once and kept: a read asks for `threads` and runs its
//! chunk decoding on that pool (zarrs decodes in parallel on the current rayon pool).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::ReadError;

static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

/// The pool of `threads` threads (at least 1).
pub fn pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, ReadError> {
    let threads = threads.max(1);
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| ReadError::new("the thread pools' lock is poisoned".to_string()))?;
    if let Some(p) = pools.get(&threads) {
        return Ok(p.clone());
    }
    let p = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("dyna-zarr-core-{threads}-{i}"))
        .build()
        .map_err(|e| ReadError::new(format!("cannot start {threads} threads: {e}")))?;
    let p = Arc::new(p);
    pools.insert(threads, p.clone());
    Ok(p)
}
