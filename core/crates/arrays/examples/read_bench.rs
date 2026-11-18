//! Read speed of the arrays crate alone, without Python: the same requests as
//! benchmarks/bench_engine_factors.py, so the bindings' share can be told apart.
//!
//!     cargo run --release --example read_bench -p dyna-zarr-arrays -- <array folder> <threads>
//!
//! It allocates with mimalloc, as the Python module does. Build with
//! `RUSTFLAGS="--cfg system_alloc"` to measure the system allocator instead.

#[cfg(not(system_alloc))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::Path;
use std::time::Instant;

use dyna_zarr_arrays::{read, Source, ZarrArray};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let threads: usize = args.get(2).map_or(1, |t| t.parse().unwrap());
    let a = ZarrArray::open(path).unwrap();
    let shape = a.shape().to_vec();
    let n = shape.len();
    // a plane region 512 x 512, one 64^3 brick, a whole plane, the whole array
    let mid: Vec<u64> = shape[..n - 2].iter().map(|s| s / 2).collect();
    let mut plane_s = mid.clone();
    plane_s.extend([20, 200]);
    let mut plane_e: Vec<u64> = mid.iter().map(|m| m + 1).collect();
    plane_e.extend([(20 + 512).min(shape[n - 2]), (200 + 512).min(shape[n - 1])]);
    let mut brick_s: Vec<u64> = shape[..n - 3].iter().map(|_| 0).collect();
    brick_s.extend([10, 100, 100]);
    let mut brick_e: Vec<u64> = shape[..n - 3].iter().map(|_| 1).collect();
    brick_e.extend([74.min(shape[n - 3]), 164, 164]);
    let mut whole_s = mid.clone();
    whole_s.extend([0, 0]);
    let mut whole_e: Vec<u64> = mid.iter().map(|m| m + 1).collect();
    whole_e.extend([shape[n - 2], shape[n - 1]]);
    let cases = [
        ("plane 512x512", plane_s, plane_e, 50),
        ("brick 64^3", brick_s, brick_e, 50),
        ("whole plane", whole_s, whole_e, 20),
        ("whole array", vec![0; n], shape.clone(), 5),
    ];
    for (name, s, e, reps) in cases {
        let mut times = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            let buf = read(&a, &s, &e, threads).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(buf);
        }
        println!("{name:14} threads {threads:2}: {:8.3} ms", median(times));
    }
}
