//! Downsampling by integer factors per axis (Workplan WP15, task DR5): lazy views whose
//! levels have the shapes and values ome_zarr_pyramid writes.
//!
//! Every level is ceil-sized (`ceil(n / f)` per axis), as `tensorstore.downsample`:
//!
//! - **stride**: elements `0, f, 2f, ...` ([`stride`], a [`Slice`] with steps).
//! - **mean**: the mean of each window `[o f, o f + f)`; a partial window at the far edge
//!   is averaged over the elements it has. Integers (and bool, as 0 / 1) are summed
//!   exactly and rounded half to even; float32 / float64 are summed in their own type
//!   in the window's C order, float16 in f64; complex parts separately. Exactly
//!   `tensorstore.downsample(..., method="mean")`.
//! - **gaussian**: a Gaussian low-pass, then every f-th sample (`0, f, 2f, ...`, so a
//!   level is placed as a stride level). The filter is scipy.ndimage.gaussian_filter's:
//!   the kernel is cut at `int(truncate * sigma + 0.5)` and normalised; edge modes
//!   reflect (default), mirror, nearest, constant. Computed in f64 throughout and
//!   rounded once (integers half to even), where scipy would round after every axis.
//!   Only the kept samples are computed: each axis is filtered and decimated in turn.
//!   Default sigma: `f / 3` per downscaled axis (scikit-image's pyramid_reduce), 0 on
//!   the others.
//!
//! Asked for a region, a view reads only the input box that region needs, splitting a
//! request whose box would be too large first, and computes output rows in parallel.

use rayon::prelude::*;

use crate::blocks::Blocks;
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::{install, piece_plan, pieces};
use crate::source::{bytes_of, read_vec, region, Source, SourceRef, SMALL};
use crate::views::Slice;
use crate::ReadError;

/// Input bytes one read may need before the request is split.
pub const DEFAULT_MAX_READ: u64 = 256 << 20;
const MAX_AXES: usize = 8;

fn levels(src: &dyn Source, factors: &[u64]) -> Result<Vec<u64>, ReadError> {
    let n = src.shape().len();
    if factors.len() != n || factors.contains(&0) {
        return Err(ReadError::new(format!(
            "factors {factors:?}: one integer >= 1 per axis of {:?}",
            src.shape()
        )));
    }
    if n == 0 || n > MAX_AXES {
        return Err(ReadError::new(format!(
            "downsampling takes 1 to {MAX_AXES} axes, not {n}"
        )));
    }
    Ok((0..n)
        .map(|i| src.shape()[i].div_ceil(factors[i]))
        .collect())
}

/// `src`'s storage grid divided by `factors` (at least 1 per axis), or None.
pub(crate) fn scaled_grid(src: &dyn Source, factors: &[u64]) -> Option<Vec<u64>> {
    src.grid()
        .map(|g| g.iter().zip(factors).map(|(c, f)| (c / f).max(1)).collect())
}

/// Elements `0, f, 2f, ...` per axis (method "stride" / ozp's "simple").
pub fn stride(src: SourceRef, factors: &[u64]) -> Result<Slice, ReadError> {
    levels(src.as_ref(), factors)?;
    let n = factors.len();
    Slice::new(src.clone(), &vec![0; n], src.shape(), factors)
}

/// Halve the output region along its longest axis while its input box is too large;
/// call `run(start, region, out)` on each piece.
pub(crate) fn split<F>(
    start: &[u64],
    r: &[u64],
    out: &Out,
    max_read: u64,
    box_bytes: &dyn Fn(&[u64], &[u64]) -> u64,
    run: &F,
) -> Result<(), ReadError>
where
    F: Fn(&[u64], &[u64], &Out) -> Result<(), ReadError>,
{
    if r.contains(&0) {
        return Ok(());
    }
    if box_bytes(start, r) > max_read && r.iter().any(|x| *x > 1) {
        let n = r.len();
        let ax = (0..n).max_by_key(|&i| r[i]).unwrap_or(0);
        let half = r[ax] / 2;
        let mut r1 = r.to_vec();
        r1[ax] = half;
        split(start, &r1, out, max_read, box_bytes, run)?;
        let mut s2 = start.to_vec();
        s2[ax] += half;
        let mut r2 = r.to_vec();
        r2[ax] -= half;
        let mut at = out.at.clone();
        at[ax] += half;
        return split(&s2, &r2, &out.moved(at), max_read, box_bytes, run);
    }
    run(start, r, out)
}

/// The cost of one piece `(start, region)`.
type PieceCost<'a> = dyn Fn(&[u64], &[u64]) -> Result<u64, ReadError> + 'a;

/// The cost of a [`split`] read: the same halving, then the most costly piece (pieces
/// run one after another).
pub(crate) fn split_cost(
    start: &[u64],
    r: &[u64],
    max_read: u64,
    box_bytes: &dyn Fn(&[u64], &[u64]) -> u64,
    cost: &PieceCost,
) -> Result<u64, ReadError> {
    if r.contains(&0) {
        return Ok(0);
    }
    if box_bytes(start, r) > max_read && r.iter().any(|x| *x > 1) {
        let ax = (0..r.len()).max_by_key(|&i| r[i]).unwrap_or(0);
        let half = r[ax] / 2;
        let mut r1 = r.to_vec();
        r1[ax] = half;
        let mut s2 = start.to_vec();
        s2[ax] += half;
        let mut r2 = r.to_vec();
        r2[ax] -= half;
        return Ok(split_cost(start, &r1, max_read, box_bytes, cost)?
            .max(split_cost(&s2, &r2, max_read, box_bytes, cost)?));
    }
    cost(start, r)
}

/// Run `row(outer, dst)` for every row of a region of `r` elements (the last axis
/// contiguous in `out`), in parallel when `threads > 1`.
fn for_rows<F>(r: &[u64], out: &Out, threads: usize, row: F) -> Result<(), ReadError>
where
    F: Fn(&[u64], &mut [u8]) + Sync,
{
    let n = r.len();
    let rows: usize = r[..n - 1].iter().product::<u64>() as usize;
    let inner = r[n - 1];
    let one = |k: usize| {
        let mut outer = [0u64; MAX_AXES];
        let mut rest = k as u64;
        for i in (0..n - 1).rev() {
            outer[i] = rest % r[i];
            rest /= r[i];
        }
        // SAFETY: each row is written by one task only, inside the checked region
        let dst = unsafe { out.row_mut(&outer[..n - 1], inner) };
        row(&outer[..n - 1], dst);
    };
    let min = (4096 / inner.max(1)).max(1) as usize;
    if threads > 1 && rows > 1 {
        // rows hold no buffer: they use whatever threads of the pool are free
        install(threads, || {
            (0..rows).into_par_iter().with_min_len(min).for_each(one)
        })?;
    } else {
        (0..rows).for_each(one);
    }
    Ok(())
}

// ------------------------------------------------------------------------- mean

/// The mean of each window of `factors` elements.
pub struct Mean {
    src: SourceRef,
    factors: Vec<u64>,
    shape: Vec<u64>,
    max_read: u64,
}

impl Mean {
    pub fn new(src: SourceRef, factors: &[u64]) -> Result<Self, ReadError> {
        let shape = levels(src.as_ref(), factors)?;
        Ok(Self {
            src,
            factors: factors.to_vec(),
            shape,
            max_read: DEFAULT_MAX_READ,
        })
    }

    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    fn input_box(&self, start: &[u64], r: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let n = r.len();
        let s = (0..n).map(|i| start[i] * self.factors[i]).collect();
        let e = (0..n)
            .map(|i| ((start[i] + r[i]) * self.factors[i]).min(self.src.shape()[i]))
            .collect();
        (s, e)
    }

    fn part<T: MeanPx>(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let (s, e) = self.input_box(start, r);
        let data = read_vec::<T>(self.src.as_ref(), &s, &e, threads)?;
        let n = r.len();
        let bshape: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
        let mut bstr = vec![1usize; n];
        for i in (0..n - 1).rev() {
            bstr[i] = bstr[i + 1] * bshape[i + 1] as usize;
        }
        let f = &self.factors;
        let item = std::mem::size_of::<T>();
        // a full window's element offsets from its first element, in C order (the
        // order the sum must run in: see mean_float)
        let mut window: Vec<usize> = vec![0];
        for i in 0..n {
            let (fi, stride) = (f[i] as usize, bstr[i]);
            window = window
                .iter()
                .flat_map(|w| (0..fi).map(move |j| w + j * stride))
                .collect();
        }
        let full_count = window.len() as u64;
        for_rows(r, out, threads, |outer, dst| {
            let mut lo = [0usize; MAX_AXES]; // window start in the box, per axis
            let mut len = [0usize; MAX_AXES];
            let mut full_outer = true;
            let mut base = 0usize;
            for i in 0..n - 1 {
                let o = (start[i] + outer[i]) * f[i];
                lo[i] = (o - s[i]) as usize;
                len[i] = (o + f[i]).min(e[i]) as usize - o as usize;
                full_outer &= len[i] == f[i] as usize;
                base += lo[i] * bstr[i];
            }
            for (k, el) in dst.chunks_exact_mut(item).enumerate() {
                let o = (start[n - 1] + k as u64) * f[n - 1];
                lo[n - 1] = (o - s[n - 1]) as usize;
                len[n - 1] = (o + f[n - 1]).min(e[n - 1]) as usize - o as usize;
                if full_outer && len[n - 1] == f[n - 1] as usize {
                    // a full window: its precomputed offsets
                    let first = base + lo[n - 1];
                    let mut acc = T::Acc::default();
                    for w in &window {
                        T::add(&mut acc, data[first + w]);
                    }
                    let v = T::finish(acc, full_count);
                    // SAFETY: `el` is one element of size_of::<T>() bytes
                    unsafe { (el.as_mut_ptr() as *mut T).write_unaligned(v) }
                    continue;
                }
                // a partial window at the far edge: walk what it has
                let mut acc = T::Acc::default();
                let mut count = 0u64;
                let mut idx = [0usize; MAX_AXES];
                'window: loop {
                    let off: usize = (0..n).map(|i| (lo[i] + idx[i]) * bstr[i]).sum();
                    T::add(&mut acc, data[off]);
                    count += 1;
                    let mut ax = n;
                    loop {
                        if ax == 0 {
                            break 'window;
                        }
                        ax -= 1;
                        idx[ax] += 1;
                        if idx[ax] < len[ax] {
                            break;
                        }
                        idx[ax] = 0;
                    }
                }
                let v = T::finish(acc, count);
                // SAFETY: `el` is one element of size_of::<T>() bytes
                unsafe { (el.as_mut_ptr() as *mut T).write_unaligned(v) }
            }
        })
    }
}

impl Source for Mean {
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    /// The source's storage grid in this level's elements: views above read in blocks
    /// of about one source chunk.
    fn grid(&self) -> Option<Vec<u64>> {
        scaled_grid(self.src.as_ref(), &self.factors)
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = region(&self.shape, start, stop)?;
        out.check(&r, self.dtype())?;
        let item = self.src.dtype().size() as u64;
        let bytes = |s: &[u64], r: &[u64]| {
            let (a, b) = self.input_box(s, r);
            (0..a.len()).map(|i| b[i] - a[i]).product::<u64>() * item
        };
        let run_on = |s: &[u64], r: &[u64], o: &Out, th: usize| match self.src.dtype() {
            DType::Bool => self.part::<Bool>(s, r, o, th),
            DType::Int8 => self.part::<i8>(s, r, o, th),
            DType::Int16 => self.part::<i16>(s, r, o, th),
            DType::Int32 => self.part::<i32>(s, r, o, th),
            DType::Int64 => self.part::<i64>(s, r, o, th),
            DType::UInt8 => self.part::<u8>(s, r, o, th),
            DType::UInt16 => self.part::<u16>(s, r, o, th),
            DType::UInt32 => self.part::<u32>(s, r, o, th),
            DType::UInt64 => self.part::<u64>(s, r, o, th),
            DType::Float16 => self.part::<half::f16>(s, r, o, th),
            DType::Float32 => self.part::<f32>(s, r, o, th),
            DType::Float64 => self.part::<f64>(s, r, o, th),
            DType::Complex64 => self.part::<[f32; 2]>(s, r, o, th),
            DType::Complex128 => self.part::<[f64; 2]>(s, r, o, th),
        };
        // Over a storage grid: blocks of about one input chunk each, at most `threads` at
        // once (each still in cache when it is averaged, as TensorStore streams chunk by
        // chunk). Otherwise: large pieces one after another, each read and averaged with
        // all threads.
        if let Some(blocks) = self.blocks(start, &r, threads) {
            // each open block gets its share of the read limit (a huge storage chunk is
            // split like any too-large request)
            let (open, _) = piece_plan(blocks.len(), threads);
            let limit = (self.max_read / open as u64).max(1);
            return pieces(blocks.len(), threads, |k, each| {
                let b = blocks.get(k);
                let s: Vec<u64> = (0..r.len()).map(|i| start[i] + b[i].0).collect();
                let br: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
                let at: Vec<u64> = (0..r.len()).map(|i| out.at[i] + b[i].0).collect();
                let run = |s: &[u64], r: &[u64], o: &Out| run_on(s, r, o, each);
                split(&s, &br, &out.moved(at), limit, &bytes, &run)
            });
        }
        let run = |s: &[u64], r: &[u64], o: &Out| run_on(s, r, o, threads);
        split(start, &r, out, self.max_read, &bytes, &run)
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        let bytes = |s: &[u64], r: &[u64]| self.box_bytes(s, r);
        if let Some(blocks) = self.blocks(start, &r, threads) {
            let (open, each) = piece_plan(blocks.len(), threads);
            let limit = (self.max_read / open as u64).max(1);
            let cost = |s: &[u64], r: &[u64]| self.part_cost(s, r, each);
            let mut most = 0u64;
            for b in blocks.sample() {
                let s: Vec<u64> = (0..r.len()).map(|i| start[i] + b[i].0).collect();
                let br: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
                most = most.max(split_cost(&s, &br, limit, &bytes, &cost)?);
            }
            return Ok(open as u64 * most);
        }
        let cost = |s: &[u64], r: &[u64]| self.part_cost(s, r, threads);
        split_cost(start, &r, self.max_read, &bytes, &cost)
    }
}

impl Mean {
    /// The blocks of a read over a source with a storage grid (about one input chunk
    /// each: read and averaged while still in cache, one after another on one thread),
    /// or None: the read is split by size instead.
    fn blocks(&self, start: &[u64], r: &[u64], _threads: usize) -> Option<Blocks> {
        if r.contains(&0) {
            return None;
        }
        if self.src.reads_halo() {
            return None; // over a filter: large pieces, not one per chunk (see split)
        }
        let grid = self.src.grid()?;
        let unit: Vec<u64> = (0..r.len())
            .map(|i| (grid[i] / self.factors[i]).max(1))
            .collect();
        let blocks = Blocks::aligned(start, r, &unit);
        (blocks.len() > 1).then_some(blocks)
    }

    fn box_bytes(&self, start: &[u64], r: &[u64]) -> u64 {
        let (a, b) = self.input_box(start, r);
        let shape: Vec<u64> = (0..a.len()).map(|i| b[i] - a[i]).collect();
        bytes_of(&shape, self.src.dtype().size())
    }

    /// One piece: its input box, a full window's offsets, and reading the box.
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let (s, e) = self.input_box(start, r);
        let window: u64 = self.factors.iter().product::<u64>() * 8;
        Ok(self.box_bytes(start, r) + window + SMALL + self.src.cost(&s, &e, threads)?)
    }
}

/// bool as one byte, averaged as 0 / 1.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
struct Bool(u8);

/// An element type with a mean: an accumulator and the final rounding.
trait MeanPx: Copy + Default + Send + Sync {
    type Acc: Default;
    fn add(acc: &mut Self::Acc, v: Self);
    fn finish(acc: Self::Acc, count: u64) -> Self;
}

/// `sum / count` rounded half to even (exact, any sign).
fn div_half_even(sum: i128, count: u64) -> i128 {
    let c = count as i128;
    let q = sum.div_euclid(c);
    let r = sum.rem_euclid(c);
    match (2 * r).cmp(&c) {
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Equal => q + (q & 1),
    }
}

/// [`div_half_even`] in 64 bits: an i128 division is a slow library call on x86, one per
/// output element.
fn div_half_even_64(sum: i64, count: u64) -> i64 {
    let c = count as i64;
    let q = sum.div_euclid(c);
    let r = sum.rem_euclid(c);
    match (2 * r).cmp(&c) {
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Equal => q + (q & 1),
    }
}

// Integers sum exactly: in i64 up to 32 bits (exact for any window under 2^31
// elements), in i128 for 64-bit types.
macro_rules! mean_int {
    (i64: $($t:ty),*) => {$(
        impl MeanPx for $t {
            type Acc = i64;
            fn add(acc: &mut i64, v: Self) {
                *acc += v as i64;
            }
            fn finish(acc: i64, count: u64) -> Self {
                div_half_even_64(acc, count) as $t
            }
        }
    )*};
    (i128: $($t:ty),*) => {$(
        impl MeanPx for $t {
            type Acc = i128;
            fn add(acc: &mut i128, v: Self) {
                *acc += v as i128;
            }
            fn finish(acc: i128, count: u64) -> Self {
                div_half_even(acc, count) as $t
            }
        }
    )*};
}
mean_int!(i64: i8, i16, i32, u8, u16, u32);
mean_int!(i128: i64, u64);

impl MeanPx for Bool {
    type Acc = i64;
    fn add(acc: &mut i64, v: Self) {
        *acc += i64::from(v.0 != 0);
    }
    fn finish(acc: i64, count: u64) -> Self {
        Bool(u8::from(div_half_even_64(acc, count) != 0))
    }
}

// Floats are summed in their own type, window elements in C order, then divided by the
// count in that type: what tensorstore.downsample does (checked on random float32
// windows: 60 of 60 equal; an f64 sum differs in the last bits).
macro_rules! mean_float {
    ($($t:ty),*) => {$(
        impl MeanPx for $t {
            type Acc = $t;
            fn add(acc: &mut $t, v: Self) {
                *acc += v;
            }
            fn finish(acc: $t, count: u64) -> Self {
                acc / count as $t
            }
        }
    )*};
}
mean_float!(f32, f64);

impl MeanPx for half::f16 {
    type Acc = f64;
    fn add(acc: &mut f64, v: Self) {
        *acc += v.to_f64();
    }
    fn finish(acc: f64, count: u64) -> Self {
        half::f16::from_f64(acc / count as f64)
    }
}

macro_rules! mean_complex {
    ($($t:ty),*) => {$(
        impl MeanPx for [$t; 2] {
            type Acc = ($t, $t);
            fn add(acc: &mut ($t, $t), v: Self) {
                acc.0 += v[0];
                acc.1 += v[1];
            }
            fn finish(acc: ($t, $t), count: u64) -> Self {
                [acc.0 / count as $t, acc.1 / count as $t]
            }
        }
    )*};
}
mean_complex!(f32, f64);

// --------------------------------------------------------------------- gaussian

/// How the Gaussian filter extends the input beyond its edges (scipy.ndimage's modes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boundary {
    /// d c b a | a b c d | d c b a (scipy's default)
    Reflect,
    /// d c b | a b c d | c b a
    Mirror,
    /// a a a | a b c d | d d d
    Nearest,
    /// k k k | a b c d | k k k
    Constant(f64),
}

impl Boundary {
    pub fn parse(name: &str, cval: f64) -> Option<Self> {
        match name {
            "reflect" => Some(Self::Reflect),
            "mirror" => Some(Self::Mirror),
            "nearest" => Some(Self::Nearest),
            "constant" => Some(Self::Constant(cval)),
            _ => None,
        }
    }

    /// The index inside `0 .. n` that position `p` reads, or None (constant: the fill).
    fn map(self, p: i64, n: i64) -> Option<i64> {
        if (0..n).contains(&p) {
            return Some(p);
        }
        match self {
            Self::Constant(_) => None,
            Self::Nearest => Some(p.clamp(0, n - 1)),
            Self::Reflect => {
                let period = 2 * n;
                let q = p.rem_euclid(period);
                Some(if q < n { q } else { period - 1 - q })
            }
            Self::Mirror => {
                if n == 1 {
                    return Some(0);
                }
                let period = 2 * n - 2;
                let q = p.rem_euclid(period);
                Some(if q < n { q } else { period - q })
            }
        }
    }
}

/// A Gaussian low-pass, then every `factors`-th sample.
pub struct Gaussian {
    src: SourceRef,
    factors: Vec<u64>,
    shape: Vec<u64>,
    kernels: Vec<Vec<f64>>,
    boundary: Boundary,
    max_read: u64,
}

impl Gaussian {
    /// `sigma`: one per axis (None: `f / 3` on downscaled axes, 0 on the others);
    /// `truncate`: the kernel's half-width in sigmas (scipy's default 4).
    pub fn new(
        src: SourceRef,
        factors: &[u64],
        sigma: Option<&[f64]>,
        boundary: Boundary,
        truncate: f64,
    ) -> Result<Self, ReadError> {
        let shape = levels(src.as_ref(), factors)?;
        let dtype = src.dtype();
        if dtype == DType::Bool || dtype.is_complex() {
            return Err(ReadError::new(format!(
                "Gaussian downsampling of {} is not supported: use stride or mean",
                dtype.name()
            )));
        }
        let n = factors.len();
        let sigma: Vec<f64> = match sigma {
            Some(s) if s.len() == n => s.to_vec(),
            Some(s) => return Err(ReadError::new(format!("{} sigmas for {n} axes", s.len()))),
            None => factors
                .iter()
                .map(|&f| if f > 1 { f as f64 / 3.0 } else { 0.0 })
                .collect(),
        };
        // NaN fails both checks too
        if sigma.iter().any(|s| s.is_nan() || *s < 0.0) || truncate.is_nan() || truncate <= 0.0 {
            return Err(ReadError::new(format!(
                "sigma {sigma:?} must be >= 0 and truncate {truncate} > 0"
            )));
        }
        // scipy.ndimage._gaussian_kernel1d with the radius gaussian_filter1d uses
        let kernels = sigma
            .iter()
            .map(|&sd| {
                if sd <= 1e-15 {
                    return vec![1.0];
                }
                let radius = (truncate * sd + 0.5) as i64;
                let w: Vec<f64> = (-radius..=radius)
                    .map(|x| (-0.5 / (sd * sd) * (x * x) as f64).exp())
                    .collect();
                let total: f64 = w.iter().sum();
                w.iter().map(|v| v / total).collect()
            })
            .collect();
        Ok(Self {
            src,
            factors: factors.to_vec(),
            shape,
            kernels,
            boundary,
            max_read: DEFAULT_MAX_READ,
        })
    }

    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    fn radius(&self, i: usize) -> u64 {
        (self.kernels[i].len() as u64 - 1) / 2
    }

    /// The input box an output region needs: its samples and the kernel's reach,
    /// clamped to the array (positions beyond it map back inside: see Boundary::map).
    fn input_box(&self, start: &[u64], r: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let n = r.len();
        let shape = self.src.shape();
        let s = (0..n)
            .map(|i| (start[i] * self.factors[i]).saturating_sub(self.radius(i)))
            .collect();
        let e = (0..n)
            .map(|i| ((start[i] + r[i] - 1) * self.factors[i] + self.radius(i) + 1).min(shape[i]))
            .collect();
        (s, e)
    }

    fn part<T: GaussPx>(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let n = r.len();
        let (s, e) = self.input_box(start, r);
        let raw = read_vec::<T>(self.src.as_ref(), &s, &e, threads)?;
        let mut cur: Vec<f64> = raw.iter().map(|v| v.val()).collect();
        drop(raw);
        let mut shape: Vec<usize> = (0..n).map(|i| (e[i] - s[i]) as usize).collect();
        let full = self.src.shape();
        // filter and decimate one axis at a time (axis 0 first, as scipy filters)
        for ax in 0..n {
            let k = &self.kernels[ax];
            let rad = (k.len() / 2) as i64;
            let fax = self.factors[ax] as i64;
            let outer: usize = shape[..ax].iter().product();
            let inner: usize = shape[ax + 1..].iter().product();
            let len = shape[ax];
            let m = r[ax] as usize;
            let (s0, n_ax) = (s[ax] as i64, full[ax] as i64);
            let fill = match self.boundary {
                Boundary::Constant(c) => c,
                _ => 0.0,
            };
            // for every kept sample along this axis, the box positions (or None: fill)
            // the kernel reads, with their weights
            let taps: Vec<Vec<(Option<usize>, f64)>> = (0..m)
                .map(|q| {
                    let c = (start[ax] as i64 + q as i64) * fax;
                    k.iter()
                        .enumerate()
                        .map(|(j, &w)| {
                            let p = c + j as i64 - rad;
                            let at = self.boundary.map(p, n_ax).map(|g| (g - s0) as usize);
                            (at, w)
                        })
                        .collect()
                })
                .collect();
            let mut next = vec![0f64; outer * m * inner];
            let cur_ref = &cur;
            let job = |(slab, dst): (usize, &mut [f64])| {
                // one (outer, q) slab of `inner` values
                let (o, q) = (slab / m, slab % m);
                for &(at, w) in &taps[q] {
                    match at {
                        Some(a) => {
                            let src = &cur_ref[(o * len + a) * inner..(o * len + a + 1) * inner];
                            for (d, v) in dst.iter_mut().zip(src) {
                                *d += w * v;
                            }
                        }
                        None => {
                            for d in dst.iter_mut() {
                                *d += w * fill;
                            }
                        }
                    }
                }
            };
            if threads > 1 && outer * m > 1 {
                // slabs hold no buffer of their own: any free threads of the pool
                install(threads, || {
                    next.par_chunks_mut(inner).enumerate().for_each(job)
                })?;
            } else {
                next.chunks_mut(inner).enumerate().for_each(job);
            }
            cur = next;
            shape[ax] = m;
        }
        // write the region, rounded once
        let item = std::mem::size_of::<T>();
        let inner = r[n - 1] as usize;
        let rs: Vec<usize> = r.iter().map(|x| *x as usize).collect();
        for_rows(r, out, threads, |outer, dst| {
            let mut row = 0usize;
            for i in 0..n - 1 {
                row = row * rs[i] + outer[i] as usize;
            }
            for (j, el) in dst.chunks_exact_mut(item).enumerate() {
                let v = T::from_f64(cur[row * inner + j]);
                // SAFETY: `el` is one element of size_of::<T>() bytes
                unsafe { (el.as_mut_ptr() as *mut T).write_unaligned(v) }
            }
        })
    }
}

impl Source for Gaussian {
    fn reads_halo(&self) -> bool {
        true
    }
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    /// The source's storage grid in this level's elements: views above read in blocks
    /// of about one source chunk.
    fn grid(&self) -> Option<Vec<u64>> {
        scaled_grid(self.src.as_ref(), &self.factors)
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = region(&self.shape, start, stop)?;
        out.check(&r, self.dtype())?;
        // the box is held as f64, plus one filtered copy
        let bytes = |s: &[u64], r: &[u64]| {
            let (a, b) = self.input_box(s, r);
            (0..a.len()).map(|i| b[i] - a[i]).product::<u64>() * 16
        };
        let run = |s: &[u64], r: &[u64], o: &Out| match self.src.dtype() {
            DType::Int8 => self.part::<i8>(s, r, o, threads),
            DType::Int16 => self.part::<i16>(s, r, o, threads),
            DType::Int32 => self.part::<i32>(s, r, o, threads),
            DType::Int64 => self.part::<i64>(s, r, o, threads),
            DType::UInt8 => self.part::<u8>(s, r, o, threads),
            DType::UInt16 => self.part::<u16>(s, r, o, threads),
            DType::UInt32 => self.part::<u32>(s, r, o, threads),
            DType::UInt64 => self.part::<u64>(s, r, o, threads),
            DType::Float16 => self.part::<half::f16>(s, r, o, threads),
            DType::Float32 => self.part::<f32>(s, r, o, threads),
            DType::Float64 => self.part::<f64>(s, r, o, threads),
            other => Err(ReadError::new(format!(
                "Gaussian downsampling of {}",
                other.name()
            ))),
        };
        split(start, &r, out, self.max_read, &bytes, &run)
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        let bytes = |s: &[u64], r: &[u64]| {
            let (a, b) = self.input_box(s, r);
            (0..a.len()).map(|i| b[i] - a[i]).product::<u64>() * 16
        };
        let cost = |s: &[u64], r: &[u64]| self.part_cost(s, r, threads);
        split_cost(start, &r, self.max_read, &bytes, &cost)
    }
}

impl Gaussian {
    /// One piece, at its three fullest moments: the raw box while it is read; the raw
    /// box and its f64 copy; then, axis by axis, the f64 data and its filtered,
    /// decimated next version with that axis' kernel taps.
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let n = r.len();
        let (s, e) = self.input_box(start, r);
        let mut shape: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
        let elems: u64 = shape.iter().product();
        let raw = elems * self.src.dtype().size() as u64;
        let mut most = (raw + self.src.cost(&s, &e, threads)?).max(raw + elems * 8);
        for ax in 0..n {
            let cur: u64 = shape.iter().product::<u64>() * 8;
            shape[ax] = r[ax];
            let next: u64 = shape.iter().product::<u64>() * 8;
            // (Option<usize>, f64) per tap of every kept sample
            let taps = r[ax] * (self.kernels[ax].len() as u64 * 16 + 24);
            most = most.max(cur + next + taps);
        }
        Ok(most + SMALL)
    }
}

/// An element type the Gaussian reads and writes through f64.
trait GaussPx: Copy + Default + Send + Sync {
    fn val(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

macro_rules! gauss_int {
    ($($t:ty),*) => {$(
        impl GaussPx for $t {
            fn val(self) -> f64 {
                self as f64
            }
            fn from_f64(v: f64) -> Self {
                // half to even, as the mean; saturating
                v.round_ties_even() as $t
            }
        }
    )*};
}
gauss_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl GaussPx for f32 {
    fn val(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl GaussPx for f64 {
    fn val(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl GaussPx for half::f16 {
    fn val(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        half::f16::from_f64(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_even_division() {
        assert_eq!(div_half_even(3, 2), 2); // 1.5 -> 2
        assert_eq!(div_half_even(1, 2), 0); // 0.5 -> 0
        assert_eq!(div_half_even(5, 2), 2); // 2.5 -> 2
        assert_eq!(div_half_even(-1, 2), 0); // -0.5 -> 0
        assert_eq!(div_half_even(-5, 2), -2); // -2.5 -> -2
        assert_eq!(div_half_even(-3, 2), -2); // -1.5 -> -2
        assert_eq!(div_half_even(10, 3), 3);
        assert_eq!(div_half_even(5, 3), 2);
        for sum in -40i64..40 {
            for count in 1u64..9 {
                assert_eq!(
                    div_half_even_64(sum, count) as i128,
                    div_half_even(sum as i128, count)
                );
            }
        }
    }

    #[test]
    fn boundary_modes_as_scipy() {
        // n = 4: indices -3 .. 6
        let got = |b: Boundary| {
            (-3..7)
                .map(|p| b.map(p, 4).unwrap_or(-1))
                .collect::<Vec<_>>()
        };
        assert_eq!(got(Boundary::Reflect), vec![2, 1, 0, 0, 1, 2, 3, 3, 2, 1]);
        assert_eq!(got(Boundary::Mirror), vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
        assert_eq!(got(Boundary::Nearest), vec![0, 0, 0, 0, 1, 2, 3, 3, 3, 3]);
        assert_eq!(
            got(Boundary::Constant(0.0)),
            vec![-1, -1, -1, 0, 1, 2, 3, -1, -1, -1]
        );
    }
}
