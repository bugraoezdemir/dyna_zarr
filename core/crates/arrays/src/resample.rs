//! Resampling under an affine transformation (Workplan WP15, task DR4): a lazy view
//! whose element at output index `o` is the source's value at `M o` (`M` maps output
//! indices to input indices, as scipy.ndimage.affine_transform's matrix does).
//!
//! Index space as NGFF 0.6 and DR2: integer indices are pixel centres. Asked for a
//! region, the view reads exactly the input box that region needs
//! ([`dyna_zarr_transforms::input_region`]), splitting a request whose box would be too
//! large into halves first (a rotation widens the box), then computes every output row
//! in parallel straight into the destination.
//!
//! Edges, as scipy's modes of the same names:
//! - `Constant` (scipy's default): a point outside the pixel centres `0 .. n - 1` on any
//!   axis is `fill`; inside, only real pixels are interpolated.
//! - `GridConstant`: the image extends half a pixel beyond its centres; beyond that it
//!   is `fill`, and linear interpolation blends towards `fill` there.
//!
//! Nearest copies whole elements (every type); linear works in f64 and writes integers
//! rounded half away from zero, as scipy does (bool and complex: nearest only). Points
//! within 1e-9 of an integer count as that integer ([`snap`]), the rule the input
//! region uses too, so a read never leaves its box.

use rayon::prelude::*;

use dyna_zarr_transforms::{input_region, snap, Interpolation, Matrix};

use crate::dtype::DType;
use crate::out::Out;
use crate::pool::install;
use crate::source::{bytes_of, c_strides, read_vec, region, Source, SourceRef, SMALL};
use crate::ReadError;

/// How points near and beyond the image's edge are treated (scipy's modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Outside the pixel centres: fill (scipy "constant", its default).
    Constant,
    /// Outside the pixels' extent (half a pixel beyond the centres): fill; linear
    /// blends towards fill beyond the centres (scipy "grid-constant").
    GridConstant,
}

impl EdgeMode {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "constant" => Some(Self::Constant),
            "grid-constant" => Some(Self::GridConstant),
            _ => None,
        }
    }
}

/// The most axes a resampled array may have (fixed-size per-pixel state).
const MAX_AXES: usize = 8;
/// Input bytes one read may need before the request is split (a rotation widens the box).
pub const DEFAULT_MAX_READ: u64 = 256 << 20;

/// `source` resampled onto a grid of `shape`: output index `o` takes the source's value
/// at `out_to_in · o`.
pub struct Resample {
    src: SourceRef,
    shape: Vec<u64>,
    m: Matrix,
    interp: Interpolation,
    mode: EdgeMode,
    fill: f64,
    fill_bytes: Vec<u8>,
    max_read: u64,
}

impl Resample {
    pub fn new(
        src: SourceRef,
        shape: &[u64],
        out_to_in: Matrix,
        interp: Interpolation,
        mode: EdgeMode,
        fill: f64,
    ) -> Result<Self, ReadError> {
        let n = src.shape().len();
        if out_to_in.ndim() != n || shape.len() != n {
            return Err(ReadError::new(format!(
                "a matrix over {} axes and an output shape of {} for a source of {n} axes",
                out_to_in.ndim(),
                shape.len()
            )));
        }
        if n == 0 || n > MAX_AXES {
            return Err(ReadError::new(format!(
                "resampling takes 1 to {MAX_AXES} axes, not {n}"
            )));
        }
        let dtype = src.dtype();
        if interp == Interpolation::Linear && (dtype == DType::Bool || dtype.is_complex()) {
            return Err(ReadError::new(format!(
                "linear interpolation of {} is not supported: use nearest",
                dtype.name()
            )));
        }
        let fill_bytes = element(dtype, fill);
        Ok(Self {
            src,
            shape: shape.to_vec(),
            m: out_to_in,
            interp,
            mode,
            fill,
            fill_bytes,
            max_read: DEFAULT_MAX_READ,
        })
    }

    /// Split requests whose input box would exceed `bytes`.
    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    fn read_part(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        if r.contains(&0) {
            return Ok(());
        }
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let src_shape = self.src.shape();
        let need = input_region(&self.m, start, &stop, self.interp, src_shape)
            .map_err(|e| ReadError::new(e.message))?;
        let Some((s, e)) = need else {
            out.fill(r, &self.fill_bytes);
            return Ok(());
        };
        let item = self.src.dtype().size() as u64;
        let bytes: u64 = (0..n).map(|i| e[i] - s[i]).product::<u64>() * item;
        if bytes > self.max_read && r.iter().any(|x| *x > 1) {
            // halve the longest axis of the output region; each half reads its own box
            let ax = (0..n).max_by_key(|&i| r[i]).unwrap_or(0);
            let half = r[ax] / 2;
            let mut r1 = r.to_vec();
            r1[ax] = half;
            self.read_part(start, &r1, out, threads)?;
            let mut s2 = start.to_vec();
            s2[ax] += half;
            let mut r2 = r.to_vec();
            r2[ax] -= half;
            let mut at = out.at.clone();
            at[ax] += half;
            return self.read_part(&s2, &r2, &out.moved(at), threads);
        }
        let box_shape: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
        let meta = BoxMeta {
            strides: c_strides(&box_shape, 1),
            start: s.clone(),
            shape: box_shape,
        };
        let job = Job {
            start,
            r,
            out,
            meta: &meta,
            src_shape,
        };
        match self.interp {
            // nearest only moves bits: elements as unsigned integers of their size
            Interpolation::Nearest => match item {
                1 => self.nearest_part::<u8>(&job, &s, &e, threads),
                2 => self.nearest_part::<u16>(&job, &s, &e, threads),
                4 => self.nearest_part::<u32>(&job, &s, &e, threads),
                8 => self.nearest_part::<u64>(&job, &s, &e, threads),
                16 => self.nearest_part::<u128>(&job, &s, &e, threads),
                other => Err(ReadError::new(format!("elements of {other} bytes"))),
            },
            Interpolation::Linear => match self.src.dtype() {
                DType::Int8 => self.linear_part::<i8>(&job, &s, &e, threads),
                DType::Int16 => self.linear_part::<i16>(&job, &s, &e, threads),
                DType::Int32 => self.linear_part::<i32>(&job, &s, &e, threads),
                DType::Int64 => self.linear_part::<i64>(&job, &s, &e, threads),
                DType::UInt8 => self.linear_part::<u8>(&job, &s, &e, threads),
                DType::UInt16 => self.linear_part::<u16>(&job, &s, &e, threads),
                DType::UInt32 => self.linear_part::<u32>(&job, &s, &e, threads),
                DType::UInt64 => self.linear_part::<u64>(&job, &s, &e, threads),
                DType::Float16 => self.linear_part::<half::f16>(&job, &s, &e, threads),
                DType::Float32 => self.linear_part::<f32>(&job, &s, &e, threads),
                DType::Float64 => self.linear_part::<f64>(&job, &s, &e, threads),
                other => Err(ReadError::new(format!(
                    "linear interpolation of {}",
                    other.name()
                ))),
            },
        }
    }

    /// The cost of `read_part`: the same halving, then each piece's typed input box and
    /// what reading it costs (pieces one after another).
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let need = input_region(&self.m, start, &stop, self.interp, self.src.shape())
            .map_err(|e| ReadError::new(e.message))?;
        let Some((s, e)) = need else {
            // all fill: only the bookkeeping
            return Ok(SMALL);
        };
        let box_shape: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
        let bytes = bytes_of(&box_shape, self.src.dtype().size());
        if bytes > self.max_read && r.iter().any(|x| *x > 1) {
            let ax = (0..n).max_by_key(|&i| r[i]).unwrap_or(0);
            let half = r[ax] / 2;
            let mut r1 = r.to_vec();
            r1[ax] = half;
            let mut s2 = start.to_vec();
            s2[ax] += half;
            let mut r2 = r.to_vec();
            r2[ax] -= half;
            return Ok(self
                .part_cost(start, &r1, threads)?
                .max(self.part_cost(&s2, &r2, threads)?));
        }
        Ok(bytes + SMALL + self.src.cost(&s, &e, threads)?)
    }

    /// The input box `s .. e` read straight into a vector of `T` (the element type, or
    /// an unsigned integer of its size).
    fn read_typed<T: Copy + Default>(
        &self,
        s: &[u64],
        e: &[u64],
        threads: usize,
    ) -> Result<Vec<T>, ReadError> {
        let v = read_vec::<T>(self.src.as_ref(), s, e, threads)?;
        Ok(v)
    }

    fn nearest_part<T: Copy + Default + Send + Sync>(
        &self,
        job: &Job,
        s: &[u64],
        e: &[u64],
        threads: usize,
    ) -> Result<(), ReadError> {
        let data = self.read_typed::<T>(s, e, threads)?;
        // SAFETY (both writes): `dst` is exactly one element of size_of::<T>() bytes
        let fill: T = unsafe { (self.fill_bytes.as_ptr() as *const T).read_unaligned() };
        self.rows(job, threads, |p, dst| {
            let v = match self.nearest(job, p) {
                Some(off) => data[off],
                None => fill,
            };
            unsafe { (dst.as_mut_ptr() as *mut T).write_unaligned(v) }
        })
    }

    fn linear_part<T: Px>(
        &self,
        job: &Job,
        s: &[u64],
        e: &[u64],
        threads: usize,
    ) -> Result<(), ReadError> {
        let data = self.read_typed::<T>(s, e, threads)?;
        self.rows(job, threads, |p, dst| {
            T::put(self.linear::<T>(job, &data, p), dst)
        })
    }

    /// Every output row of the job, in parallel: `pixel(p, dst)` writes the element at
    /// input point `p` into `dst`.
    fn rows<F>(&self, job: &Job, threads: usize, pixel: F) -> Result<(), ReadError>
    where
        F: Fn(&[f64], &mut [u8]) + Sync,
    {
        let n = job.r.len();
        let inner = job.r[n - 1];
        let rows: u64 = job.r[..n - 1].iter().product();
        let item = self.src.dtype().size();
        let step: Vec<f64> = (0..n).map(|i| self.m.get(i, n - 1)).collect();
        let one_row = |row: usize| {
            let mut outer = [0u64; MAX_AXES];
            let mut rest = row as u64;
            for i in (0..n - 1).rev() {
                outer[i] = rest % job.r[i];
                rest /= job.r[i];
            }
            let mut idx = [0f64; MAX_AXES];
            for i in 0..n - 1 {
                idx[i] = (job.start[i] + outer[i]) as f64;
            }
            idx[n - 1] = job.start[n - 1] as f64;
            let p0 = self.m.apply(&idx[..n]);
            // SAFETY: each row is written by one task only, inside the checked region
            let dst = unsafe { job.out.row_mut(&outer[..n - 1], inner) };
            let mut p = [0f64; MAX_AXES];
            for (k, el) in dst.chunks_exact_mut(item).enumerate() {
                for i in 0..n {
                    p[i] = p0[i] + k as f64 * step[i];
                }
                pixel(&p[..n], el);
            }
        };
        let min_rows = (4096 / inner.max(1)).max(1) as usize;
        let rows = rows as usize;
        if threads > 1 && rows > 1 {
            // rows hold no buffer: they use whatever threads of the pool are free
            install(threads, || {
                (0..rows)
                    .into_par_iter()
                    .with_min_len(min_rows)
                    .for_each(one_row)
            })?;
        } else {
            (0..rows).for_each(one_row);
        }
        Ok(())
    }

    /// The offset in the input box of the pixel nearest to `p`, or None for fill.
    fn nearest(&self, job: &Job, p: &[f64]) -> Option<usize> {
        let b = job.meta;
        let mut off = 0usize;
        for (i, &x) in p.iter().enumerate() {
            let size = job.src_shape[i] as f64;
            if self.mode == EdgeMode::Constant {
                let s = snap(x);
                if s < 0.0 || s > size - 1.0 {
                    return None;
                }
            }
            let j = snap(x + 0.5).floor();
            let rel = j - b.start[i] as f64;
            if j < 0.0 || j > size - 1.0 || rel < 0.0 || rel >= b.shape[i] as f64 {
                return None;
            }
            off += rel as usize * b.strides[i];
        }
        Some(off)
    }

    /// The linear interpolation at `p` of the typed input box `data`.
    fn linear<T: Px>(&self, job: &Job, data: &[T], p: &[f64]) -> f64 {
        let b = job.meta;
        let n = p.len();
        let mut lo = [0i64; MAX_AXES];
        let mut w = [0f64; MAX_AXES];
        let mut moving = [0usize; MAX_AXES]; // axes with a fractional part
        let mut k = 0;
        let mut base = 0i64; // the lower corner's offset in the box
        for i in 0..n {
            let size = job.src_shape[i] as f64;
            let s = snap(p[i]);
            let outside = match self.mode {
                EdgeMode::Constant => s < 0.0 || s > size - 1.0,
                EdgeMode::GridConstant => s <= -1.0 || s >= size,
            };
            if outside {
                return self.fill;
            }
            let f = s.floor();
            lo[i] = f as i64;
            w[i] = s - f;
            base += (lo[i] - b.start[i] as i64) * b.strides[i] as i64;
            if w[i] > 0.0 {
                moving[k] = i;
                k += 1;
            }
        }
        if self.mode == EdgeMode::Constant {
            // inside the pixel centres, every corner with a weight is in the image and in
            // the box read (it holds floor .. ceil of every point): no checks
            let base = base as usize;
            let v = |off: usize| data[off].val();
            // unrolled for the usual cases: 1, 2 or 3 axes with a fractional position
            match k {
                0 => return v(base),
                1 => {
                    let (a, s) = (w[moving[0]], b.strides[moving[0]]);
                    return v(base) * (1.0 - a) + v(base + s) * a;
                }
                2 => {
                    let (a, sa) = (w[moving[0]], b.strides[moving[0]]);
                    let (c, sc) = (w[moving[1]], b.strides[moving[1]]);
                    let lo = v(base) * (1.0 - c) + v(base + sc) * c;
                    let hi = v(base + sa) * (1.0 - c) + v(base + sa + sc) * c;
                    return lo * (1.0 - a) + hi * a;
                }
                3 => {
                    let (a, sa) = (w[moving[0]], b.strides[moving[0]]);
                    let (c, sc) = (w[moving[1]], b.strides[moving[1]]);
                    let (d, sd) = (w[moving[2]], b.strides[moving[2]]);
                    let line = |o: usize| v(o) * (1.0 - d) + v(o + sd) * d;
                    let plane = |o: usize| line(o) * (1.0 - c) + line(o + sc) * c;
                    return plane(base) * (1.0 - a) + plane(base + sa) * a;
                }
                _ => {}
            }
            let mut sum = 0.0;
            for corner in 0..(1usize << k) {
                let mut weight = 1.0;
                let mut off = base;
                for (m, &i) in moving[..k].iter().enumerate() {
                    if corner >> m & 1 == 1 {
                        weight *= w[i];
                        off += b.strides[i];
                    } else {
                        weight *= 1.0 - w[i];
                    }
                }
                sum += weight * data[off].val();
            }
            return sum;
        }
        // grid-constant: corners beyond the image are fill
        let mut sum = 0.0;
        for corner in 0..(1usize << k) {
            let mut weight = 1.0;
            let mut off = 0usize;
            let mut inside = true;
            let mut m = 0;
            for i in 0..n {
                let mut j = lo[i];
                if m < k && moving[m] == i {
                    if corner >> m & 1 == 1 {
                        j += 1;
                        weight *= w[i];
                    } else {
                        weight *= 1.0 - w[i];
                    }
                    m += 1;
                }
                let rel = j - b.start[i] as i64;
                if j < 0 || j >= job.src_shape[i] as i64 || rel < 0 || rel >= b.shape[i] as i64 {
                    inside = false;
                } else {
                    off += rel as usize * b.strides[i];
                }
            }
            sum += weight * if inside { data[off].val() } else { self.fill };
        }
        sum
    }
}

/// Where the input box read for one request lies, and its C strides in elements.
struct BoxMeta {
    start: Vec<u64>,
    shape: Vec<u64>,
    strides: Vec<usize>,
}

/// One request: its output region and destination, and the input box it reads.
struct Job<'j, 'o> {
    start: &'j [u64],
    r: &'j [u64],
    out: &'j Out<'o>,
    meta: &'j BoxMeta,
    src_shape: &'j [u64],
}

impl Source for Resample {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
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
        self.read_part(start, &r, out, threads)
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        self.part_cost(start, &r, threads)
    }
}

/// An element type linear interpolation reads and writes through f64.
trait Px: Copy + Default + Send + Sync + 'static {
    /// The element's value.
    fn val(self) -> f64;
    /// `v` written as one element.
    fn put(v: f64, b: &mut [u8]);
}

macro_rules! px_int {
    ($($t:ty),*) => {$(
        impl Px for $t {
            fn val(self) -> f64 {
                self as f64
            }
            fn put(v: f64, b: &mut [u8]) {
                // half away from zero, as scipy writes integer outputs; saturating
                b.copy_from_slice(&(v.round() as $t).to_ne_bytes());
            }
        }
    )*};
}
px_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Px for f32 {
    fn val(self) -> f64 {
        self as f64
    }
    fn put(v: f64, b: &mut [u8]) {
        b.copy_from_slice(&(v as f32).to_ne_bytes());
    }
}

impl Px for f64 {
    fn val(self) -> f64 {
        self
    }
    fn put(v: f64, b: &mut [u8]) {
        b.copy_from_slice(&v.to_ne_bytes());
    }
}

impl Px for half::f16 {
    fn val(self) -> f64 {
        self.to_f64()
    }
    fn put(v: f64, b: &mut [u8]) {
        b.copy_from_slice(&half::f16::from_f64(v).to_ne_bytes());
    }
}

/// One element of `dtype` with value `v` (bool: nonzero; integers rounded and
/// saturated; complex: real part `v`).
fn element(dtype: DType, v: f64) -> Vec<u8> {
    let mut b = vec![0u8; dtype.size()];
    match dtype {
        DType::Bool => b[0] = u8::from(v != 0.0),
        DType::Int8 => i8::put(v, &mut b),
        DType::Int16 => i16::put(v, &mut b),
        DType::Int32 => i32::put(v, &mut b),
        DType::Int64 => i64::put(v, &mut b),
        DType::UInt8 => u8::put(v, &mut b),
        DType::UInt16 => u16::put(v, &mut b),
        DType::UInt32 => u32::put(v, &mut b),
        DType::UInt64 => u64::put(v, &mut b),
        DType::Float16 => half::f16::put(v, &mut b),
        DType::Float32 => f32::put(v, &mut b),
        DType::Float64 => f64::put(v, &mut b),
        DType::Complex64 => f32::put(v, &mut b[..4]),
        DType::Complex128 => f64::put(v, &mut b[..8]),
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::read;
    use std::sync::Arc;

    /// An in-memory C-ordered array.
    struct Mem {
        shape: Vec<u64>,
        dtype: DType,
        data: Vec<u8>,
    }

    impl Source for Mem {
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            self.dtype
        }
        fn read_into(
            &self,
            start: &[u64],
            stop: &[u64],
            out: &Out,
            _threads: usize,
        ) -> Result<(), ReadError> {
            let r = region(&self.shape, start, stop)?;
            out.check(&r, self.dtype)?;
            let item = self.dtype.size();
            let strides = c_strides(&self.shape, item);
            let off: usize = (0..r.len()).map(|i| start[i] as usize * strides[i]).sum();
            out.copy_from(&r, &self.data[off..], &strides, item, None);
            Ok(())
        }

        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(0)
        }
    }

    fn mem_f32(shape: &[u64], values: Vec<f32>) -> SourceRef {
        Arc::new(Mem {
            shape: shape.to_vec(),
            dtype: DType::Float32,
            data: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        })
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn rows(r: &[&[f64]]) -> Matrix {
        Matrix::from_rows(&r.iter().map(|x| x.to_vec()).collect::<Vec<_>>()).unwrap()
    }

    fn all(src: &Resample, threads: usize) -> Vec<u8> {
        let shape = src.shape().to_vec();
        read(src, &vec![0; shape.len()], &shape, threads).unwrap()
    }

    #[test]
    fn identity_and_shift() {
        let src = mem_f32(&[2, 5], (0..10).map(|v| v as f32).collect());
        for interp in [Interpolation::Nearest, Interpolation::Linear] {
            let id = Resample::new(
                src.clone(),
                &[2, 5],
                Matrix::identity(2),
                interp,
                EdgeMode::Constant,
                -1.0,
            )
            .unwrap();
            assert_eq!(
                f32s(&all(&id, 1)),
                (0..10).map(|v| v as f32).collect::<Vec<_>>()
            );
            // out[y, x] = src[y, x + 2]: the last two columns fall outside -> fill
            let shift = rows(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 2.0]]);
            let s = Resample::new(
                src.clone(),
                &[2, 5],
                shift,
                interp,
                EdgeMode::Constant,
                -1.0,
            )
            .unwrap();
            assert_eq!(
                f32s(&all(&s, 2)),
                vec![2.0, 3.0, 4.0, -1.0, -1.0, 7.0, 8.0, 9.0, -1.0, -1.0]
            );
        }
    }

    #[test]
    fn a_quarter_turn_is_exact() {
        // out[y, x] = src[x, 3 - y]: rotated by 90 degrees
        let src = mem_f32(&[4, 4], (0..16).map(|v| v as f32).collect());
        let turn = rows(&[&[0.0, 1.0, 0.0], &[-1.0, 0.0, 3.0]]);
        for interp in [Interpolation::Nearest, Interpolation::Linear] {
            let r = Resample::new(
                src.clone(),
                &[4, 4],
                turn.clone(),
                interp,
                EdgeMode::Constant,
                0.0,
            )
            .unwrap();
            let got = f32s(&all(&r, 3));
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(got[y * 4 + x], (x * 4 + (3 - y)) as f32);
                }
            }
        }
    }

    #[test]
    fn linear_midpoints_and_integer_rounding_as_scipy() {
        let src = mem_f32(&[4], vec![10.0, 20.0, 30.0, 40.0]);
        let half = rows(&[&[0.5, 0.0]]);
        let r = Resample::new(
            src,
            &[7],
            half.clone(),
            Interpolation::Linear,
            EdgeMode::Constant,
            -1.0,
        )
        .unwrap();
        assert_eq!(
            f32s(&all(&r, 1)),
            vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        );
        // uint8 [0, 1] at 0, 0.25, 0.5, 0.75: 0, 0, 1, 1 (half away from zero)
        let u: SourceRef = Arc::new(Mem {
            shape: vec![2],
            dtype: DType::UInt8,
            data: vec![0, 1],
        });
        let quarter = rows(&[&[0.25, 0.0]]);
        let r = Resample::new(
            u,
            &[4],
            quarter,
            Interpolation::Linear,
            EdgeMode::Constant,
            0.0,
        )
        .unwrap();
        assert_eq!(all(&r, 1), vec![0, 0, 1, 1]);
    }

    #[test]
    fn edge_modes_as_scipy() {
        // sample [10, 20, 30, 40] at -0.4, 1.5, 3.4 (out index i -> -0.4 + 1.9 i)
        let src = mem_f32(&[4], vec![10.0, 20.0, 30.0, 40.0]);
        let m = rows(&[&[1.9, -0.4]]);
        let get = |interp, mode| {
            let r = Resample::new(src.clone(), &[3], m.clone(), interp, mode, -1.0).unwrap();
            f32s(&all(&r, 1))
        };
        // scipy.ndimage.map_coordinates with cval=-1 gives these: 1.5 rounds up to index
        // 2 (30); grid-constant keeps 3.4 inside the last pixel and blends -0.4 with fill
        assert_eq!(
            get(Interpolation::Nearest, EdgeMode::Constant),
            vec![-1.0, 30.0, -1.0]
        );
        assert_eq!(
            get(Interpolation::Nearest, EdgeMode::GridConstant),
            vec![10.0, 30.0, 40.0]
        );
        assert_eq!(
            get(Interpolation::Linear, EdgeMode::Constant),
            vec![-1.0, 25.0, -1.0]
        );
        let g = get(Interpolation::Linear, EdgeMode::GridConstant);
        assert!(
            (g[0] - 5.6).abs() < 1e-5 && g[1] == 25.0 && (g[2] - 23.6).abs() < 1e-5,
            "{g:?}"
        );
    }

    #[test]
    fn splitting_large_requests_changes_nothing() {
        let src = mem_f32(&[30, 40], (0..1200).map(|v| (v as f32).sin()).collect());
        let (c, s) = (0.3f64.cos(), 0.3f64.sin());
        let m = rows(&[&[c, -s, 8.0], &[s, c, -5.0]]);
        let whole = Resample::new(
            src.clone(),
            &[30, 40],
            m.clone(),
            Interpolation::Linear,
            EdgeMode::Constant,
            0.0,
        )
        .unwrap();
        let split = Resample::new(
            src,
            &[30, 40],
            m,
            Interpolation::Linear,
            EdgeMode::Constant,
            0.0,
        )
        .unwrap()
        .with_max_read(64);
        assert_eq!(all(&whole, 1), all(&split, 4));
    }
}
