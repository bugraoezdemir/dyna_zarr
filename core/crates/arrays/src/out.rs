//! Where a read goes: a region of a C-ordered destination buffer that the caller owns
//! (a Rust buffer, a NumPy array's memory, a GPU staging buffer).
//!
//! An [`Out`] is the whole destination (its bytes and shape, C order) and the corner
//! `at` where the region read lands. The region's shape is given by the read. Views
//! derive new `Out`s over the same bytes (another corner, an axis merged or inserted),
//! so a concatenation's parts land in their places without a copy. Different reads into
//! one buffer must target disjoint regions; the views here only ever do.

use unsafe_cell_slice::UnsafeCellSlice;

use crate::dtype::DType;
use crate::ReadError;

/// Converts one source element into one destination element.
pub(crate) type ConvertRef<'f> = &'f dyn Fn(&[u8], &mut [u8]);

/// A destination region: `shape` and `at` have one value per axis of the read.
#[derive(Clone)]
pub struct Out<'a> {
    pub(crate) buf: UnsafeCellSlice<'a, u8>,
    pub(crate) shape: Vec<u64>,
    pub(crate) at: Vec<u64>,
    pub(crate) dtype: DType,
}

impl<'a> Out<'a> {
    /// The whole C-ordered buffer `bytes` of `shape` elements of `dtype`; the region read
    /// lands at `at`.
    pub fn new(
        bytes: &'a mut [u8],
        shape: &[u64],
        at: &[u64],
        dtype: DType,
    ) -> Result<Self, ReadError> {
        if shape.len() != at.len() {
            return Err(ReadError::new(format!(
                "destination of {} axes with a corner of {}",
                shape.len(),
                at.len()
            )));
        }
        let want = shape.iter().product::<u64>() as usize * dtype.size();
        if bytes.len() != want {
            return Err(ReadError::new(format!(
                "destination of {} bytes for {shape:?} {} ({want} bytes)",
                bytes.len(),
                dtype.name()
            )));
        }
        Ok(Self {
            buf: UnsafeCellSlice::new(bytes),
            shape: shape.to_vec(),
            at: at.to_vec(),
            dtype,
        })
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The same buffer, the region at `at`.
    pub fn moved(&self, at: Vec<u64>) -> Self {
        Self { at, ..self.clone() }
    }

    /// Check that a region of `region` elements per axis fits at `at`.
    pub(crate) fn check(&self, region: &[u64], dtype: DType) -> Result<(), ReadError> {
        if dtype != self.dtype {
            return Err(ReadError::new(format!(
                "reading {} into a {} destination",
                dtype.name(),
                self.dtype.name()
            )));
        }
        if region.len() != self.ndim() {
            return Err(ReadError::new(format!(
                "a region of {} axes into a destination of {}",
                region.len(),
                self.ndim()
            )));
        }
        for i in 0..region.len() {
            if self.at[i] + region[i] > self.shape[i] {
                return Err(ReadError::new(format!(
                    "a region of {region:?} at {:?} does not fit a destination of {:?}",
                    self.at, self.shape
                )));
            }
        }
        Ok(())
    }

    /// The destination for a read with an extra axis of extent 1 at `axis` (a squeezed
    /// view: the source has the axis, this destination does not): the axis is inserted
    /// with size 1, which changes no byte offset.
    pub(crate) fn with_axis(&self, axis: usize) -> Self {
        let mut shape = self.shape.clone();
        let mut at = self.at.clone();
        shape.insert(axis, 1);
        at.insert(axis, 0);
        Self {
            shape,
            at,
            ..self.clone()
        }
    }

    /// The destination without `axis`, along which the region has extent 1 (an inserted
    /// axis: the source lacks it). Merged into the next axis (same bytes, C order); a
    /// last axis of size 1 is dropped. None when the last axis has size > 1: the region
    /// is then not one C-ordered block and the caller copies through a scratch buffer.
    pub(crate) fn without_axis(&self, axis: usize) -> Option<Self> {
        let n = self.ndim();
        let mut shape = self.shape.clone();
        let mut at = self.at.clone();
        if axis + 1 < n {
            shape[axis + 1] *= shape[axis];
            at[axis + 1] += at[axis] * self.shape[axis + 1];
        } else if self.shape[axis] != 1 {
            return None;
        }
        shape.remove(axis);
        at.remove(axis);
        Some(Self {
            shape,
            at,
            ..self.clone()
        })
    }

    /// The bytes of one row of the region: the elements `0 .. len` along the last axis at
    /// the region's index `outer` on the other axes.
    ///
    /// # Safety
    /// Callers writing rows from several threads must write different rows; the row
    /// lies inside the region this Out was given (`check`).
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn row_mut(&self, outer: &[u64], len: u64) -> &mut [u8] {
        let n = self.ndim();
        let strides = self.strides();
        let mut e = self.at[n - 1] as usize;
        for i in 0..n - 1 {
            e += (self.at[i] + outer[i]) as usize * strides[i];
        }
        let item = self.dtype.size();
        unsafe { self.buf.index_mut(e * item..(e + len as usize) * item) }
    }

    /// Every element of a region of `region` elements set to `value` (one element's
    /// bytes).
    pub(crate) fn fill(&self, region: &[u64], value: &[u8]) {
        let n = region.len();
        if n == 0 || region.contains(&0) {
            return;
        }
        let rows: u64 = region[..n - 1].iter().product();
        let mut outer = vec![0u64; n - 1];
        for r in 0..rows {
            let mut rest = r;
            for i in (0..n - 1).rev() {
                outer[i] = rest % region[i];
                rest /= region[i];
            }
            // SAFETY: one thread, each row once, inside the region
            let row = unsafe { self.row_mut(&outer, region[n - 1]) };
            for el in row.chunks_mut(value.len()) {
                el.copy_from_slice(value);
            }
        }
    }

    /// C strides of the destination, in elements.
    fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.ndim()];
        for i in (0..self.ndim().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1] as usize;
        }
        s
    }

    /// Copy a region of `region` elements from `src`, whose element (i, j, ...) is the
    /// `src_item` bytes at `sum(index * src_strides)` (byte strides, any order), each
    /// converted by `convert` (source element -> destination element) or copied as is
    /// (then `src_item` must be the destination's item size).
    pub(crate) fn copy_from(
        &self,
        region: &[u64],
        src: &[u8],
        src_strides: &[usize],
        src_item: usize,
        convert: Option<ConvertRef>,
    ) {
        let n = region.len();
        if n == 0 || region.contains(&0) {
            return;
        }
        let item = self.dtype.size();
        let dst_strides = self.strides();
        let base: usize = (0..n).map(|i| self.at[i] as usize * dst_strides[i]).sum();
        let inner = region[n - 1] as usize;
        let contiguous = convert.is_none() && src_strides[n - 1] == item;
        let mut idx = vec![0u64; n];
        loop {
            let d0 = (base
                + (0..n - 1)
                    .map(|i| idx[i] as usize * dst_strides[i])
                    .sum::<usize>())
                * item;
            let s0: usize = (0..n - 1).map(|i| idx[i] as usize * src_strides[i]).sum();
            // SAFETY: the destination range lies inside the region this Out was given
            // (checked by `check`), which no other concurrent read targets.
            let dst = unsafe { self.buf.index_mut(d0..d0 + inner * item) };
            if contiguous {
                dst.copy_from_slice(&src[s0..s0 + inner * item]);
            } else if let Some(f) = convert {
                let step = src_strides[n - 1];
                for k in 0..inner {
                    let s = &src[s0 + k * step..s0 + k * step + src_item];
                    f(s, &mut dst[k * item..(k + 1) * item]);
                }
            } else {
                // a strided copy of whole elements: moved as integers of their size
                // (a byte copy per element costs a call each)
                let step = src_strides[n - 1];
                let need = s0 + (inner - 1) * step + item;
                assert!(need <= src.len(), "strided copy beyond its source");
                match item {
                    1 => strided::<u8>(src, s0, step, dst, inner),
                    2 => strided::<u16>(src, s0, step, dst, inner),
                    4 => strided::<u32>(src, s0, step, dst, inner),
                    8 => strided::<u64>(src, s0, step, dst, inner),
                    16 => strided::<u128>(src, s0, step, dst, inner),
                    _ => {
                        for k in 0..inner {
                            dst[k * item..(k + 1) * item]
                                .copy_from_slice(&src[s0 + k * step..s0 + k * step + item]);
                        }
                    }
                }
            }
            let mut ax = n - 1;
            loop {
                if ax == 0 {
                    return;
                }
                ax -= 1;
                idx[ax] += 1;
                if idx[ax] < region[ax] {
                    break;
                }
                idx[ax] = 0;
            }
        }
    }
}

/// `n` elements of type `T` from `src` (bytes, the first at byte `s0`, then every `step`
/// bytes) into `dst` (contiguous). The caller checked that every element lies in `src`.
fn strided<T: Copy>(src: &[u8], s0: usize, step: usize, dst: &mut [u8], n: usize) {
    let size = std::mem::size_of::<T>();
    debug_assert!(dst.len() >= n * size && s0 + (n - 1) * step + size <= src.len());
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    for k in 0..n {
        // SAFETY: both ranges were checked above; unaligned reads and writes
        unsafe {
            let v = (sp.add(s0 + k * step) as *const T).read_unaligned();
            (dp.add(k * size) as *mut T).write_unaligned(v);
        }
    }
}
