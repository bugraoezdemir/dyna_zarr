//! What scipy's n-D filters share (`NI_InitFilterOffsets` and its iterators): a
//! footprint around each output, positions past the array's edges mapped back by the
//! mode, and the block extended to the box the outputs read. The rank filter and the
//! n-D correlation build on it.

use rayon::prelude::*;

use crate::{FilterError, Mode, Span};

/// Which positions around an output a filter reads: a box of `shape` (C order), all of
/// it or the elements `mask` marks, shifted by `origin` per axis as scipy shifts it.
#[derive(Debug, Clone, PartialEq)]
pub struct Footprint {
    pub shape: Vec<usize>,
    pub mask: Option<Vec<bool>>,
    pub origin: Vec<isize>,
}

impl Footprint {
    /// A whole box of `shape`, centred.
    pub fn sized(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            mask: None,
            origin: vec![0; shape.len()],
        }
    }

    /// The number of positions read (scipy's `filter_size`).
    pub fn size(&self) -> usize {
        match &self.mask {
            Some(m) => m.iter().filter(|&&b| b).count(),
            None => self.shape.iter().product(),
        }
    }

    /// How far the footprint reaches before and after an output along `axis`.
    pub fn reach(&self, axis: usize) -> (usize, usize) {
        // within 0 .. shape (checked when the filter is made)
        let before = ((self.shape[axis] / 2) as isize + self.origin[axis]) as usize;
        (before, self.shape[axis] - 1 - before)
    }

    /// Positions read, in the footprint's C order (scipy's buffer order), as offsets in
    /// the box.
    pub(crate) fn positions(&self) -> Vec<Vec<usize>> {
        let n = self.shape.len();
        let total: usize = self.shape.iter().product();
        let mut out = Vec::new();
        let mut idx = vec![0usize; n];
        for f in 0..total {
            if self.mask.as_ref().is_none_or(|m| m[f]) {
                out.push(idx.clone());
            }
            for k in (0..n).rev() {
                idx[k] += 1;
                if idx[k] < self.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        out
    }

    /// The shape, mask and origins fit together (scipy's checks); the footprint may
    /// read nothing.
    pub(crate) fn check_shape(&self) -> Result<(), FilterError> {
        let n = self.shape.len();
        if self.origin.len() != n {
            return Err(FilterError::new(format!(
                "{} origins for a {n}-d footprint",
                self.origin.len()
            )));
        }
        if self.shape.contains(&0) {
            return Err(FilterError::new(format!(
                "a footprint of shape {:?}",
                self.shape
            )));
        }
        if let Some(m) = &self.mask {
            if m.len() != self.shape.iter().product::<usize>() {
                return Err(FilterError::new(format!(
                    "a mask of {} for a footprint of shape {:?}",
                    m.len(),
                    self.shape
                )));
            }
        }
        for (a, (&s, &o)) in self.shape.iter().zip(&self.origin).enumerate() {
            let c = (s / 2) as isize + o;
            if c < 0 || c >= s as isize {
                return Err(FilterError::new(format!(
                    "origin {o} for a footprint of {s} along axis {a} (scipy: invalid origin)"
                )));
            }
        }
        Ok(())
    }

    /// As [`Footprint::check_shape`], and the footprint reads at least one position.
    pub(crate) fn check(&self) -> Result<(), FilterError> {
        self.check_shape()?;
        if self.size() == 0 {
            return Err(FilterError::new("a footprint with no element"));
        }
        Ok(())
    }
}

/// The box a run's outputs read: per axis, the block index each of its positions reads
/// (None: the constant), and the box's C strides.
pub(crate) struct Window {
    maps: Vec<Vec<Option<usize>>>,
    pub ext: Vec<usize>,
    pub out_len: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Window {
    /// The box for a footprint on a block of `shape` holding, per axis, the part
    /// `spans[a]` describes (as for `correlate1d`); None when there is no output. Every
    /// position the mode maps a reach to must be in the block.
    pub fn new(
        fp: &Footprint,
        mode: Mode,
        input_len: usize,
        shape: &[usize],
        spans: &[Span],
        output_len: usize,
    ) -> Result<Option<Self>, FilterError> {
        let n = fp.shape.len();
        if n == 0 || shape.len() != n || spans.len() != n {
            return Err(FilterError::new(format!(
                "a {}-d footprint on a {}-d block with {} spans",
                n,
                shape.len(),
                spans.len()
            )));
        }
        if input_len != shape.iter().product::<usize>() {
            return Err(FilterError::new(format!(
                "{input_len} elements for a block of shape {shape:?}"
            )));
        }
        let out_len: Vec<usize> = spans.iter().map(|s| s.out_len).collect();
        if output_len != out_len.iter().product::<usize>() {
            return Err(FilterError::new(format!(
                "{output_len} outputs for {out_len:?}"
            )));
        }
        if output_len == 0 {
            return Ok(None);
        }
        let mut maps: Vec<Vec<Option<usize>>> = Vec::with_capacity(n);
        for a in 0..n {
            let s = spans[a];
            let (before, _) = fp.reach(a);
            let ext = s.out_len + fp.shape[a] - 1;
            if s.start + shape[a] > s.len || s.out_start + s.out_len > s.len {
                return Err(FilterError::new(format!(
                    "a span {s:?} for a block of {} along axis {a}",
                    shape[a]
                )));
            }
            let mut m = Vec::with_capacity(ext);
            for i in 0..ext {
                let p = s.out_start as i64 - before as i64 + i as i64;
                match mode.index(p, s.len as i64) {
                    None => m.push(None),
                    Some(g) => {
                        let g = g as usize;
                        if g < s.start || g >= s.start + shape[a] {
                            return Err(FilterError::new(format!(
                                "position {p} along axis {a} reads {g}, outside the block \
                                 ({} .. {})",
                                s.start,
                                s.start + shape[a]
                            )));
                        }
                        m.push(Some(g - s.start));
                    }
                }
            }
            maps.push(m);
        }
        let ext: Vec<usize> = maps.iter().map(|m| m.len()).collect();
        let mut strides = vec![1usize; n];
        for a in (0..n - 1).rev() {
            strides[a] = strides[a + 1] * ext[a + 1];
        }
        Ok(Some(Self {
            maps,
            ext,
            out_len,
            strides,
        }))
    }

    /// Whether the block is the box itself (nothing to extend).
    pub fn is_block(&self, shape: &[usize]) -> bool {
        (0..shape.len()).all(|a| {
            self.ext[a] == shape[a] && self.maps[a].iter().enumerate().all(|(i, &g)| g == Some(i))
        })
    }

    /// The block extended to the box: each element converted by `conv`, `fill` where
    /// any axis reads the constant.
    pub fn extend<T: Copy, V: Copy>(
        &self,
        input: &[T],
        shape: &[usize],
        conv: impl Fn(T) -> V,
        fill: V,
    ) -> Vec<V> {
        let n = shape.len();
        let total: usize = self.ext.iter().product();
        let mut out = Vec::with_capacity(total);
        let last = self.ext[n - 1];
        let in_last = shape[n - 1];
        let mut idx = vec![0usize; n - 1];
        for _ in 0..total / last {
            let mut src = Some(0usize);
            for a in 0..n - 1 {
                src = match (src, self.maps[a][idx[a]]) {
                    (Some(s), Some(g)) => Some(s * shape[a] + g),
                    _ => None,
                };
            }
            match src {
                None => out.extend(std::iter::repeat_n(fill, last)),
                Some(s) => {
                    let row = &input[s * in_last..(s + 1) * in_last];
                    out.extend(
                        self.maps[n - 1]
                            .iter()
                            .map(|g| g.map_or(fill, |g| conv(row[g]))),
                    );
                }
            }
            for a in (0..n - 1).rev() {
                idx[a] += 1;
                if idx[a] < self.ext[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        out
    }

    /// The footprint's positions as offsets in the box.
    pub fn offsets(&self, positions: &[Vec<usize>]) -> Vec<usize> {
        positions
            .iter()
            .map(|p| p.iter().zip(&self.strides).map(|(i, s)| i * s).sum())
            .collect()
    }

    /// Where output line `o` (C order over every axis but the last) starts in the box.
    pub fn line_base(&self, o: usize) -> usize {
        let n = self.out_len.len();
        let mut rem = o;
        let mut base = 0usize;
        for a in (0..n - 1).rev() {
            base += (rem % self.out_len[a]) * self.strides[a];
            rem /= self.out_len[a];
        }
        base
    }

    /// Run `line(scratch, o, dst)` for every output line `o` (`dst` its outputs), in
    /// parallel on the current rayon pool when `parallel`.
    pub fn lines<U: Send, S: Send>(
        &self,
        parallel: bool,
        output: &mut [U],
        scratch: impl Fn() -> S + Sync + Send,
        line: impl Fn(&mut S, usize, &mut [U]) + Sync + Send,
    ) {
        let last = *self.out_len.last().unwrap_or(&1);
        if parallel {
            output
                .par_chunks_mut(last)
                .enumerate()
                .for_each_init(&scratch, |s, (o, dst)| line(s, o, dst));
        } else {
            let mut s = scratch();
            for (o, dst) in output.chunks_mut(last).enumerate() {
                line(&mut s, o, dst);
            }
        }
    }
}
