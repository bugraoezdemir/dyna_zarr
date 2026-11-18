//! Lazy views: arrays defined over other arrays, read only when asked. A view maps the
//! region asked of it to regions of its sources and, where it can, hands them the
//! caller's destination directly (a slice, a concatenation's parts, an axis added or
//! removed); otherwise it reads into a scratch buffer and copies (a transpose, a
//! stepped slice, a type conversion).

use std::sync::Arc;

use rayon::prelude::*;

use crate::cast::converter;
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::pool;
use crate::source::{c_strides, read, region, Source, SourceRef};
use crate::ReadError;

/// Source elements per stepped axis per block when the source has no storage grid.
const STEP_BLOCK: u64 = 256;

/// `source[start:stop:step]` per axis (step >= 1).
pub struct Slice {
    src: SourceRef,
    start: Vec<u64>,
    step: Vec<u64>,
    shape: Vec<u64>,
}

impl Slice {
    pub fn new(
        src: SourceRef,
        start: &[u64],
        stop: &[u64],
        step: &[u64],
    ) -> Result<Self, ReadError> {
        region(src.shape(), start, stop)?;
        if step.len() != start.len() || step.contains(&0) {
            return Err(ReadError::new(format!(
                "steps {step:?}: one step >= 1 per axis"
            )));
        }
        let shape = (0..start.len())
            .map(|i| (stop[i] - start[i]).div_ceil(step[i]))
            .collect();
        Ok(Self {
            src,
            start: start.to_vec(),
            step: step.to_vec(),
            shape,
        })
    }
}

impl Source for Slice {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        if self.step.iter().all(|s| *s == 1) {
            self.src.grid()
        } else {
            None
        }
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
        let n = r.len();
        if r.contains(&0) {
            return Ok(());
        }
        if self.step.iter().all(|s| *s == 1) {
            let s: Vec<u64> = (0..n).map(|i| self.start[i] + start[i]).collect();
            let e: Vec<u64> = (0..n).map(|i| s[i] + r[i]).collect();
            return self.src.read_into(&s, &e, out, threads);
        }
        // stepped axes in blocks of about one storage unit of the source, each block's
        // contiguous span read into scratch and every step-th element copied
        let grid = self.src.grid();
        let per_block: Vec<u64> = (0..n)
            .map(|i| {
                if self.step[i] == 1 {
                    r[i]
                } else {
                    let unit = grid.as_ref().map_or(STEP_BLOCK, |g| g[i]);
                    (unit / self.step[i]).max(1)
                }
            })
            .collect();
        let blocks: Vec<Vec<(u64, u64)>> = (0..n)
            .map(|i| {
                (0..r[i])
                    .step_by(per_block[i] as usize)
                    .map(|o| (o, (o + per_block[i]).min(r[i])))
                    .collect()
            })
            .collect();
        let item = self.dtype().size();
        let mut combo = vec![0usize; n];
        loop {
            let b: Vec<(u64, u64)> = (0..n).map(|i| blocks[i][combo[i]]).collect();
            let s: Vec<u64> = (0..n)
                .map(|i| self.start[i] + (start[i] + b[i].0) * self.step[i])
                .collect();
            let e: Vec<u64> = (0..n)
                .map(|i| self.start[i] + (start[i] + b[i].1 - 1) * self.step[i] + 1)
                .collect();
            let span: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
            let scratch = read(self.src.as_ref(), &s, &e, threads)?;
            let strides: Vec<usize> = c_strides(&span, item)
                .iter()
                .zip(&self.step)
                .map(|(st, k)| st * *k as usize)
                .collect();
            let sub: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
            let at: Vec<u64> = (0..n).map(|i| out.at[i] + b[i].0).collect();
            out.moved(at)
                .copy_from(&sub, &scratch, &strides, item, None);
            let mut ax = n;
            loop {
                if ax == 0 {
                    return Ok(());
                }
                ax -= 1;
                combo[ax] += 1;
                if combo[ax] < blocks[ax].len() {
                    break;
                }
                combo[ax] = 0;
            }
        }
    }
}

/// Axes reordered: output axis i is source axis `perm[i]`.
pub struct Transpose {
    src: SourceRef,
    perm: Vec<usize>,
    shape: Vec<u64>,
}

impl Transpose {
    pub fn new(src: SourceRef, perm: &[usize]) -> Result<Self, ReadError> {
        let n = src.shape().len();
        let mut seen = vec![false; n];
        if perm.len() != n
            || perm
                .iter()
                .any(|&p| p >= n || std::mem::replace(&mut seen[p], true))
        {
            return Err(ReadError::new(format!(
                "{perm:?} is not a permutation of {n} axes"
            )));
        }
        let shape = perm.iter().map(|&p| src.shape()[p]).collect();
        Ok(Self {
            src,
            perm: perm.to_vec(),
            shape,
        })
    }
}

impl Source for Transpose {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        self.src
            .grid()
            .map(|g| self.perm.iter().map(|&p| g[p]).collect())
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
        let n = r.len();
        let mut s = vec![0u64; n];
        let mut e = vec![0u64; n];
        for i in 0..n {
            s[self.perm[i]] = start[i];
            e[self.perm[i]] = stop[i];
        }
        let item = self.dtype().size();
        let scratch = read(self.src.as_ref(), &s, &e, threads)?;
        let span: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
        let cs = c_strides(&span, item);
        let strides: Vec<usize> = self.perm.iter().map(|&p| cs[p]).collect();
        out.copy_from(&r, &scratch, &strides, item, None);
        Ok(())
    }
}

/// A new axis of length 1 at `axis`.
pub struct ExpandDims {
    src: SourceRef,
    axis: usize,
    shape: Vec<u64>,
}

impl ExpandDims {
    pub fn new(src: SourceRef, axis: usize) -> Result<Self, ReadError> {
        if axis > src.shape().len() {
            return Err(ReadError::new(format!(
                "axis {axis} for an array of {} axes",
                src.shape().len()
            )));
        }
        let mut shape = src.shape().to_vec();
        shape.insert(axis, 1);
        Ok(Self { src, axis, shape })
    }
}

impl Source for ExpandDims {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        self.src.grid().map(|mut g| {
            g.insert(self.axis, 1);
            g
        })
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
        if r.contains(&0) {
            return Ok(());
        }
        let mut s = start.to_vec();
        let mut e = stop.to_vec();
        s.remove(self.axis);
        e.remove(self.axis);
        if let Some(inner) = out.without_axis(self.axis) {
            return self.src.read_into(&s, &e, &inner, threads);
        }
        // a new last axis inside a wider destination: not one C-ordered block
        let item = self.dtype().size();
        let scratch = read(self.src.as_ref(), &s, &e, threads)?;
        let span: Vec<u64> = (0..s.len()).map(|i| e[i] - s[i]).collect();
        let mut strides = c_strides(&span, item);
        strides.insert(self.axis, 0);
        out.copy_from(&r, &scratch, &strides, item, None);
        Ok(())
    }
}

/// Axis `axis`, of length 1, removed.
pub struct Squeeze {
    src: SourceRef,
    axis: usize,
    shape: Vec<u64>,
}

impl Squeeze {
    pub fn new(src: SourceRef, axis: usize) -> Result<Self, ReadError> {
        if src.shape().get(axis) != Some(&1) {
            return Err(ReadError::new(format!(
                "axis {axis} of shape {:?} is not of length 1",
                src.shape()
            )));
        }
        let mut shape = src.shape().to_vec();
        shape.remove(axis);
        Ok(Self { src, axis, shape })
    }
}

impl Source for Squeeze {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        self.src.grid().map(|mut g| {
            g.remove(self.axis);
            g
        })
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
        let mut s = start.to_vec();
        let mut e = stop.to_vec();
        s.insert(self.axis, 0);
        e.insert(self.axis, 1);
        self.src
            .read_into(&s, &e, &out.with_axis(self.axis), threads)
    }
}

/// Arrays joined along `axis` (same shape and type otherwise).
pub struct Concat {
    parts: Vec<SourceRef>,
    axis: usize,
    offsets: Vec<u64>,
    shape: Vec<u64>,
}

impl Concat {
    pub fn new(parts: Vec<SourceRef>, axis: usize) -> Result<Self, ReadError> {
        let first = parts
            .first()
            .ok_or_else(|| ReadError::new("nothing to join".to_string()))?;
        let mut shape = first.shape().to_vec();
        if axis >= shape.len() {
            return Err(ReadError::new(format!(
                "axis {axis} for arrays of {} axes",
                shape.len()
            )));
        }
        let mut offsets = vec![0u64];
        shape[axis] = 0;
        for (k, p) in parts.iter().enumerate() {
            let s = p.shape();
            let same = s.len() == shape.len()
                && (0..s.len()).all(|i| i == axis || s[i] == first.shape()[i]);
            if !same || p.dtype() != first.dtype() {
                return Err(ReadError::new(format!(
                    "part {k} is {:?} {}, part 0 {:?} {}: only the join axis {axis} may differ",
                    s,
                    p.dtype().name(),
                    first.shape(),
                    first.dtype().name()
                )));
            }
            shape[axis] += s[axis];
            offsets.push(shape[axis]);
        }
        Ok(Self {
            parts,
            axis,
            offsets,
            shape,
        })
    }

    /// Arrays stacked along a new axis `axis` (all of one shape).
    pub fn stack(parts: Vec<SourceRef>, axis: usize) -> Result<Self, ReadError> {
        let expanded = parts
            .into_iter()
            .map(|p| ExpandDims::new(p, axis).map(|v| Arc::new(v) as SourceRef))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(expanded, axis)
    }
}

impl Source for Concat {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.parts[0].dtype()
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
        let a = self.axis;
        let jobs: Vec<(usize, u64, u64)> = (0..self.parts.len())
            .filter_map(|k| {
                let lo = self.offsets[k].max(start[a]);
                let hi = self.offsets[k + 1].min(stop[a]);
                (lo < hi).then_some((k, lo, hi))
            })
            .collect();
        let one = |&(k, lo, hi): &(usize, u64, u64)| -> Result<(), ReadError> {
            let mut s = start.to_vec();
            let mut e = stop.to_vec();
            s[a] = lo - self.offsets[k];
            e[a] = hi - self.offsets[k];
            let mut at = out.at.clone();
            at[a] += lo - start[a];
            self.parts[k].read_into(&s, &e, &out.moved(at), threads)
        };
        if threads > 1 && jobs.len() > 1 {
            // parts in parallel: their regions of the destination are disjoint
            pool(threads)?.install(|| jobs.par_iter().try_for_each(one))
        } else {
            jobs.iter().try_for_each(one)
        }
    }
}

/// Elements converted to another type.
pub struct Cast {
    src: SourceRef,
    to: DType,
}

impl Cast {
    pub fn new(src: SourceRef, to: DType) -> Result<Self, ReadError> {
        // refuse an unsupported conversion when the view is made, not at the first read
        converter(src.dtype(), to).map(drop)?;
        Ok(Self { src, to })
    }
}

impl Source for Cast {
    fn shape(&self) -> &[u64] {
        self.src.shape()
    }
    fn dtype(&self) -> DType {
        self.to
    }
    fn grid(&self) -> Option<Vec<u64>> {
        self.src.grid()
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        if self.src.dtype() == self.to {
            return self.src.read_into(start, stop, out, threads);
        }
        let r = region(self.shape(), start, stop)?;
        out.check(&r, self.to)?;
        let from = self.src.dtype();
        let scratch = read(self.src.as_ref(), start, stop, threads)?;
        let f = converter(from, self.to)?;
        out.copy_from(
            &r,
            &scratch,
            &c_strides(&r, from.size()),
            from.size(),
            Some(&*f),
        );
        Ok(())
    }
}
