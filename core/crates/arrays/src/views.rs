//! Lazy views: arrays defined over other arrays, read only when asked. A view maps the
//! region asked of it to regions of its sources and, where it can, hands them the
//! caller's destination directly (a slice, a concatenation's parts, an axis added or
//! removed); otherwise it reads into a scratch buffer and copies (a transpose, a
//! stepped slice, a type conversion).

use std::sync::Arc;

use crate::blocks::Blocks;
use crate::cast::converter;
use crate::downsample::scaled_grid;
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::{piece_plan, pieces};
use crate::source::{bytes_of, c_strides, read, region, Source, SourceRef, SMALL};
use crate::ReadError;

/// Source elements per stepped axis per block when the source has no storage grid.
const STEP_BLOCK: u64 = 256;

/// Source bytes per block of a stepped read over a view that reads a halo.
const HALO_SPAN: u64 = 64 << 20;

/// `source[start:stop:step]` as a view: a source that can compute only every step-th
/// element (a filter, [`Source::stepped`]) does, under an unstepped slice; any other is
/// read by a stepped [`Slice`].
pub fn sliced(
    src: SourceRef,
    start: &[u64],
    stop: &[u64],
    step: &[u64],
) -> Result<SourceRef, ReadError> {
    let plain = Slice::new(src.clone(), start, stop, step)?;
    if step.iter().all(|s| *s == 1) {
        return Ok(Arc::new(plain));
    }
    match src.stepped(start, step) {
        Some(v) => {
            let n = start.len();
            let zero = vec![0; n];
            let one = vec![1; n];
            Ok(Arc::new(Slice::new(v, &zero, &plain.shape, &one)?))
        }
        None => Ok(Arc::new(plain)),
    }
}

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
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    /// The source's grid in this view's elements (a hint for block sizes above it).
    fn grid(&self) -> Option<Vec<u64>> {
        scaled_grid(self.src.as_ref(), &self.step)
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
            let (s, e) = self.unstepped(start, &r);
            return self.src.read_into(&s, &e, out, threads);
        }
        // blocks of about one storage unit of the source on every axis, at most
        // `threads` at once; each block's contiguous span is read into scratch and every
        // step-th element copied
        let blocks = self.blocks(&r);
        let item = self.dtype().size();
        pieces(blocks.len(), threads, |k, each| {
            let b = blocks.get(k);
            let (s, e) = self.span(start, &b);
            let span: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
            let scratch = read(self.src.as_ref(), &s, &e, each)?;
            let strides: Vec<usize> = c_strides(&span, item)
                .iter()
                .zip(&self.step)
                .map(|(st, k)| st * *k as usize)
                .collect();
            let sub: Vec<u64> = b.iter().map(|(a, z)| z - a).collect();
            let at: Vec<u64> = (0..n).map(|i| out.at[i] + b[i].0).collect();
            out.moved(at)
                .copy_from(&sub, &scratch, &strides, item, None);
            Ok(())
        })
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        if r.contains(&0) {
            return Ok(0);
        }
        if self.step.iter().all(|s| *s == 1) {
            let (s, e) = self.unstepped(start, &r);
            return self.src.cost(&s, &e, threads);
        }
        let blocks = self.blocks(&r);
        let (open, each) = piece_plan(blocks.len(), threads);
        let item = self.dtype().size();
        let mut most = 0u64;
        for b in blocks.sample() {
            let (s, e) = self.span(start, &b);
            let span: Vec<u64> = (0..s.len()).map(|i| e[i] - s[i]).collect();
            most = most.max(bytes_of(&span, item) + SMALL + self.src.cost(&s, &e, each)?);
        }
        Ok(open as u64 * most)
    }
}

impl Slice {
    /// The source region of an unstepped read.
    fn unstepped(&self, start: &[u64], r: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let s: Vec<u64> = (0..r.len()).map(|i| self.start[i] + start[i]).collect();
        let e: Vec<u64> = (0..r.len()).map(|i| s[i] + r[i]).collect();
        (s, e)
    }

    /// A stepped read's blocks: about one storage unit of the source per axis (or
    /// STEP_BLOCK source elements per stepped axis without a grid).
    fn blocks(&self, r: &[u64]) -> Blocks {
        if self.src.reads_halo() {
            // over a filter: as few blocks as fit HALO_SPAN, halving the block's widest
            // axis (each block reads its halo; a block per chunk read each up to 9x in 2-D)
            let item = self.dtype().size() as u64;
            let span = |p: &[u64]| -> u64 {
                p.iter().zip(&self.step).map(|(a, s)| a * s).product::<u64>() * item
            };
            let mut per = r.to_vec();
            while span(&per) > HALO_SPAN && per.iter().any(|x| *x > 1) {
                let ax = (0..r.len())
                    .max_by_key(|&i| per[i] * self.step[i])
                    .unwrap_or(0);
                per[ax] = per[ax].div_ceil(2);
            }
            return Blocks::regular(r, &per);
        }
        let grid = self.src.grid();
        let per: Vec<u64> = (0..r.len())
            .map(|i| match &grid {
                Some(g) => (g[i] / self.step[i]).max(1),
                None if self.step[i] == 1 => r[i],
                None => (STEP_BLOCK / self.step[i]).max(1),
            })
            .collect();
        Blocks::regular(r, &per)
    }

    /// The contiguous source span of block `b` of a stepped read starting at `start`:
    /// to the last element taken, and on to just before the next block's first (inside
    /// the source), so that a block of one storage unit reads that whole unit (zarrs then
    /// decodes it straight in, without its partial-chunk copies).
    fn span(&self, start: &[u64], b: &[(u64, u64)]) -> (Vec<u64>, Vec<u64>) {
        let n = b.len();
        let src = self.src.shape();
        let s = (0..n)
            .map(|i| self.start[i] + (start[i] + b[i].0) * self.step[i])
            .collect();
        let e = (0..n)
            .map(|i| {
                let last = self.start[i] + (start[i] + b[i].1 - 1) * self.step[i] + 1;
                let next = self.start[i] + (start[i] + b[i].1) * self.step[i];
                last.max(next.min(src[i]))
            })
            .collect();
        (s, e)
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
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }
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
        let (s, e) = self.source_region(start, stop);
        let item = self.dtype().size();
        let scratch = read(self.src.as_ref(), &s, &e, threads)?;
        let span: Vec<u64> = (0..n).map(|i| e[i] - s[i]).collect();
        let cs = c_strides(&span, item);
        let strides: Vec<usize> = self.perm.iter().map(|&p| cs[p]).collect();
        out.copy_from(&r, &scratch, &strides, item, None);
        Ok(())
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        let (s, e) = self.source_region(start, stop);
        Ok(bytes_of(&r, self.dtype().size()) + SMALL + self.src.cost(&s, &e, threads)?)
    }
}

impl Transpose {
    fn source_region(&self, start: &[u64], stop: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let n = start.len();
        let mut s = vec![0u64; n];
        let mut e = vec![0u64; n];
        for i in 0..n {
            s[self.perm[i]] = start[i];
            e[self.perm[i]] = stop[i];
        }
        (s, e)
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
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }
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

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        if r.contains(&0) {
            return Ok(0);
        }
        let mut s = start.to_vec();
        let mut e = stop.to_vec();
        s.remove(self.axis);
        e.remove(self.axis);
        // a new last axis may need the scratch copy (when the destination is wider)
        let scratch = if self.axis + 1 == self.shape.len() {
            bytes_of(&r, self.dtype().size()) + SMALL
        } else {
            0
        };
        Ok(scratch + self.src.cost(&s, &e, threads)?)
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
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }
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

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        region(&self.shape, start, stop)?;
        let mut s = start.to_vec();
        let mut e = stop.to_vec();
        s.insert(self.axis, 0);
        e.insert(self.axis, 1);
        self.src.cost(&s, &e, threads)
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
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        for p in &self.parts {
            p.stores(out)
        }
    }
    fn reads_halo(&self) -> bool {
        self.parts.iter().any(|p| p.reads_halo())
    }
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
        let jobs = self.jobs(start, stop);
        // parts at most `threads` at once (their regions of the destination are disjoint)
        pieces(jobs.len(), threads, |j, each| {
            let (k, s, e) = &jobs[j];
            let mut at = out.at.clone();
            at[a] += s[a] + self.offsets[*k] - start[a];
            self.parts[*k].read_into(s, e, &out.moved(at), each)
        })
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        region(&self.shape, start, stop)?;
        let jobs = self.jobs(start, stop);
        let (open, each) = piece_plan(jobs.len(), threads);
        let mut costs = jobs
            .iter()
            .map(|(k, s, e)| self.parts[*k].cost(s, e, each))
            .collect::<Result<Vec<u64>, _>>()?;
        // the `open` most costly parts at once
        costs.sort_unstable_by(|x, y| y.cmp(x));
        Ok(costs.iter().take(open).sum())
    }
}

impl Concat {
    /// The parts a read of `start .. stop` touches: (part, its start, its stop).
    fn jobs(&self, start: &[u64], stop: &[u64]) -> Vec<(usize, Vec<u64>, Vec<u64>)> {
        let a = self.axis;
        (0..self.parts.len())
            .filter_map(|k| {
                let lo = self.offsets[k].max(start[a]);
                let hi = self.offsets[k + 1].min(stop[a]);
                (lo < hi).then(|| {
                    let mut s = start.to_vec();
                    let mut e = stop.to_vec();
                    s[a] = lo - self.offsets[k];
                    e[a] = hi - self.offsets[k];
                    (k, s, e)
                })
            })
            .collect()
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
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        self.src.reads_halo()
    }
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

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let src = self.src.cost(start, stop, threads)?;
        if self.src.dtype() == self.to {
            return Ok(src);
        }
        let r = region(self.shape(), start, stop)?;
        Ok(bytes_of(&r, self.src.dtype().size()) + SMALL + src)
    }
}
