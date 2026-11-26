//! Filters as lazy views (Workplan WP15, task DR7): chains of 1-D passes from
//! `dyna-zarr-filters` (correlations, uniform windows, sliding minima and maxima), with
//! scipy.ndimage's values; one chain, or several each applied to the source and combined
//! (scipy's generic_laplace and generic_gradient_magnitude).
//!
//! A region is read as its input box: the region grown by every pass's reach, cut back at
//! the array's edges, plus whatever positions the mode maps the reach past an edge to
//! (wrap mode reaches the far end of the axis). A chain's passes then shrink the box as
//! they go: pass `i` computes only what the later passes need. So any region gives the
//! values of the whole filtered array, as scipy computes them, axis by axis in the
//! element type.
//!
//! A stepped filter (`filter[phase::step]`, [`Separable::with_steps`]: a filter followed
//! by a stride) computes only the elements it keeps: on a stepped axis the last pass
//! along it computes every step-th position only, and the passes after it run on the
//! decimated data (a stepped axis no pass filters is decimated as read). The kept values
//! are exactly the full filter's (each output's sum is the same, in the same order); a
//! 2 x 2 Gaussian downscale does about a third of the work.

use dyna_zarr_filters as fl;
use fl::Element;

use crate::downsample::{scaled_grid, split, split_cost, DEFAULT_MAX_READ};
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::install;
use crate::source::{read_vec, region, Source, SourceRef, SMALL};
use std::sync::Arc;
use crate::ReadError;

pub use fl::{Combine, Line as FilterLine, Mode as FilterMode, Pass};

/// A box: start and stop per axis, in the source's positions (on an axis held decimated,
/// the positions start, start + step, ... before stop).
type Box_ = (Vec<u64>, Vec<u64>);

/// The data between two passes of a chain: its box, and per axis whether it is held
/// decimated (every step-th position).
#[derive(Clone)]
struct Stage {
    at: Box_,
    dec: Vec<bool>,
}

/// `chains` of passes, each applied in order to `src`; several are combined by `combine`.
/// With steps: element `j` is the filter's element `phase + j * step` per axis.
#[derive(Clone)]
pub struct Separable {
    src: SourceRef,
    chains: Vec<Vec<Pass>>,
    combine: Option<Combine>,
    max_read: u64,
    phase: Vec<u64>,
    step: Vec<u64>,
    shape: Vec<u64>,
}

impl Separable {
    /// One chain of `passes`.
    pub fn new(src: SourceRef, passes: Vec<Pass>) -> Result<Self, ReadError> {
        Self::terms(src, vec![passes], None)
    }

    /// Several chains, each applied to `src`, combined by `combine` (needed for more
    /// than one chain).
    pub fn terms(
        src: SourceRef,
        chains: Vec<Vec<Pass>>,
        combine: Option<Combine>,
    ) -> Result<Self, ReadError> {
        let n = src.shape().len();
        if chains.is_empty() {
            return Err(ReadError::new(
                "a filter needs at least one chain".to_string(),
            ));
        }
        if chains.len() > 1 && combine.is_none() {
            return Err(ReadError::new(
                "several chains need a way to combine them".to_string(),
            ));
        }
        for p in chains.iter().flatten() {
            if p.axis >= n {
                return Err(ReadError::new(format!(
                    "a pass along axis {} of a {n}-d array",
                    p.axis
                )));
            }
            let taps = p.line.taps();
            let o = p.line.origin();
            if taps == 0 || o < -((taps / 2) as isize) || o > ((taps - 1) / 2) as isize {
                return Err(ReadError::new(format!(
                    "a pass of size {taps} with origin {o}"
                )));
            }
            if let FilterLine::Correlate { weights, .. } = &p.line {
                if weights.iter().any(|w| !w.is_finite()) {
                    return Err(ReadError::new("a pass needs finite weights".to_string()));
                }
            }
        }
        let float = matches!(src.dtype(), DType::Float32 | DType::Float64);
        if combine == Some(Combine::Hypot) && !float {
            return Err(ReadError::new(format!(
                "a gradient magnitude of {} is not supported (floating-point only)",
                src.dtype().name()
            )));
        }
        match src.dtype() {
            DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64
            | DType::Float32
            | DType::Float64 => {}
            other => {
                return Err(ReadError::new(format!(
                    "filtering {} is not supported (scipy.ndimage does not filter it either)",
                    other.name()
                )))
            }
        }
        let shape = src.shape().to_vec();
        Ok(Self {
            src,
            chains,
            combine,
            max_read: DEFAULT_MAX_READ,
            phase: vec![0; n],
            step: vec![1; n],
            shape,
        })
    }

    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    /// This filter's elements `phase + j * step` per axis (`self[phase::step]`), computing
    /// only those.
    pub fn with_steps(&self, phase: &[u64], step: &[u64]) -> Result<Self, ReadError> {
        let n = self.shape.len();
        if phase.len() != n || step.len() != n || step.contains(&0) {
            return Err(ReadError::new(format!(
                "phase {phase:?} and steps {step:?}: one of each (steps >= 1) per axis"
            )));
        }
        let mut out = self.clone();
        for i in 0..n {
            out.phase[i] = self.phase[i] + phase[i] * self.step[i];
            out.step[i] = self.step[i] * step[i];
            let len = self.src.shape()[i];
            out.shape[i] = len.saturating_sub(out.phase[i]).div_ceil(out.step[i]);
        }
        Ok(out)
    }

    /// Elements held on axis `a` of a stage's data.
    fn extent(&self, st: &Stage, a: usize) -> u64 {
        let (lo, hi) = (st.at.0[a], st.at.1[a]);
        if hi <= lo {
            0
        } else if st.dec[a] {
            (hi - lo - 1) / self.step[a] + 1
        } else {
            hi - lo
        }
    }

    fn stage_shape(&self, st: &Stage) -> Vec<u64> {
        (0..st.dec.len()).map(|a| self.extent(st, a)).collect()
    }

    /// What every pass of `chain` reads: `stages[i]` is pass i's input, the last is the
    /// region (`start .. stop` in this view's elements).
    fn stages(&self, chain: &[Pass], start: &[u64], stop: &[u64]) -> Vec<Stage> {
        let shape = self.src.shape();
        let n = start.len();
        let lo: Vec<u64> = (0..n).map(|a| self.phase[a] + start[a] * self.step[a]).collect();
        let hi: Vec<u64> = (0..n)
            .map(|a| {
                if stop[a] <= start[a] {
                    lo[a]
                } else {
                    self.phase[a] + (stop[a] - 1) * self.step[a] + 1
                }
            })
            .collect();
        let mut cur = Stage {
            at: (lo, hi),
            dec: (0..n).map(|a| self.step[a] > 1).collect(),
        };
        let mut out = vec![cur.clone()];
        for p in chain.iter().rev() {
            let a = p.axis;
            let len = shape[a] as i64;
            let (before, after) = p.reach();
            // the outputs lo, lo + step, ... (or lo .. hi) read lo - before .. hi + after
            let lo_p = cur.at.0[a] as i64 - before as i64;
            let hi_p = cur.at.1[a] as i64 + after as i64;
            let (l, h) = mapped_range(p.mode, len, lo_p, hi_p);
            cur.at.0[a] = l as u64;
            cur.at.1[a] = h as u64;
            cur.dec[a] = false; // a pass reads its axis whole
            out.push(cur.clone());
        }
        out.reverse();
        out
    }

    /// Every chain's stages, and the box read from the source (covering every chain's
    /// first stage, contiguous).
    fn plan(&self, start: &[u64], stop: &[u64]) -> (Vec<Vec<Stage>>, Box_) {
        let all: Vec<Vec<Stage>> = self
            .chains
            .iter()
            .map(|c| self.stages(c, start, stop))
            .collect();
        let mut union = all[0][0].at.clone();
        for b in &all[1..] {
            for k in 0..union.0.len() {
                union.0[k] = union.0[k].min(b[0].at.0[k]);
                union.1[k] = union.1[k].max(b[0].at.1[k]);
            }
        }
        (all, union)
    }

    fn part<T: Element + Default>(
        &self,
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (plans, union) = self.plan(start, &stop);
        let full = self.src.shape();
        let parallel = threads > 1;
        let single = self.chains.len() == 1;
        let mut read = Some(read_vec::<T>(
            self.src.as_ref(),
            &union.0,
            &union.1,
            threads,
        )?);
        let mut acc: Option<Vec<T>> = None;
        for (chain, stages) in self.chains.iter().zip(&plans) {
            // the chain's input: the read box itself, or the part it needs (decimated on
            // the stepped axes no pass filters)
            let first = &stages[0];
            let as_read = first.at == union && !first.dec.iter().any(|d| *d);
            let mut cur: Vec<T> = if single && as_read {
                read.take().unwrap_or_default()
            } else {
                gather(read.as_deref().unwrap_or(&[]), &union, first, &self.step)
            };
            for (i, p) in chain.iter().enumerate() {
                let (b, next) = (&stages[i], &stages[i + 1]);
                let a = p.axis;
                let shape: Vec<usize> =
                    self.stage_shape(b).iter().map(|&x| x as usize).collect();
                let out_len = self.extent(next, a) as usize;
                let span = fl::Span {
                    len: full[a] as usize,
                    start: b.at.0[a] as usize,
                    out_start: next.at.0[a] as usize,
                    out_len,
                };
                let step = if next.dec[a] { self.step[a] as usize } else { 1 };
                let count = cur.len() / shape[a].max(1) * out_len;
                let mut out_buf = vec![T::default(); count];
                let mut run = || p.run_stepped(parallel, &cur, &shape, span, step, &mut out_buf);
                let done = if parallel {
                    install(threads, &mut run)?
                } else {
                    run()
                };
                done.map_err(ReadError::filter)?;
                cur = out_buf;
            }
            // `cur` is this chain's region
            match (&mut acc, self.combine) {
                (None, Some(c)) => {
                    c.start(&mut cur);
                    acc = Some(cur);
                }
                (None, None) => acc = Some(cur),
                (Some(a), Some(c)) => c.add(a, &cur),
                (Some(_), None) => {
                    return Err(ReadError::new(
                        "several chains need a way to combine them".to_string(),
                    ))
                }
            }
        }
        let mut res = acc.unwrap_or_default();
        if let Some(c) = self.combine {
            c.finish(&mut res).map_err(ReadError::filter)?;
        }
        // `res` is the region, C-ordered
        let item = std::mem::size_of::<T>();
        let mut strides = vec![item; n];
        for k in (0..n.saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * r[k + 1] as usize;
        }
        // SAFETY: plain numeric elements, viewed as their bytes
        let bytes =
            unsafe { std::slice::from_raw_parts(res.as_ptr() as *const u8, res.len() * item) };
        out.copy_from(r, bytes, &strides, item, None);
        Ok(())
    }

    /// One piece at its fullest: the read box with what reading it costs; then for each
    /// chain, beside what several chains keep (the read box and the result so far), its
    /// input and each pass's input and output boxes and buffers (`Pass::bytes`: a row
    /// table, and per thread a line or a block's scratch).
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (plans, union) = self.plan(start, &stop);
        let item = self.src.dtype().size() as u64;
        let size = |st: &Stage| self.stage_shape(st).iter().product::<u64>() * item;
        let raw = (0..n).map(|k| union.1[k] - union.0[k]).product::<u64>() * item;
        let mut most = raw + self.src.cost(&union.0, &union.1, threads)?;
        let region = r.iter().product::<u64>() * item;
        let kept = if self.chains.len() == 1 {
            0
        } else {
            raw + region
        };
        for (chain, stages) in self.chains.iter().zip(&plans) {
            // a chain's input, gathered beside the read box
            most = most.max(raw + kept + size(&stages[0]));
            for (i, p) in chain.iter().enumerate() {
                let a = p.axis;
                let out_len = self.extent(&stages[i + 1], a) as usize;
                // a stepped running window computes its contiguous run first
                let step = if stages[i + 1].dec[a] { self.step[a] } else { 1 };
                let run = if step > 1 && !matches!(p.line, FilterLine::Correlate { .. }) {
                    size(&stages[i + 1]) * step
                } else {
                    0
                };
                let (shared, each) = p.bytes(out_len.max(1) * step as usize);
                let tables = shared + threads.max(1) as u64 * each;
                most = most.max(kept + size(&stages[i]) + size(&stages[i + 1]) + run + tables);
            }
        }
        Ok(most + SMALL)
    }

    /// Input bytes of a piece (its read box twice), the measure for splitting a large
    /// request.
    fn box_bytes(&self, start: &[u64], r: &[u64]) -> u64 {
        let n = r.len();
        let stop: Vec<u64> = (0..n).map(|i| start[i] + r[i]).collect();
        let (_, b) = self.plan(start, &stop);
        (0..n).map(|k| b.1[k] - b.0[k]).product::<u64>() * self.src.dtype().size() as u64 * 2
    }
}

/// The positions of an axis of length `n` that positions `lo_p .. hi_p` read: those
/// inside the axis, and those the mode maps the ones past an edge to (wrap reaches the
/// far end).
pub(crate) fn mapped_range(mode: FilterMode, n: i64, lo_p: i64, hi_p: i64) -> (i64, i64) {
    let (mut lo, mut hi) = (lo_p.max(0), hi_p.min(n));
    for q in (lo_p..0.min(hi_p)).chain(n.max(lo_p)..hi_p) {
        if lo == 0 && hi == n {
            break;
        }
        if let Some(g) = mode.index(q, n) {
            lo = lo.min(g);
            hi = hi.max(g + 1);
        }
    }
    (lo, hi)
}

/// The part `want` of a C-ordered block `data` that holds the box `have` (contiguous),
/// every `step[a]`-th position on the axes `want` holds decimated.
fn gather<T: Copy>(data: &[T], have: &Box_, want: &Stage, step: &[u64]) -> Vec<T> {
    let n = have.0.len();
    let hs: Vec<usize> = (0..n).map(|k| (have.1[k] - have.0[k]) as usize).collect();
    let st: Vec<usize> = (0..n)
        .map(|k| if want.dec[k] { step[k] as usize } else { 1 })
        .collect();
    let ws: Vec<usize> = (0..n)
        .map(|k| {
            let (lo, hi) = (want.at.0[k], want.at.1[k]);
            if hi <= lo {
                0
            } else {
                ((hi - lo - 1) as usize) / st[k] + 1
            }
        })
        .collect();
    let off: Vec<usize> = (0..n).map(|k| (want.at.0[k] - have.0[k]) as usize).collect();
    let count: usize = ws.iter().product();
    let mut out = Vec::with_capacity(count);
    if count == 0 || n == 0 {
        return out;
    }
    let inner = ws[n - 1];
    let mut idx = vec![0usize; n - 1];
    for _ in 0..count / inner {
        let mut at = 0usize;
        for k in 0..n - 1 {
            at = at * hs[k] + off[k] + idx[k] * st[k];
        }
        at = at * hs[n - 1] + off[n - 1];
        if st[n - 1] == 1 {
            out.extend_from_slice(&data[at..at + inner]);
        } else {
            out.extend((0..inner).map(|j| data[at + j * st[n - 1]]));
        }
        for k in (0..n - 1).rev() {
            idx[k] += 1;
            if idx[k] < ws[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

impl Source for Separable {
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn reads_halo(&self) -> bool {
        true
    }
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        scaled_grid(self.src.as_ref(), &self.step)
    }
    /// `self[phase::step]` computing only the kept elements.
    fn stepped(&self, phase: &[u64], step: &[u64]) -> Option<SourceRef> {
        self.with_steps(phase, step)
            .ok()
            .map(|v| Arc::new(v) as SourceRef)
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
        let bytes = |s: &[u64], r: &[u64]| self.box_bytes(s, r);
        let run = |s: &[u64], r: &[u64], o: &Out| match self.src.dtype() {
            DType::Int8 => self.part::<i8>(s, r, o, threads),
            DType::Int16 => self.part::<i16>(s, r, o, threads),
            DType::Int32 => self.part::<i32>(s, r, o, threads),
            DType::Int64 => self.part::<i64>(s, r, o, threads),
            DType::UInt8 => self.part::<u8>(s, r, o, threads),
            DType::UInt16 => self.part::<u16>(s, r, o, threads),
            DType::UInt32 => self.part::<u32>(s, r, o, threads),
            DType::UInt64 => self.part::<u64>(s, r, o, threads),
            DType::Float32 => self.part::<f32>(s, r, o, threads),
            DType::Float64 => self.part::<f64>(s, r, o, threads),
            other => Err(ReadError::new(format!("filtering {}", other.name()))),
        };
        split(start, &r, out, self.max_read, &bytes, &run)
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        let bytes = |s: &[u64], r: &[u64]| self.box_bytes(s, r);
        let cost = |s: &[u64], r: &[u64]| self.part_cost(s, r, threads);
        split_cost(start, &r, self.max_read, &bytes, &cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::read;
    use std::sync::Arc;

    /// An in-memory f64 source for the tests.
    struct Mem {
        shape: Vec<u64>,
        data: Vec<f64>,
    }

    impl Source for Mem {
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            DType::Float64
        }
        fn read_into(
            &self,
            start: &[u64],
            stop: &[u64],
            out: &Out,
            _threads: usize,
        ) -> Result<(), ReadError> {
            let r = region(&self.shape, start, stop)?;
            let n = r.len();
            let mut strides = vec![8usize; n];
            for k in (0..n - 1).rev() {
                strides[k] = strides[k + 1] * self.shape[k + 1] as usize;
            }
            let off: usize = (0..n).map(|k| start[k] as usize * strides[k]).sum();
            // SAFETY: f64s as bytes
            let bytes = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 8)
            };
            out.copy_from(&r, &bytes[off..], &strides, 8, None);
            Ok(())
        }
        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(0)
        }
    }

    const SHAPE: [usize; 3] = [7, 9, 11];

    fn source() -> (SourceRef, Vec<f64>) {
        let data: Vec<f64> = (0..693).map(|v| ((v * 7919) % 101) as f64 - 50.0).collect();
        let src: SourceRef = Arc::new(Mem {
            shape: SHAPE.iter().map(|&s| s as u64).collect(),
            data: data.clone(),
        });
        (src, data)
    }

    /// Every region read alone equals that region of `want` (the whole filtered array).
    fn regions_equal(view: &Separable, want: &[f64], what: &str) {
        for (s, e) in [
            ([0, 0, 0], [7, 9, 11]),
            ([0, 0, 0], [1, 1, 1]),
            ([6, 8, 10], [7, 9, 11]),
            ([2, 3, 4], [5, 6, 9]),
            ([0, 4, 9], [3, 9, 11]),
        ] {
            for threads in [1, 3] {
                let got = read(view, &s, &e, threads).unwrap();
                // SAFETY: the read gave f64s
                let got: &[f64] = unsafe {
                    std::slice::from_raw_parts(got.as_ptr() as *const f64, got.len() / 8)
                };
                let mut k = 0;
                for i in s[0]..e[0] {
                    for j in s[1]..e[1] {
                        for l in s[2]..e[2] {
                            let w = want[((i * 9 + j) * 11 + l) as usize];
                            assert_eq!(got[k], w, "{what} {s:?}..{e:?} at {i},{j},{l}");
                            k += 1;
                        }
                    }
                }
            }
        }
    }

    const MODES: [FilterMode; 5] = [
        FilterMode::Reflect,
        FilterMode::Mirror,
        FilterMode::Nearest,
        FilterMode::Wrap,
        FilterMode::Constant(2.5),
    ];

    #[test]
    fn any_region_equals_the_whole_filtered_array() {
        let (src, data) = source();
        for mode in MODES {
            let passes = vec![
                Pass::correlate(0, vec![0.25, 0.5, 0.25], mode),
                Pass::correlate(1, vec![0.1, 0.2, 0.3, 0.4], mode),
                Pass::correlate(2, fl::gaussian_weights(2.0, 0, 8), mode),
            ];
            let mut want = vec![0f64; data.len()];
            fl::separable(&data, &SHAPE, &passes, &mut want).unwrap();
            let view = Separable::new(src.clone(), passes).unwrap();
            regions_equal(&view, &want, &format!("{mode:?}"));
        }
    }

    #[test]
    fn running_passes_too() {
        // a uniform window, then a maximum with an origin, then a minimum
        let (src, data) = source();
        for mode in MODES {
            let passes = vec![
                Pass {
                    axis: 0,
                    line: FilterLine::Uniform { size: 3, origin: 0 },
                    mode,
                },
                Pass {
                    axis: 1,
                    line: FilterLine::MinMax {
                        size: 4,
                        origin: -1,
                        minimum: false,
                    },
                    mode,
                },
                Pass {
                    axis: 2,
                    line: FilterLine::MinMax {
                        size: 5,
                        origin: 0,
                        minimum: true,
                    },
                    mode,
                },
            ];
            let mut want = vec![0f64; data.len()];
            fl::separable(&data, &SHAPE, &passes, &mut want).unwrap();
            let view = Separable::new(src.clone(), passes).unwrap();
            regions_equal(&view, &want, &format!("{mode:?}"));
        }
    }

    #[test]
    fn combined_chains_too() {
        // a laplace (sum of one pass per axis) and a gradient magnitude (hypot of
        // Gaussian chains with one derivative each)
        let (src, data) = source();
        for mode in MODES {
            let lap: Vec<Vec<Pass>> = (0..3)
                .map(|a| vec![Pass::correlate(a, vec![1.0, -2.0, 1.0], mode)])
                .collect();
            let grad: Vec<Vec<Pass>> = (0..3)
                .map(|d| {
                    (0..3)
                        .map(|a| {
                            let order = usize::from(a == d);
                            Pass::correlate(a, fl::gaussian_weights(1.0, order, 4), mode)
                        })
                        .collect()
                })
                .collect();
            for (chains, combine) in [(lap, Combine::Sum), (grad, Combine::Hypot)] {
                let mut want = vec![0f64; data.len()];
                fl::terms(&data, &SHAPE, &chains, combine, &mut want).unwrap();
                let view = Separable::terms(src.clone(), chains, Some(combine)).unwrap();
                regions_equal(&view, &want, &format!("{combine:?} {mode:?}"));
            }
        }
    }

    #[test]
    fn a_split_read_gives_the_same_values() {
        let shape = [20usize, 30];
        let data: Vec<f64> = (0..600).map(|v| ((v * 31) % 17) as f64).collect();
        let src: SourceRef = Arc::new(Mem {
            shape: vec![20, 30],
            data: data.clone(),
        });
        let passes = vec![
            Pass::correlate(0, fl::gaussian_weights(1.5, 0, 6), FilterMode::Mirror),
            Pass::correlate(1, fl::gaussian_weights(1.5, 1, 6), FilterMode::Mirror),
        ];
        let mut want = vec![0f64; data.len()];
        fl::separable(&data, &shape, &passes, &mut want).unwrap();
        let view = Separable::new(src, passes).unwrap().with_max_read(200);
        let got = read(&view, &[0, 0], &[20, 30], 2).unwrap();
        // SAFETY: the read gave f64s
        let got: &[f64] =
            unsafe { std::slice::from_raw_parts(got.as_ptr() as *const f64, got.len() / 8) };
        assert_eq!(got, &want[..]);
    }

    #[test]
    fn wrap_reads_the_far_end() {
        let src: SourceRef = Arc::new(Mem {
            shape: vec![100],
            data: vec![0.0; 100],
        });
        let view = Separable::new(
            src,
            vec![Pass::correlate(0, vec![1.0; 5], FilterMode::Wrap)],
        )
        .unwrap();
        assert_eq!(view.plan(&[0], &[3]).1, (vec![0], vec![100]));
        assert_eq!(view.plan(&[40], &[43]).1, (vec![38], vec![45]));
    }

    #[test]
    fn gather_takes_the_inner_box() {
        let data: Vec<u32> = (0..60).collect();
        let have = (vec![10, 20, 30], vec![13, 24, 35]);
        let want = (vec![11, 21, 32], vec![13, 23, 34]);
        // (i, j, k) of the 3 x 4 x 5 block is i*20 + j*5 + k
        let want = Stage {
            at: want,
            dec: vec![false; 3],
        };
        assert_eq!(
            gather(&data, &have, &want, &[1, 1, 1]),
            vec![27, 28, 32, 33, 47, 48, 52, 53]
        );
        // every 2nd on the last axis: positions 30, 32, 34 of 30 .. 35
        let every = Stage {
            at: (vec![10, 20, 30], vec![11, 21, 35]),
            dec: vec![false, false, true],
        };
        assert_eq!(gather(&data, &have, &every, &[1, 1, 2]), vec![0, 2, 4]);
    }

    #[test]
    fn a_stepped_filter_is_the_filter_then_the_stride() {
        // every chain kind, every mode, steps on filtered and unfiltered axes, any region:
        // the same values as reading the filter whole and keeping every step-th
        let (src, _) = source();
        let w = vec![0.0625, 0.25, 0.375, 0.25, 0.0625];
        let chains: Vec<Vec<Pass>> = vec![
            vec![
                Pass::correlate(1, w.clone(), FilterMode::Reflect),
                Pass::correlate(2, w.clone(), FilterMode::Reflect),
            ],
            vec![Pass::correlate(2, vec![0.1, 0.5, 0.2, 0.15], FilterMode::Mirror)],
            vec![
                Pass { axis: 1, line: FilterLine::Uniform { size: 3, origin: 0 }, mode: FilterMode::Nearest },
                Pass::correlate(2, w.clone(), FilterMode::Constant(1.5)),
            ],
            vec![Pass::correlate(0, w.clone(), FilterMode::Wrap)],
        ];
        for passes in chains {
            let f = Separable::new(src.clone(), passes.clone()).unwrap();
            let all = read(&f, &[0, 0, 0], &[7, 9, 11], 1).unwrap();
            let all: Vec<f64> = all
                .chunks_exact(8)
                .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            for (phase, step) in [([0, 0, 0], [1, 2, 2]), ([1, 1, 0], [2, 3, 2]), ([0, 2, 1], [3, 1, 4])] {
                let s = f.with_steps(&phase, &step).unwrap();
                let sh = s.shape().to_vec();
                for (a, b) in [([0, 0, 0], sh.clone()), ([1, 1, 1], sh.clone())] {
                    if (0..3).any(|i| a[i] >= b[i]) {
                        continue;
                    }
                    for threads in [1, 3] {
                        let got = read(&s, &a, &b, threads).unwrap();
                        let got: Vec<f64> = got
                            .chunks_exact(8)
                            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                            .collect();
                        let mut k = 0;
                        for i in a[0]..b[0] {
                            for j in a[1]..b[1] {
                                for l in a[2]..b[2] {
                                    let (x, y, z) = (
                                        (phase[0] + i * step[0]) as usize,
                                        (phase[1] + j * step[1]) as usize,
                                        (phase[2] + l * step[2]) as usize,
                                    );
                                    let want = all[(x * 9 + y) * 11 + z];
                                    assert_eq!(got[k].to_bits(), want.to_bits(), "{passes:?} {phase:?} {step:?}");
                                    k += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
