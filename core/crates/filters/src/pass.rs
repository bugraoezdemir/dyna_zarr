//! A filter as passes along axes, and filters made of several such chains: what every
//! filter of this crate reduces to.
//!
//! - A [`Pass`] is one 1-D operation along one axis (a correlation, a uniform window, a
//!   sliding minimum or maximum), with its extension mode.
//! - A chain of passes runs as scipy runs a separable filter: each pass reads the one
//!   before in the output type ([`separable`]).
//! - Some filters combine chains each applied to the input ([`terms`]): scipy's
//!   generic_laplace adds them in the output type (laplace, gaussian_laplace);
//!   generic_gradient_magnitude adds their squares and takes the square root
//!   (gaussian_gradient_magnitude).

use crate::correlate::{correlate1d_on, correlate1d_stepped, pass_bytes};
use crate::running::{min_max1d_on, running_pass_bytes, uniform1d_on};
use crate::{Element, FilterError, Mode, Span};

/// The 1-D operation of a pass.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    /// scipy.ndimage.correlate1d with these weights
    Correlate { weights: Vec<f64>, origin: isize },
    /// scipy.ndimage.uniform_filter1d of this size
    Uniform { size: usize, origin: isize },
    /// scipy.ndimage.minimum_filter1d (`minimum`) or maximum_filter1d of this size
    MinMax {
        size: usize,
        origin: isize,
        minimum: bool,
    },
}

impl Line {
    /// The window's length (the number of weights for a correlation).
    pub fn taps(&self) -> usize {
        match self {
            Self::Correlate { weights, .. } => weights.len(),
            Self::Uniform { size, .. } | Self::MinMax { size, .. } => *size,
        }
    }

    pub fn origin(&self) -> isize {
        match self {
            Self::Correlate { origin, .. }
            | Self::Uniform { origin, .. }
            | Self::MinMax { origin, .. } => *origin,
        }
    }
}

/// One pass of a filter: `line` along `axis`, the line extended by `mode`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pass {
    pub axis: usize,
    pub line: Line,
    pub mode: Mode,
}

impl Pass {
    /// A correlation with `weights` (origin 0).
    pub fn correlate(axis: usize, weights: Vec<f64>, mode: Mode) -> Self {
        Self {
            axis,
            line: Line::Correlate { weights, origin: 0 },
            mode,
        }
    }

    /// How far the pass reaches before and after a position (with its origin), so a block
    /// padded by this where the array has data gives the whole array's values.
    pub fn reach(&self) -> (usize, usize) {
        let n = self.line.taps().max(1) as isize;
        let (s1, s2) = (n / 2, n - n / 2 - 1);
        let o = self.line.origin();
        ((s1 + o).max(0) as usize, (s2 - o).max(0) as usize)
    }

    /// Heap bytes of the pass computing `out_len` positions: (shared, per thread).
    pub fn bytes(&self, out_len: usize) -> (u64, u64) {
        match &self.line {
            Line::Correlate { weights, .. } => pass_bytes(out_len, weights.len()),
            Line::Uniform { size, .. } | Line::MinMax { size, .. } => {
                running_pass_bytes(out_len, *size)
            }
        }
    }

    /// Run the pass on a block (as [`crate::correlate1d_on`]).
    pub fn run_on<T: Element, U: Element>(
        &self,
        parallel: bool,
        input: &[T],
        shape: &[usize],
        span: Span,
        output: &mut [U],
    ) -> Result<(), FilterError> {
        let (axis, mode) = (self.axis, self.mode);
        match &self.line {
            Line::Correlate { weights, origin } => correlate1d_on(
                parallel, input, shape, axis, weights, *origin, mode, span, output,
            ),
            Line::Uniform { size, origin } => uniform1d_on(
                parallel, input, shape, axis, *size, *origin, mode, span, output,
            ),
            Line::MinMax {
                size,
                origin,
                minimum,
            } => min_max1d_on(
                parallel, input, shape, axis, *size, *origin, *minimum, mode, span, output,
            ),
        }
    }
}

impl Pass {
    /// [`Pass::run_on`] keeping only every `step`-th output along the pass's axis:
    /// output k is position `span.out_start + k * step` (`span.out_len` outputs), each
    /// with the value the whole pass gives it. A correlation computes only those; a
    /// running window (uniform, minimum, maximum) slides over every position anyway, so it
    /// computes the run from the first to the last and keeps every `step`-th.
    pub fn run_stepped<T: Element, U: Element + Default>(
        &self,
        parallel: bool,
        input: &[T],
        shape: &[usize],
        span: Span,
        step: usize,
        output: &mut [U],
    ) -> Result<(), FilterError> {
        if step <= 1 || span.out_len == 0 {
            return self.run_on(parallel, input, shape, span, output);
        }
        if let Line::Correlate { weights, origin } = &self.line {
            return correlate1d_stepped(
                parallel, input, shape, self.axis, weights, *origin, self.mode, span, step, output,
            );
        }
        let a = self.axis;
        let outer: usize = shape[..a].iter().product();
        let inner: usize = shape[a + 1..].iter().product();
        let run = (span.out_len - 1) * step + 1;
        let mut all = vec![U::default(); outer * run * inner];
        self.run_on(
            parallel,
            input,
            shape,
            Span {
                out_len: run,
                ..span
            },
            &mut all,
        )?;
        if output.len() != outer * span.out_len * inner {
            return Err(FilterError::new(format!(
                "output of {} elements; the pass keeps {}",
                output.len(),
                outer * span.out_len * inner
            )));
        }
        for o in 0..outer {
            for k in 0..span.out_len {
                let from = (o * run + k * step) * inner;
                let to = (o * span.out_len + k) * inner;
                output[to..to + inner].copy_from_slice(&all[from..from + inner]);
            }
        }
        Ok(())
    }
}

/// Run `passes` over the whole of `input` (C order, `shape`) into `output`, as scipy
/// runs a separable filter: the first pass reads `input`, every later pass reads the
/// one before in the output type. No passes: `output` is `input` cast.
pub fn separable<T: Element, U: Element + Default>(
    input: &[T],
    shape: &[usize],
    passes: &[Pass],
    output: &mut [U],
) -> Result<(), FilterError> {
    if output.len() != input.len() {
        return Err(FilterError::new(format!(
            "output of {} elements for an input of {}",
            output.len(),
            input.len()
        )));
    }
    let parallel = rayon::current_thread_index().is_some();
    let span = |p: &Pass| Span::whole(*shape.get(p.axis).unwrap_or(&0));
    let Some((first, rest)) = passes.split_first() else {
        for (o, v) in output.iter_mut().zip(input) {
            *o = U::from_f64(v.to_f64());
        }
        return Ok(());
    };
    first.run_on(parallel, input, shape, span(first), output)?;
    let mut tmp = vec![U::default(); output.len()];
    for p in rest {
        tmp.copy_from_slice(output);
        p.run_on(parallel, &tmp, shape, span(p), output)?;
    }
    Ok(())
}

/// How [`terms`] combines its chains, in the output type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combine {
    /// scipy's generic_laplace: the first chain, then each other added (integers wrap)
    Sum,
    /// scipy's generic_gradient_magnitude: each chain squared, added, then the square root
    /// (floating-point types only)
    Hypot,
}

impl Combine {
    /// `acc` after the first chain's values (in place): Hypot squares it.
    pub fn start<U: Element>(self, acc: &mut [U]) {
        if self == Self::Hypot {
            for a in acc {
                *a = a.mul_wrapping(*a);
            }
        }
    }

    /// Fold a later chain's values `t` into `acc`.
    pub fn add<U: Element>(self, acc: &mut [U], t: &[U]) {
        match self {
            Self::Sum => {
                for (a, v) in acc.iter_mut().zip(t) {
                    *a = a.add_wrapping(*v);
                }
            }
            Self::Hypot => {
                for (a, v) in acc.iter_mut().zip(t) {
                    *a = a.add_wrapping(v.mul_wrapping(*v));
                }
            }
        }
    }

    /// After the last chain: Hypot takes the square root.
    pub fn finish<U: Element>(self, acc: &mut [U]) -> Result<(), FilterError> {
        if self == Self::Hypot {
            for a in acc {
                *a = a.sqrt().ok_or_else(|| {
                    FilterError::new(format!(
                        "a gradient magnitude in {} is not supported (floating-point only)",
                        U::NAME
                    ))
                })?;
            }
        }
        Ok(())
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "sum" => Some(Self::Sum),
            "hypot" => Some(Self::Hypot),
            _ => None,
        }
    }
}

/// A filter made of several chains of passes, each applied to the whole of `input`, and
/// combined in the output type (`combine`), as scipy's generic_laplace and
/// generic_gradient_magnitude: the first chain is written to `output`, every other one
/// to a temporary and folded in. No chains: `output` is `input` cast.
pub fn terms<T: Element, U: Element + Default>(
    input: &[T],
    shape: &[usize],
    chains: &[Vec<Pass>],
    combine: Combine,
    output: &mut [U],
) -> Result<(), FilterError> {
    let Some((first, rest)) = chains.split_first() else {
        return separable(input, shape, &[], output);
    };
    if combine == Combine::Hypot && U::from_f64(0.5).sqrt().is_none() {
        return Combine::Hypot.finish(&mut [U::default()]);
    }
    separable(input, shape, first, output)?;
    combine.start(output);
    let mut tmp = vec![U::default(); output.len()];
    for chain in rest {
        separable(input, shape, chain, &mut tmp)?;
        combine.add(output, &tmp);
    }
    combine.finish(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reach_follows_the_origin() {
        let p = |taps, origin| Pass {
            axis: 0,
            line: Line::Uniform { size: taps, origin },
            mode: Mode::Reflect,
        };
        assert_eq!(p(3, 0).reach(), (1, 1));
        assert_eq!(p(4, 0).reach(), (2, 1));
        assert_eq!(p(4, -1).reach(), (1, 2));
        assert_eq!(p(5, 2).reach(), (4, 0));
    }

    #[test]
    fn laplace_terms_add_in_the_output_type() {
        // uint8 laplace of a spike of 100 in a 3 x 3 x 3 block
        let mut x = vec![0u8; 27];
        x[13] = 100;
        let chains: Vec<Vec<Pass>> = (0..3)
            .map(|a| vec![Pass::correlate(a, vec![1.0, -2.0, 1.0], Mode::Reflect)])
            .collect();
        let mut out = vec![0u8; 27];
        terms(&x, &[3, 3, 3], &chains, Combine::Sum, &mut out).unwrap();
        // each axis gives -200 at the centre (56 as uint8); three of them wrap to 168
        assert_eq!(out[13], 56u8.wrapping_add(56).wrapping_add(56));
        assert_eq!(out[12], 100); // a neighbour along the last axis: +100 from that axis
    }

    #[test]
    fn hypot_is_for_floats() {
        let x = vec![1u16; 8];
        let chains = vec![vec![Pass::correlate(
            0,
            vec![-1.0, 0.0, 1.0],
            Mode::Reflect,
        )]];
        let mut out = vec![0u16; 8];
        let e = terms(&x, &[8], &chains, Combine::Hypot, &mut out).unwrap_err();
        assert!(e.0.contains("floating-point"));
    }
}
