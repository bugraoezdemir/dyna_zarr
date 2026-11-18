//! Regions under transformations: what part of an input array an output region needs.
//!
//! Index space follows NGFF 0.6: an integer index is the centre of its pixel (the pixel
//! spans index - 0.5 to index + 0.5), which is also what ozp's level translations assume
//! (a stride level keeps level 0's translation, a mean level is shifted by (F - 1) / 2).

use crate::matrix::Matrix;
use crate::TransformError;

/// How values between pixel centres are read; it sets how far around a mapped point
/// pixels are needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// The nearest pixel centre.
    Nearest,
    /// Linear between the two neighbouring pixel centres on each axis.
    Linear,
}

impl Interpolation {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "nearest" => Some(Self::Nearest),
            "linear" => Some(Self::Linear),
            _ => None,
        }
    }
}

/// An index box: start and stop (exclusive) per axis.
pub type IndexBox = (Vec<u64>, Vec<u64>);

/// Mapped coordinates within this of an integer count as that integer.
const SNAP: f64 = 1e-9;

/// `x`, or the integer it lies within 1e-9 of. Rounding in a matrix product puts a
/// pixel centre at 2.9999999999 instead of 3; snapped, the pixels an interpolation
/// reads are the same whichever way the product rounded. The resampler (DR4) snaps the
/// same way, so it never reads outside the region [`input_region`] gives.
pub fn snap(x: f64) -> f64 {
    let r = x.round();
    if (x - r).abs() <= SNAP {
        r
    } else {
        x
    }
}

impl Interpolation {
    /// The first and last index this interpolation reads around coordinate `x`.
    pub fn support(self, x: f64) -> (f64, f64) {
        match self {
            Interpolation::Nearest => {
                let i = snap(x + 0.5).floor();
                (i, i)
            }
            Interpolation::Linear => {
                let s = snap(x);
                (s.floor(), s.ceil())
            }
        }
    }
}

/// The axis-aligned box `[lo, hi]` (continuous coordinates, per axis) of the image of
/// the box `[lo, hi]` under `m`: exact for any affine (centre mapped, half-extent
/// through the absolute linear part).
pub fn bounding_box(
    m: &Matrix,
    lo: &[f64],
    hi: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), TransformError> {
    let n = m.ndim();
    if lo.len() != n || hi.len() != n {
        return Err(TransformError::new(format!(
            "a box of {} / {} coordinates for a transformation over {n} axes",
            lo.len(),
            hi.len()
        )));
    }
    let centre: Vec<f64> = lo.iter().zip(hi).map(|(a, b)| (a + b) / 2.0).collect();
    let half: Vec<f64> = lo
        .iter()
        .zip(hi)
        .map(|(a, b)| (b - a).abs() / 2.0)
        .collect();
    let c = m.apply(&centre);
    let mut out_lo = Vec::with_capacity(n);
    let mut out_hi = Vec::with_capacity(n);
    for (i, ci) in c.iter().enumerate() {
        let h: f64 = (0..n).map(|j| m.get(i, j).abs() * half[j]).sum();
        out_lo.push(ci - h);
        out_hi.push(ci + h);
    }
    Ok((out_lo, out_hi))
}

/// The index box `(start, stop)` (stop exclusive) of the input array that the output
/// region `start .. stop` needs, where `out_to_in` maps output indices to input
/// indices; the pixels `interpolation` reads around each mapped point included,
/// clamped to `input_shape`. None when the region lies wholly outside the input (or is
/// empty): every output pixel there is fill.
pub fn input_region(
    out_to_in: &Matrix,
    start: &[u64],
    stop: &[u64],
    interpolation: Interpolation,
    input_shape: &[u64],
) -> Result<Option<IndexBox>, TransformError> {
    let n = out_to_in.ndim();
    if start.len() != n || stop.len() != n || input_shape.len() != n {
        return Err(TransformError::new(format!(
            "region of {} / {} values and input shape of {} for a transformation over {n} axes",
            start.len(),
            stop.len(),
            input_shape.len()
        )));
    }
    if start.iter().zip(stop).any(|(a, b)| a >= b) || input_shape.contains(&0) {
        return Ok(None);
    }
    // the output pixel centres: start .. stop - 1
    let lo: Vec<f64> = start.iter().map(|&a| a as f64).collect();
    let hi: Vec<f64> = stop.iter().map(|&b| (b - 1) as f64).collect();
    let (a, b) = bounding_box(out_to_in, &lo, &hi)?;
    let mut in_start = Vec::with_capacity(n);
    let mut in_stop = Vec::with_capacity(n);
    for i in 0..n {
        // an affine takes its extremes over a box at the box's corners, which are pixel
        // centres of the region: so the region is exactly what mapping every pixel reads
        let (first, _) = interpolation.support(a[i]);
        let (_, last) = interpolation.support(b[i]);
        let size = input_shape[i] as f64;
        if last < 0.0 || first > size - 1.0 {
            return Ok(None);
        }
        in_start.push(first.max(0.0) as u64);
        in_stop.push(last.min(size - 1.0) as u64 + 1);
    }
    Ok(Some((in_start, in_stop)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: &[&[f64]]) -> Matrix {
        Matrix::from_rows(&rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>()).unwrap()
    }

    /// Every pixel the interpolation reads, mapping each output pixel centre one by one.
    fn brute(
        mat: &Matrix,
        start: &[u64],
        stop: &[u64],
        interp: Interpolation,
        shape: &[u64],
    ) -> Option<(Vec<u64>, Vec<u64>)> {
        let n = start.len();
        let mut lo = vec![i64::MAX; n];
        let mut hi = vec![i64::MIN; n];
        let mut idx: Vec<u64> = start.to_vec();
        loop {
            let p = mat.apply(&idx.iter().map(|&v| v as f64).collect::<Vec<_>>());
            for i in 0..n {
                let (f, l) = interp.support(p[i]);
                lo[i] = lo[i].min(f as i64);
                hi[i] = hi[i].max(l as i64);
            }
            let mut d = n;
            loop {
                if d == 0 {
                    let s: Vec<u64> = (0..n).map(|i| lo[i].max(0) as u64).collect();
                    let e: Vec<u64> = (0..n)
                        .map(|i| (hi[i].min(shape[i] as i64 - 1) + 1) as u64)
                        .collect();
                    if (0..n).any(|i| hi[i] < 0 || lo[i] > shape[i] as i64 - 1) {
                        return None;
                    }
                    return Some((s, e));
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < stop[d] {
                    break;
                }
                idx[d] = start[d];
            }
        }
    }

    #[test]
    fn a_box_under_rotation_is_its_rotated_corners() {
        // 90 degrees: (y, x) -> (-x, y)
        let r = m(&[&[0.0, -1.0, 0.0], &[1.0, 0.0, 0.0]]);
        let (lo, hi) = bounding_box(&r, &[0.0, 0.0], &[2.0, 4.0]).unwrap();
        assert_eq!((lo, hi), (vec![-4.0, 0.0], vec![0.0, 2.0]));
    }

    #[test]
    fn input_regions_match_mapping_every_pixel() {
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let cases: Vec<Matrix> = vec![
            m(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]),     // identity
            m(&[&[2.0, 0.0, 0.5], &[0.0, 2.0, 0.5]]),     // 2x up, centred
            m(&[&[0.5, 0.0, -0.25], &[0.0, 0.5, -0.25]]), // 2x down
            m(&[&[c, -c, 20.0], &[c, c, 3.5]]),           // 45 degrees
            m(&[&[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]]),     // swapped axes
            m(&[&[1.0, 0.3, -7.2], &[0.0, 1.0, 11.6]]),   // shear
        ];
        let shape = [40u64, 50];
        for mat in &cases {
            for interp in [Interpolation::Nearest, Interpolation::Linear] {
                for (start, stop) in [
                    ([0u64, 0u64], [8u64, 8u64]),
                    ([5, 9], [21, 13]),
                    ([30, 40], [64, 64]),
                ] {
                    let got = input_region(mat, &start, &stop, interp, &shape).unwrap();
                    let want = brute(mat, &start, &stop, interp, &shape);
                    assert_eq!(got, want, "{mat:?} {interp:?} {start:?}");
                }
            }
        }
    }

    #[test]
    fn a_region_outside_the_input_needs_nothing() {
        let shift = m(&[&[1.0, 0.0, 1000.0], &[0.0, 1.0, 0.0]]);
        assert_eq!(
            input_region(&shift, &[0, 0], &[4, 4], Interpolation::Linear, &[10, 10]).unwrap(),
            None
        );
        assert_eq!(
            input_region(&shift, &[3, 0], &[3, 4], Interpolation::Linear, &[10, 10]).unwrap(),
            None
        );
    }

    #[test]
    fn on_pixel_centres_linear_reads_no_more_than_nearest() {
        let id = Matrix::identity(1);
        let near = input_region(&id, &[2], &[5], Interpolation::Nearest, &[10]).unwrap();
        let lin = input_region(&id, &[2], &[5], Interpolation::Linear, &[10]).unwrap();
        assert_eq!(near, Some((vec![2], vec![5])));
        assert_eq!(lin, Some((vec![2], vec![5])));
        // half a pixel off: linear needs both neighbours, nearest rounds half up
        let half = Matrix::from_rows(&[vec![1.0, 0.5]]).unwrap();
        assert_eq!(
            input_region(&half, &[2], &[5], Interpolation::Linear, &[10]).unwrap(),
            Some((vec![2], vec![6]))
        );
        assert_eq!(
            input_region(&half, &[2], &[5], Interpolation::Nearest, &[10]).unwrap(),
            Some((vec![3], vec![6]))
        );
    }

    #[test]
    fn snapping_absorbs_rounding_in_the_matrix() {
        assert_eq!(snap(2.9999999999), 3.0);
        assert_eq!(snap(2.99), 2.99);
        assert_eq!(Interpolation::Linear.support(3.0000000001), (3.0, 3.0));
        assert_eq!(Interpolation::Nearest.support(2.4999999999), (3.0, 3.0));
    }
}
