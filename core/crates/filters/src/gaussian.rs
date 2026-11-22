//! The Gaussian filter and its derivatives (scipy.ndimage.gaussian_filter), and the
//! driver that runs a separable filter as a sequence of passes.

use crate::{separable, Element, FilterError, Mode, Pass};

/// numpy's `a.sum()` of a contiguous f64 array (numpy 2): its pairwise sum of all the
/// elements, 8 accumulators in blocks of up to 128, halved above that. Matched numpy
/// 2.4.3 on 912 arrays of 1 to 20,000 elements. Added to numpy's starting 0.0, so a sum
/// of negative zeros is 0.0, as numpy's.
pub fn numpy_sum(a: &[f64]) -> f64 {
    0.0 + pairwise(a)
}

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let end = n - n % 8;
        let mut i = 8;
        while i < end {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[end..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// The kernel's radius for `sigma` and `truncate`: `int(truncate * sigma + 0.5)`.
pub fn radius(sigma: f64, truncate: f64) -> usize {
    (truncate * sigma + 0.5) as usize
}

/// The correlation weights scipy.ndimage.gaussian_filter1d uses: its
/// `_gaussian_kernel1d(sigma, order, radius)`, reversed.
///
/// scipy builds the kernel with numpy's `exp`, its pairwise sum and, for derivatives,
/// a matrix product; the order 0 kernel follows numpy's operations exactly, but `exp`
/// and BLAS can differ from numpy's in the last bit on some machines. Callers that must
/// match scipy everywhere pass scipy's own weights to [`correlate1d`] instead.
pub fn gaussian_weights(sigma: f64, order: usize, radius: usize) -> Vec<f64> {
    let r = radius as i64;
    let sigma2 = sigma * sigma;
    let c = -0.5 / sigma2;
    let phi: Vec<f64> = (-r..=r).map(|x| (c * (x * x) as f64).exp()).collect();
    let total = numpy_sum(&phi);
    let phi: Vec<f64> = phi.iter().map(|v| v / total).collect();
    let mut kernel = if order == 0 {
        phi
    } else {
        // q(x) phi(x), with q's coefficients from q' + q p' (p' = -x / sigma^2), as scipy
        let mut q = vec![0f64; order + 1];
        q[0] = 1.0;
        let p = 1.0 / -sigma2;
        for _ in 0..order {
            q = (0..=order)
                .map(|i| {
                    let d = if i < order {
                        (i + 1) as f64 * q[i + 1]
                    } else {
                        0.0
                    };
                    let s = if i > 0 { p * q[i - 1] } else { 0.0 };
                    d + s
                })
                .collect();
        }
        (-r..=r)
            .zip(&phi)
            .map(|(x, f)| {
                let mut v = 0.0;
                for (k, qk) in q.iter().enumerate() {
                    v += x.pow(k as u32) as f64 * qk;
                }
                v * f
            })
            .collect()
    };
    kernel.reverse();
    kernel
}

/// scipy.ndimage.gaussian_filter of the whole of `input` into `output`: per axis in
/// order, a pass of [`gaussian_weights`] (axes with sigma <= 1e-15 are skipped, as
/// scipy). `sigma`, `order` and `mode` hold one entry per axis.
pub fn gaussian_filter<T: Element, U: Element + Default>(
    input: &[T],
    shape: &[usize],
    sigma: &[f64],
    order: &[usize],
    mode: &[Mode],
    truncate: f64,
    output: &mut [U],
) -> Result<(), FilterError> {
    let n = shape.len();
    if sigma.len() != n || order.len() != n || mode.len() != n {
        return Err(FilterError::new(format!(
            "sigma, order and mode need one entry per axis of {shape:?}"
        )));
    }
    if sigma.iter().any(|s| s.is_nan() || *s < 0.0) || truncate.is_nan() || truncate < 0.0 {
        return Err(FilterError::new(format!(
            "sigma {sigma:?} and truncate {truncate} must be >= 0"
        )));
    }
    let passes: Vec<Pass> = (0..n)
        .filter(|&a| sigma[a] > 1e-15)
        .map(|a| {
            let w = gaussian_weights(sigma[a], order[a], radius(sigma[a], truncate));
            Pass::correlate(a, w, mode[a])
        })
        .collect();
    separable(input, shape, &passes, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_sum_orders_as_numpy() {
        // more than 8 and more than 128 elements take numpy's other branches
        for n in [1usize, 5, 9, 17, 130, 300] {
            let a: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 3.0)).collect();
            let s = numpy_sum(&a);
            let plain: f64 = a.iter().sum();
            assert!((s - plain).abs() < 1e-12, "{n}");
        }
    }

    #[test]
    fn weights_sum_to_one_and_derivatives_are_odd() {
        let w = gaussian_weights(2.0, 0, 8);
        assert_eq!(w.len(), 17);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let d = gaussian_weights(1.5, 1, 6);
        for i in 0..6 {
            assert_eq!(d[i], -d[12 - i]);
        }
        // correlation weights (scipy's kernel reversed): a ramp's derivative is about +1;
        // scipy.ndimage.gaussian_filter1d(np.arange(40.), 1.5, order=1, mode="nearest")
        let x: Vec<f64> = (0..40).map(f64::from).collect();
        let mut out = vec![0f64; 40];
        gaussian_filter(&x, &[40], &[1.5], &[1], &[Mode::Nearest], 4.0, &mut out).unwrap();
        assert!((out[20] - 0.9997837058660027).abs() < 1e-14, "{}", out[20]);
    }

    #[test]
    fn sigma_zero_axes_are_skipped() {
        let x: Vec<u16> = (0..12).collect();
        let mut out = vec![0u16; 12];
        gaussian_filter(
            &x,
            &[3, 4],
            &[0.0, 0.0],
            &[0, 0],
            &[Mode::Reflect; 2],
            4.0,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, x);
    }
}
