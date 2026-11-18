//! Homogeneous matrices over `n` axes: (n + 1) x (n + 1), row-major, acting on column
//! vectors (`p' = M p`), as ozp's `spatial.matrix`. Small `n` (an image has at most a
//! handful of axes), so plain loops; no linear-algebra dependency.

use crate::TransformError;

/// A homogeneous (n + 1) x (n + 1) matrix over `n` axes. The last row is `[0 ... 0 1]`
/// for every matrix the crate makes.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    n: usize,
    a: Vec<f64>,
}

impl Matrix {
    /// The identity over `n` axes.
    pub fn identity(n: usize) -> Self {
        let mut a = vec![0.0; (n + 1) * (n + 1)];
        for i in 0..=n {
            a[i * (n + 1) + i] = 1.0;
        }
        Self { n, a }
    }

    /// A matrix from its first `n` rows of `n + 1` values (the last row is implied).
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self, TransformError> {
        let n = rows.len();
        let mut m = Self::identity(n);
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n + 1 {
                return Err(TransformError::new(format!(
                    "a matrix over {n} axes needs rows of {} values, row {i} has {}",
                    n + 1,
                    row.len()
                )));
            }
            for (j, v) in row.iter().enumerate() {
                m.set(i, j, *v);
            }
        }
        Ok(m)
    }

    /// The number of axes.
    pub fn ndim(&self) -> usize {
        self.n
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.a[i * (self.n + 1) + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.a[i * (self.n + 1) + j] = v;
    }

    /// The first `n` rows (the last, `[0 ... 0 1]`, is implied): NGFF's affine form.
    pub fn rows(&self) -> Vec<Vec<f64>> {
        (0..self.n)
            .map(|i| (0..=self.n).map(|j| self.get(i, j)).collect())
            .collect()
    }

    /// The translation column.
    pub fn offset(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.get(i, self.n)).collect()
    }

    /// `self @ other`: apply `other` first, then `self`.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, TransformError> {
        if self.n != other.n {
            return Err(TransformError::new(format!(
                "cannot compose matrices over {} and {} axes",
                self.n, other.n
            )));
        }
        let k = self.n + 1;
        let mut out = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for t in 0..k {
                    s += self.a[i * k + t] * other.a[t * k + j];
                }
                out[i * k + j] = s;
            }
        }
        Ok(Matrix { n: self.n, a: out })
    }

    /// The image of the point `p` (n coordinates).
    pub fn apply(&self, p: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let mut s = self.get(i, self.n);
                for (j, x) in p.iter().enumerate().take(self.n) {
                    s += self.get(i, j) * x;
                }
                s
            })
            .collect()
    }

    /// True when the linear part (the first n x n block) is diagonal, exactly.
    pub fn is_diagonal(&self) -> bool {
        (0..self.n).all(|i| (0..self.n).all(|j| i == j || self.get(i, j) == 0.0))
    }

    /// The diagonal of the linear part.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.get(i, i)).collect()
    }

    /// The determinant of the linear part (LU with partial pivoting).
    pub fn det(&self) -> f64 {
        let n = self.n;
        let mut m: Vec<f64> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| self.get(i, j))
            .collect();
        let mut det = 1.0;
        for c in 0..n {
            let p = (c..n)
                .max_by(|&a, &b| m[a * n + c].abs().total_cmp(&m[b * n + c].abs()))
                .unwrap_or(c);
            if m[p * n + c] == 0.0 {
                return 0.0;
            }
            if p != c {
                for j in 0..n {
                    m.swap(p * n + j, c * n + j);
                }
                det = -det;
            }
            let pivot = m[c * n + c];
            det *= pivot;
            for r in c + 1..n {
                let f = m[r * n + c] / pivot;
                for j in c..n {
                    m[r * n + j] -= f * m[c * n + j];
                }
            }
        }
        det
    }

    /// The inverse (Gauss-Jordan with partial pivoting on the whole homogeneous matrix,
    /// as NumPy's inv; a diagonal linear part is inverted exactly: 1 / s and -t / s).
    /// Singular when |det| of the linear part is below 1e-300, as ozp.
    pub fn inverse(&self) -> Result<Matrix, TransformError> {
        let n = self.n;
        let det = self.det();
        if det.abs() < 1e-300 {
            return Err(TransformError::new(
                "singular, it has no inverse".to_string(),
            ));
        }
        if self.is_diagonal() {
            let mut out = Matrix::identity(n);
            for i in 0..n {
                let s = self.get(i, i);
                out.set(i, i, 1.0 / s);
                out.set(i, n, -self.get(i, n) / s);
            }
            return Ok(out);
        }
        let k = n + 1;
        let mut m = self.a.clone();
        let mut inv = Matrix::identity(n).a;
        for c in 0..k {
            let p = (c..k)
                .max_by(|&a, &b| m[a * k + c].abs().total_cmp(&m[b * k + c].abs()))
                .unwrap_or(c);
            if m[p * k + c] == 0.0 {
                return Err(TransformError::new(
                    "singular, it has no inverse".to_string(),
                ));
            }
            if p != c {
                for j in 0..k {
                    m.swap(p * k + j, c * k + j);
                    inv.swap(p * k + j, c * k + j);
                }
            }
            let pivot = m[c * k + c];
            for j in 0..k {
                m[c * k + j] /= pivot;
                inv[c * k + j] /= pivot;
            }
            for r in 0..k {
                if r != c {
                    let f = m[r * k + c];
                    if f != 0.0 {
                        for j in 0..k {
                            m[r * k + j] -= f * m[c * k + j];
                            inv[r * k + j] -= f * inv[c * k + j];
                        }
                    }
                }
            }
        }
        Ok(Matrix { n, a: inv })
    }

    /// The matrix over `full` axes that acts as `self` on the `sub` axes (named, in
    /// `self`'s order) and as the identity on the others: a z, y, x transformation on a
    /// t, c, z, y, x array.
    pub fn embed(&self, sub: &[&str], full: &[&str]) -> Result<Matrix, TransformError> {
        if sub.len() != self.n {
            return Err(TransformError::new(format!(
                "{} axis names for a transformation over {} axes",
                sub.len(),
                self.n
            )));
        }
        let mut at = Vec::with_capacity(sub.len());
        for name in sub {
            let i = full.iter().position(|f| f == name).ok_or_else(|| {
                TransformError::new(format!("axis {name:?} is not one of {full:?}"))
            })?;
            if at.contains(&i) {
                return Err(TransformError::new(format!("axis {name:?} named twice")));
            }
            at.push(i);
        }
        let m = full.len();
        let mut out = Matrix::identity(m);
        for (i, &fi) in at.iter().enumerate() {
            for (j, &fj) in at.iter().enumerate() {
                out.set(fi, fj, self.get(i, j));
            }
            out.set(fi, m, self.get(i, self.n));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(m: &Matrix) -> Vec<Vec<f64>> {
        m.rows()
    }

    #[test]
    fn matmul_applies_the_right_operand_first() {
        let scale = Matrix::from_rows(&[vec![2.0, 0.0, 0.0], vec![0.0, 3.0, 0.0]]).unwrap();
        let shift = Matrix::from_rows(&[vec![1.0, 0.0, 10.0], vec![0.0, 1.0, 20.0]]).unwrap();
        // scale first, then shift: p -> 2p + 10
        let m = shift.matmul(&scale).unwrap();
        assert_eq!(m.apply(&[1.0, 1.0]), vec![12.0, 23.0]);
    }

    #[test]
    fn inverse_of_a_rotation_with_offset() {
        let m = Matrix::from_rows(&[vec![0.0, -1.0, 5.0], vec![1.0, 0.0, -2.0]]).unwrap();
        let inv = m.inverse().unwrap();
        let p = m.apply(&[3.0, 4.0]);
        let back = inv.apply(&p);
        assert!((back[0] - 3.0).abs() < 1e-12 && (back[1] - 4.0).abs() < 1e-12);
        let id = m.matmul(&inv).unwrap();
        assert_eq!(rows(&id), rows(&Matrix::identity(2)));
    }

    #[test]
    fn a_diagonal_inverse_is_exact_and_stays_diagonal() {
        let m = Matrix::from_rows(&[vec![0.5, 0.0, 3.0], vec![0.0, 4.0, -1.0]]).unwrap();
        let inv = m.inverse().unwrap();
        assert!(inv.is_diagonal());
        assert_eq!(inv.diagonal(), vec![2.0, 0.25]);
        assert_eq!(inv.offset(), vec![-6.0, 0.25]);
    }

    #[test]
    fn a_singular_matrix_has_no_inverse() {
        let m = Matrix::from_rows(&[vec![1.0, 2.0, 0.0], vec![2.0, 4.0, 0.0]]).unwrap();
        assert!(m.inverse().is_err());
        assert_eq!(m.det(), 0.0);
    }

    #[test]
    fn embed_acts_on_the_named_axes_only() {
        // y, x: scale 2 and 3, offset 1 and 4; embedded into t, c, z, y, x
        let m = Matrix::from_rows(&[vec![2.0, 0.0, 1.0], vec![0.0, 3.0, 4.0]]).unwrap();
        let full = m.embed(&["y", "x"], &["t", "c", "z", "y", "x"]).unwrap();
        assert_eq!(
            full.apply(&[1.0, 2.0, 3.0, 1.0, 1.0]),
            vec![1.0, 2.0, 3.0, 3.0, 7.0]
        );
        assert!(m.embed(&["y", "q"], &["y", "x"]).is_err());
        assert!(m.embed(&["y", "y"], &["y", "x"]).is_err());
    }
}
