//! A region cut into blocks: cuts per axis, and every combination of them is a block.
//! Blocks are computed from their number when needed, never listed all at once (a
//! whole-array read can have millions).

/// The blocks of a region, relative to its start.
pub(crate) struct Blocks {
    /// Per axis: (from, to) of each cut, relative to the region's start.
    cuts: Vec<Vec<(u64, u64)>>,
}

/// Blocks of more than this many are costed from a sample ([`Blocks::sample`]).
const COST_ALL_UP_TO: usize = 4096;

impl Blocks {
    /// Cuts every `per[i]` elements from the region's start (`r` elements per axis).
    pub fn regular(r: &[u64], per: &[u64]) -> Self {
        let cuts = (0..r.len())
            .map(|i| {
                let p = per[i].max(1);
                (0..r[i])
                    .step_by(p as usize)
                    .map(|o| (o, (o + p).min(r[i])))
                    .collect()
            })
            .collect();
        Self { cuts }
    }

    /// Cuts at the multiples of `unit` (absolute positions) inside `start .. start + r`.
    pub fn aligned(start: &[u64], r: &[u64], unit: &[u64]) -> Self {
        let cuts = (0..r.len())
            .map(|i| {
                let (a, b) = (start[i], start[i] + r[i]);
                let u = unit[i].max(1);
                let mut at = vec![a];
                let mut c = (a / u + 1) * u;
                while c < b {
                    at.push(c);
                    c += u;
                }
                at.push(b);
                at.windows(2).map(|w| (w[0] - a, w[1] - a)).collect()
            })
            .collect();
        Self { cuts }
    }

    pub fn len(&self) -> usize {
        self.cuts
            .iter()
            .map(|c| c.len())
            .try_fold(1usize, |a, n| a.checked_mul(n))
            .unwrap_or(usize::MAX)
    }

    /// Block `k` (C order over the axes' cuts): per axis (from, to) relative to the
    /// region's start.
    pub fn get(&self, k: usize) -> Vec<(u64, u64)> {
        let n = self.cuts.len();
        let mut out = vec![(0, 0); n];
        let mut rest = k;
        for i in (0..n).rev() {
            let m = self.cuts[i].len();
            out[i] = self.cuts[i][rest % m];
            rest /= m;
        }
        out
    }

    /// The blocks a cost is taken over: all of them when there are few; otherwise every
    /// combination of each axis' first, second and last cut, which covers every block
    /// shape and every position relative to a regular storage grid.
    pub fn sample(&self) -> Vec<Vec<(u64, u64)>> {
        let total = self.len();
        if total <= COST_ALL_UP_TO {
            return (0..total).map(|k| self.get(k)).collect();
        }
        let picks: Vec<Vec<(u64, u64)>> = self
            .cuts
            .iter()
            .map(|c| {
                let mut p = vec![c[0]];
                if c.len() > 1 {
                    p.push(c[1]);
                }
                if c.len() > 2 {
                    p.push(c[c.len() - 1]);
                }
                p
            })
            .collect();
        let mut out: Vec<Vec<(u64, u64)>> = vec![Vec::new()];
        for p in &picks {
            out = out
                .into_iter()
                .flat_map(|b| {
                    p.iter().map(move |c| {
                        let mut b = b.clone();
                        b.push(*c);
                        b
                    })
                })
                .collect();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_and_aligned_cuts() {
        let b = Blocks::regular(&[5, 4], &[2, 4]);
        assert_eq!(b.len(), 3);
        assert_eq!(b.get(0), vec![(0, 2), (0, 4)]);
        assert_eq!(b.get(2), vec![(4, 5), (0, 4)]);
        let a = Blocks::aligned(&[3], &[10], &[4]);
        assert_eq!(a.len(), 4);
        assert_eq!(
            (0..4).map(|k| a.get(k)[0]).collect::<Vec<_>>(),
            vec![(0, 1), (1, 5), (5, 9), (9, 10)]
        );
    }

    #[test]
    fn sample_covers_shapes() {
        let b = Blocks::regular(&[10_000, 10_000], &[3, 7]);
        let s = b.sample();
        assert_eq!(s.len(), 9);
        assert!(s.contains(&vec![(9999, 10_000), (9996, 10_000)]));
    }
}
