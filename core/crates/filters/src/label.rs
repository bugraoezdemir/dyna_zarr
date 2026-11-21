//! Connected components as scipy.ndimage.label numbers them (Workplan WP15, task DR7
//! step 8): the structure `generate_binary_structure(ndim, connectivity)` (neighbours
//! whose offset has an L1 norm of at most `connectivity`, which scipy raises to 1 when
//! below), and the components numbered
//! 1, 2, ... in the order their first element appears in a C-order scan, which is the
//! numbering scipy's merge table produces.
//!
//! One scan in C order gives each foreground element the provisional label of a
//! neighbour already scanned (the offsets that come earlier in C order), or a new one,
//! and joins the labels it meets (union-find, the smaller label the root). A second scan
//! numbers the roots by first appearance.

use crate::FilterError;

/// Union-find over provisional labels: `parent[l]`, roots point to themselves.
struct Sets {
    parent: Vec<u32>,
}

impl Sets {
    #[inline]
    fn find(&mut self, mut l: u32) -> u32 {
        while self.parent[l as usize] != l {
            let p = self.parent[l as usize];
            self.parent[l as usize] = self.parent[p as usize]; // path halving
            l = p;
        }
        l
    }

    #[inline]
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let (ra, rb) = (self.find(a), self.find(b));
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi as usize] = lo;
        lo
    }
}

/// The offsets of the structure that come before an element in C order: per offset, its
/// change along the leading axes and along the last axis.
fn backward_offsets(ndim: usize, connectivity: usize) -> Vec<(Vec<i64>, i64)> {
    let mut out = Vec::new();
    let total = 3usize.pow(ndim as u32);
    for code in 0..total {
        let mut d = vec![0i64; ndim];
        let mut c = code;
        for a in (0..ndim).rev() {
            d[a] = (c % 3) as i64 - 1;
            c /= 3;
        }
        let l1: i64 = d.iter().map(|v| v.abs()).sum();
        if l1 == 0 || l1 as usize > connectivity {
            continue;
        }
        // before in C order: the first nonzero change is negative
        if d.iter().find(|&&v| v != 0) == Some(&-1) {
            out.push((d[..ndim - 1].to_vec(), d[ndim - 1]));
        }
    }
    out
}

/// Label the foreground (`fg[i] != 0`) of an array of `shape` (C order) as
/// scipy.ndimage.label does with `generate_binary_structure(ndim, connectivity)`; the
/// labels (0 for background) and their number.
pub fn label(
    fg: &[u8],
    shape: &[usize],
    connectivity: usize,
) -> Result<(Vec<u32>, u32), FilterError> {
    let n = shape.len();
    if n == 0 {
        return Err(FilterError::new("labeling a 0-d array"));
    }
    let total: usize = shape.iter().product();
    if fg.len() != total {
        return Err(FilterError::new(format!(
            "{} elements for a block of shape {shape:?}",
            fg.len()
        )));
    }
    if total >= u32::MAX as usize {
        return Err(FilterError::new(format!(
            "a block of {total} elements: at most {} can be labeled at once",
            u32::MAX - 1
        )));
    }
    let mut lab = vec![0u32; total];
    if total == 0 {
        return Ok((lab, 0));
    }
    // generate_binary_structure raises a connectivity below 1 to 1
    let offsets = backward_offsets(n, connectivity.max(1));
    let last = shape[n - 1];
    let rows = total / last;
    let lead = &shape[..n - 1];
    let mut sets = Sets { parent: vec![0] };
    let mut idx = vec![0usize; n - 1];
    // per offset: the start of the neighbouring row (None outside the array) and the
    // change along the last axis
    let mut nbr: Vec<(Option<usize>, i64)> = Vec::with_capacity(offsets.len());
    for r in 0..rows {
        nbr.clear();
        for (dl, dx) in &offsets {
            let mut row = 0usize;
            let mut inside = true;
            for a in 0..n - 1 {
                let p = idx[a] as i64 + dl[a];
                if p < 0 || p >= lead[a] as i64 {
                    inside = false;
                    break;
                }
                row = row * lead[a] + p as usize;
            }
            nbr.push((inside.then_some(row * last), *dx));
        }
        let base = r * last;
        let row_fg = &fg[base..base + last];
        let mut x = 0usize;
        while x < last {
            // background, eight elements at a time
            while x + 8 <= last && u64::from_ne_bytes(row_fg[x..x + 8].try_into().unwrap()) == 0 {
                x += 8;
            }
            if x >= last {
                break;
            }
            if row_fg[x] == 0 {
                x += 1;
                continue;
            }
            let mut got = 0u32;
            for &(row, dx) in &nbr {
                let Some(row) = row else { continue };
                let nx = x as i64 + dx;
                if nx < 0 || nx >= last as i64 {
                    continue;
                }
                let l = lab[row + nx as usize];
                if l == 0 {
                    continue;
                }
                got = if got == 0 {
                    sets.find(l)
                } else {
                    sets.union(got, l)
                };
            }
            if got == 0 {
                got = sets.parent.len() as u32;
                sets.parent.push(got);
            }
            lab[base + x] = got;
            x += 1;
        }
        for a in (0..n - 1).rev() {
            idx[a] += 1;
            if idx[a] < lead[a] {
                break;
            }
            idx[a] = 0;
        }
    }
    // number the components by their first element in C order
    let mut number = vec![0u32; sets.parent.len()];
    let mut next = 0u32;
    for l in lab.iter_mut() {
        if *l == 0 {
            continue;
        }
        let root = sets.find(*l) as usize;
        if number[root] == 0 {
            next += 1;
            number[root] = next;
        }
        *l = number[root];
    }
    Ok((lab, next))
}

/// Bytes the labeling of `total` elements holds at most beyond its input: the labels,
/// and the union-find and numbering tables (one entry per provisional label, at most one
/// per element, with a vector's doubling).
pub fn label_bytes(total: usize) -> u64 {
    total as u64 * (4 + 8 + 4) + 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_dimensional_numbering_follows_the_first_element() {
        // a U shape joins late: provisional 1 and 2 merge, numbered by first element
        let fg: Vec<u8> = [
            1, 0, 1, 0, 0, //
            1, 0, 1, 0, 1, //
            1, 1, 1, 0, 0, //
            0, 0, 0, 1, 0, //
        ]
        .to_vec();
        let (lab, n) = label(&fg, &[4, 5], 1).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            lab,
            vec![
                1, 0, 1, 0, 0, //
                1, 0, 1, 0, 2, //
                1, 1, 1, 0, 0, //
                0, 0, 0, 3, 0, //
            ]
        );
        // with diagonals the lone element at (3, 3) joins the U
        let (lab, n) = label(&fg, &[4, 5], 2).unwrap();
        assert_eq!(n, 2);
        assert_eq!(lab[3 * 5 + 3], 1);
        assert_eq!(lab[5 + 4], 2);
    }

    #[test]
    fn offsets_are_the_earlier_half_of_the_structure() {
        assert_eq!(backward_offsets(1, 1).len(), 1);
        assert_eq!(backward_offsets(2, 1).len(), 2);
        assert_eq!(backward_offsets(2, 2).len(), 4);
        assert_eq!(backward_offsets(3, 1).len(), 3);
        assert_eq!(backward_offsets(3, 3).len(), 13);
        assert!(backward_offsets(3, 0).is_empty());
    }

    #[test]
    fn errors_and_empty_blocks() {
        assert!(label(&[1, 0], &[3], 1).is_err());
        assert!(label(&[], &[], 1).is_err());
        assert_eq!(label(&[], &[0, 4], 1).unwrap(), (vec![], 0));
        assert_eq!(label(&[0, 0, 0], &[3], 1).unwrap(), (vec![0, 0, 0], 0));
    }
}
