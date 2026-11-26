//! Labels of a tiled mask as a lazy view (Workplan WP15, task DR7 step 8): what
//! tilewise-ccl's Phase B computes. The mask is cut into tiles of `tile` elements (the
//! last ones clipped at the array's end); each tile's foreground (`v > 0`) is labeled on
//! its own as scipy.ndimage.label labels it ([`dyna_zarr_filters::label`]), and its
//! local labels are mapped through that tile's table (from tilewise-ccl's Phase A) to
//! the final labels. A region is read as the tiles it overlaps, every tile whole (the
//! local labels depend on the whole tile), in parallel.

use dyna_zarr_filters as fl;

use crate::dtype::DType;
use crate::out::Out;
use crate::pool::pieces;
use crate::source::{c_strides, read_vec, region, Source, SourceRef, SMALL};
use crate::ReadError;

/// A tiled mask's final labels.
pub struct TileLabels {
    src: SourceRef,
    tile: Vec<u64>,
    grid: Vec<u64>,
    connectivity: usize,
    /// every tile's table, one after another (tiles in C order of the tile grid)
    tables: Vec<i64>,
    /// where tile k's table starts in `tables` (and, at k + 1, ends)
    offsets: Vec<usize>,
    out_t: DType,
}

impl TileLabels {
    /// Labels of `src` (any numeric or bool type; foreground where `v > 0`) in tiles of
    /// `tile`, with scipy's structure of `connectivity`; tile k's local label l becomes
    /// `tables[offsets[k] + l]`, written as `out_t` (int32 or int64).
    pub fn new(
        src: SourceRef,
        tile: &[u64],
        connectivity: usize,
        tables: Vec<i64>,
        offsets: Vec<usize>,
        out_t: DType,
    ) -> Result<Self, ReadError> {
        let shape = src.shape();
        let n = shape.len();
        if n == 0 || tile.len() != n || tile.contains(&0) {
            return Err(ReadError::new(format!(
                "tiles of {tile:?} for an array of shape {shape:?}"
            )));
        }
        if !matches!(out_t, DType::Int32 | DType::Int64) {
            return Err(ReadError::new(format!(
                "labels as {}: use int32 or int64",
                out_t.name()
            )));
        }
        if src.dtype().is_complex() {
            return Err(ReadError::new(format!(
                "a mask of {} (complex numbers have no foreground)",
                src.dtype().name()
            )));
        }
        let grid: Vec<u64> = shape
            .iter()
            .zip(tile)
            .map(|(s, t)| s.div_ceil(*t))
            .collect();
        let count = grid.iter().product::<u64>() as usize;
        if offsets.len() != count + 1
            || offsets[0] != 0
            || offsets.windows(2).any(|w| w[1] < w[0] + 1)
            || offsets[count] != tables.len()
        {
            return Err(ReadError::new(format!(
                "tables for {count} tiles: {} offsets, ending at {} of {} entries (each \
                 table holds at least the background's entry)",
                offsets.len(),
                offsets.last().copied().unwrap_or(0),
                tables.len()
            )));
        }
        if out_t == DType::Int32
            && tables
                .iter()
                .any(|&v| v < i32::MIN as i64 || v > i32::MAX as i64)
        {
            return Err(ReadError::new(
                "a label above int32's range for int32 labels".to_string(),
            ));
        }
        Ok(Self {
            src,
            tile: tile.to_vec(),
            grid,
            connectivity,
            tables,
            offsets,
            out_t,
        })
    }

    /// The tiles (grid indices) a region overlaps.
    fn tiles_of(&self, start: &[u64], r: &[u64]) -> Vec<Vec<u64>> {
        let n = r.len();
        if r.contains(&0) {
            return Vec::new();
        }
        let lo: Vec<u64> = (0..n).map(|a| start[a] / self.tile[a]).collect();
        let hi: Vec<u64> = (0..n)
            .map(|a| (start[a] + r[a] - 1) / self.tile[a])
            .collect();
        let mut out = Vec::new();
        let mut t = lo.clone();
        loop {
            out.push(t.clone());
            let mut a = n;
            loop {
                if a == 0 {
                    return out;
                }
                a -= 1;
                t[a] += 1;
                if t[a] <= hi[a] {
                    break;
                }
                t[a] = lo[a];
            }
        }
    }

    /// Tile `t`'s core (start, stop).
    fn core(&self, t: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let shape = self.src.shape();
        let s: Vec<u64> = t.iter().zip(&self.tile).map(|(i, w)| i * w).collect();
        let e: Vec<u64> = (0..t.len())
            .map(|a| (s[a] + self.tile[a]).min(shape[a]))
            .collect();
        (s, e)
    }

    /// Tile `t`'s position in C order of the grid.
    fn flat(&self, t: &[u64]) -> usize {
        t.iter()
            .zip(&self.grid)
            .fold(0u64, |acc, (i, g)| acc * g + i) as usize
    }

    /// The foreground of a tile's core: `v > 0` in the mask's type.
    fn foreground(&self, s: &[u64], e: &[u64], threads: usize) -> Result<Vec<u8>, ReadError> {
        macro_rules! fg {
            ($t:ty, $zero:expr) => {
                read_vec::<$t>(self.src.as_ref(), s, e, threads)?
                    .into_iter()
                    .map(|v| u8::from(v > $zero))
                    .collect()
            };
        }
        Ok(match self.src.dtype() {
            DType::Bool | DType::UInt8 => fg!(u8, 0),
            DType::Int8 => fg!(i8, 0),
            DType::Int16 => fg!(i16, 0),
            DType::Int32 => fg!(i32, 0),
            DType::Int64 => fg!(i64, 0),
            DType::UInt16 => fg!(u16, 0),
            DType::UInt32 => fg!(u32, 0),
            DType::UInt64 => fg!(u64, 0),
            DType::Float16 => read_vec::<u16>(self.src.as_ref(), s, e, threads)?
                .into_iter()
                .map(|b| u8::from(half::f16::from_bits(b).to_f32() > 0.0))
                .collect(),
            DType::Float32 => fg!(f32, 0.0),
            DType::Float64 => fg!(f64, 0.0),
            other => {
                return Err(ReadError::new(format!("a mask of {}", other.name())));
            }
        })
    }

    /// Tile `t` labeled and mapped, the part of it inside the region at `start` written
    /// into `out`.
    fn one(
        &self,
        t: &[u64],
        start: &[u64],
        r: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let n = t.len();
        let (cs, ce) = self.core(t);
        let fg = self.foreground(&cs, &ce, threads)?;
        let cshape: Vec<usize> = (0..n).map(|a| (ce[a] - cs[a]) as usize).collect();
        let (lab, count) = fl::label(&fg, &cshape, self.connectivity).map_err(ReadError::filter)?;
        drop(fg);
        let k = self.flat(t);
        let table = &self.tables[self.offsets[k]..self.offsets[k + 1]];
        if count as usize >= table.len() {
            return Err(ReadError::new(format!(
                "tile {t:?} has {count} components but a table of {} labels: the tables \
                 do not come from this mask (tilewise-ccl's Phase A)",
                table.len() - 1
            )));
        }
        // the part of the core inside the region
        let is: Vec<u64> = (0..n).map(|a| cs[a].max(start[a])).collect();
        let ie: Vec<u64> = (0..n).map(|a| ce[a].min(start[a] + r[a])).collect();
        let ir: Vec<u64> = (0..n).map(|a| ie[a] - is[a]).collect();
        let cstr = c_strides(&cshape.iter().map(|&v| v as u64).collect::<Vec<_>>(), 1);
        // the intersection's first element in the core, and the core's strides in bytes
        let first: usize = (0..n).map(|a| (is[a] - cs[a]) as usize * cstr[a]).sum();
        let at: Vec<u64> = (0..n).map(|a| out.at[a] + is[a] - start[a]).collect();
        let dst = out.moved(at);
        let item = self.out_t.size();
        let strides: Vec<usize> = cstr.iter().map(|s| s * item).collect();
        let mut lab = lab;
        match self.out_t {
            DType::Int32 => {
                // mapped in place: an int32 label has the size of a local one
                for l in lab.iter_mut() {
                    *l = table[*l as usize] as i32 as u32;
                }
                // SAFETY: u32 elements viewed as their bytes
                let bytes =
                    unsafe { std::slice::from_raw_parts(lab.as_ptr() as *const u8, lab.len() * 4) };
                dst.copy_from(&ir, &bytes[first * 4..], &strides, 4, None);
            }
            _ => {
                let v: Vec<i64> = lab.iter().map(|&l| table[l as usize]).collect();
                drop(lab);
                // SAFETY: i64 elements viewed as their bytes
                let bytes =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) };
                dst.copy_from(&ir, &bytes[first * 8..], &strides, 8, None);
            }
        }
        Ok(())
    }

    /// The most one tile holds while it is read and labeled.
    fn tile_cost(&self, t: &[u64], threads: usize) -> Result<u64, ReadError> {
        let n = t.len();
        let (cs, ce) = self.core(t);
        let voxels: u64 = (0..n).map(|a| ce[a] - cs[a]).product();
        let raw = voxels * self.src.dtype().size() as u64;
        let reading = raw + voxels + self.src.cost(&cs, &ce, threads)?;
        let labeling = voxels + fl::label_bytes(voxels as usize);
        // int64 labels are mapped into a buffer of the core beside the local ones
        let mapped = if self.out_t == DType::Int64 {
            voxels * 8
        } else {
            0
        };
        let writing = fl::label_bytes(voxels as usize) + mapped;
        Ok(reading.max(labeling).max(writing) + SMALL)
    }
}

impl Source for TileLabels {
    fn stores(&self, out: &mut Vec<std::sync::Arc<crate::ReadStore>>) {
        self.src.stores(out)
    }
    fn shape(&self) -> &[u64] {
        self.src.shape()
    }
    fn dtype(&self) -> DType {
        self.out_t
    }
    fn grid(&self) -> Option<Vec<u64>> {
        Some(self.tile.clone())
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = region(self.src.shape(), start, stop)?;
        out.check(&r, self.out_t)?;
        let tiles = self.tiles_of(start, &r);
        pieces(tiles.len(), threads, |k, each| {
            self.one(&tiles[k], start, &r, out, each)
        })
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(self.src.shape(), start, stop)?;
        let tiles = self.tiles_of(start, &r);
        let (open, each) = crate::pool::piece_plan(tiles.len(), threads);
        let mut worst = 0u64;
        for t in &tiles {
            worst = worst.max(self.tile_cost(t, each)?);
        }
        Ok(worst * open as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::read;
    use std::sync::Arc;

    struct Mem {
        shape: Vec<u64>,
        data: Vec<u8>,
    }

    impl Source for Mem {
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            DType::UInt8
        }
        fn read_into(
            &self,
            start: &[u64],
            stop: &[u64],
            out: &Out,
            _: usize,
        ) -> Result<(), ReadError> {
            let r = region(&self.shape, start, stop)?;
            let strides = c_strides(&self.shape, 1);
            let off: usize = start
                .iter()
                .zip(&strides)
                .map(|(s, t)| *s as usize * t)
                .sum();
            out.copy_from(&r, &self.data[off..], &strides, 1, None);
            Ok(())
        }
        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(0)
        }
    }

    #[test]
    fn tiles_label_on_their_own_and_map_through_their_tables() {
        // 1 x 6: [1 1 0 | 1 1 1] in tiles of 3: tile 0 has one component, tile 1 one
        let src: SourceRef = Arc::new(Mem {
            shape: vec![6],
            data: vec![1, 1, 0, 1, 1, 1],
        });
        let v =
            TileLabels::new(src, &[3], 1, vec![0, 7, 0, 7], vec![0, 2, 4], DType::Int32).unwrap();
        for (s, e) in [(0u64, 6u64), (1, 5), (4, 6)] {
            let got: Vec<i32> = read(&v, &[s], &[e], 2)
                .unwrap()
                .chunks(4)
                .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            let want = [7, 7, 0, 7, 7, 7];
            assert_eq!(got, want[s as usize..e as usize]);
        }
    }

    #[test]
    fn tables_that_do_not_fit_are_errors() {
        let src: SourceRef = Arc::new(Mem {
            shape: vec![6],
            data: vec![1, 0, 1, 0, 0, 0],
        });
        // tile 0 has two components, its table only one label
        let v = TileLabels::new(
            src.clone(),
            &[3],
            1,
            vec![0, 5, 0],
            vec![0, 2, 3],
            DType::Int32,
        )
        .unwrap();
        assert!(read(&v, &[0], &[6], 1).is_err());
        assert!(
            TileLabels::new(src.clone(), &[3], 1, vec![0, 5], vec![0, 2], DType::Int32).is_err()
        );
        assert!(TileLabels::new(src, &[3], 1, vec![0, 0], vec![0, 1, 2], DType::Float32).is_err());
    }
}
