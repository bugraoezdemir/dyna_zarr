# dyna-zarr-core

The native engine of [dyna-zarr](https://github.com/bugraoezdemir/dyna_zarr)
(Python module `dyna_zarr_core`). It is not meant to be used on its own: dyna-zarr
calls it, and keeps its own Python path as the reference.

The engine is written in Rust on [zarrs](https://github.com/zarrs/zarrs). So far it
has:

- lazy arrays (`dyna_zarr_core.Array`): Zarr v2 and v3 arrays in a local folder or a
  finished zip, read region by region in parallel without Python's global lock, straight
  into NumPy arrays, and views over them (slicing, transposing, joining, stacking, type
  conversion);
- coordinate transformations (`dyna_zarr_core.transforms`): NGFF scale, translation,
  affine, rotation, axis reordering and sequences, as ome_zarr_pyramid models them, and
  the part of an array a resampled region needs;
- affine resampling (`Array.resample`): nearest and linear, with the matrix convention
  and edge modes of scipy.ndimage.affine_transform and the same results, read lazily
  region by region and computed on all threads;
- downsampling by integer factors (`Array.downsample`): "stride" and "mean" exactly as
  tensorstore.downsample (ome_zarr_pyramid's levels), and "gaussian" (a Gaussian
  low-pass as scipy.ndimage.gaussian_filter, then every f-th sample);
- memory and threads you set and can rely on: a read's `threads` caps how much it holds
  at once, `Array.cost(start, stop, threads)` says how much that is (checked against the
  measured peak in the tests), and reads that share a `Pool` never use more threads
  than it has. A read with `threads=1` uses one thread.

Its wheels are abi3: one per platform for every Python from 3.11. Building from
source needs a Rust toolchain and a C compiler (zarrs builds the blosc and zstd C
libraries).

MIT licence. The wheels include third-party libraries, whose notices are in
[THIRD_PARTY_LICENSES.txt](https://github.com/bugraoezdemir/dyna_zarr/blob/main/core/THIRD_PARTY_LICENSES.txt).
