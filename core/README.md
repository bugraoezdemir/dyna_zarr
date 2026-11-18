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
  the part of an array a resampled region needs.

Resampling and downscaling under these transformations come next.

Its wheels are abi3: one per platform for every Python from 3.11. Building from
source needs a Rust toolchain and a C compiler (zarrs builds the blosc and zstd C
libraries).

MIT licence. The wheels include third-party libraries, whose notices are in
[THIRD_PARTY_LICENSES.txt](https://github.com/bugraoezdemir/dyna_zarr/blob/main/core/THIRD_PARTY_LICENSES.txt).
