# dyna-zarr-core

The native engine of [dyna-zarr](https://github.com/bugraoezdemir/dyna_zarr)
(Python module `dyna_zarr_core`). It is not meant to be used on its own: dyna-zarr
calls it, and keeps its own Python path as the reference.

The engine is written in Rust on [zarrs](https://github.com/zarrs/zarrs). It will
read regions of Zarr arrays in parallel without Python's global lock, and resample
and downscale them under coordinate transformations (scale, translation, rotation,
any affine). This first version only sets up the package: it describes a Zarr array
(`array_info`), which proves the build on every platform.

Its wheels are abi3: one per platform for every Python from 3.11. Building from
source needs a Rust toolchain and a C compiler (zarrs builds the blosc and zstd C
libraries).

MIT licence. The wheels include third-party libraries, whose notices are in
[THIRD_PARTY_LICENSES.txt](https://github.com/bugraoezdemir/dyna_zarr/blob/main/core/THIRD_PARTY_LICENSES.txt).
