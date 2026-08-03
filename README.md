# dyna_zarr

A lightweight, **dask-free** Python library for lazy, **memory-bounded** operations on large Zarr (and TIFF) arrays — with an optional GPU path.

## Overview

**dyna_zarr** is a thin, pull-based array layer over [Zarr](https://zarr-python.readthedocs.io/). Instead of building a task graph, every operation is a lazy *transform* whose `read(key)` maps an output slice back to a bounded input read, ending at a direct zarr/TensorStore read. Slicing a result pulls **only** that region through the whole operation chain — no intermediates are materialized.

The practical consequence is **memory-boundedness**: the streaming `io.write` path processes a large array region by region, so peak RAM is a function of the region/worker budget, not of the array size. This makes it possible to read, transform, and write arrays far larger than memory.

Highlights:

- **Pull-based & lazy** — operations defer until `.compute()` (materialize) or `io.write` (stream to disk).
- **Memory-bounded streaming** — region-wise `io.write` with per-worker memory + worker-count knobs; even reshape/flatten/rechunk of incompatibly-chunked data stay bounded.
- **NumPy-like** — operator overloads, array methods (`.astype/.clip/.round`), and the NumPy ufunc protocol (`np.sqrt(a)`, `np.add(a, 2)`) all work on a `DynamicArray`.
- **Rich op set** — pointwise ufuncs, streaming reductions, neighborhood (halo) filters, structural reshaping, differences, and array creation — ~85 operations.
- **Multi-format I/O** — read TIFF, Zarr v2, Zarr v3 (local, S3/GCS, HTTP); write Zarr v2/v3 with optional sharding.
- **Optional GPU** — run an op chain on CUDA via CuPy, with a single host↔device transfer per region.

## Installation

```bash
pip install git+https://github.com/bugraoezdemir/dyna_zarr.git
```

Optional GPU support (pick the extra matching your CUDA toolkit from `nvidia-smi`):

```bash
pip install "dyna_zarr[gpu-cu12]"   # CUDA 12.x  (`[gpu]` is an alias for this)
pip install "dyna_zarr[gpu-cu11]"   # CUDA 11.x
pip install "dyna_zarr[gpu-cu13]"   # CUDA 13.x (e.g. Blackwell)
```

## Quick start

### Read

```python
from dyna_zarr import io

arr = io.read("image.tiff")      # TIFF via tifffile's zarr bridge
arr = io.read("array_v2.zarr")   # Zarr v2
arr = io.read("array_v3.zarr")   # Zarr v3  (also s3://, gs://, http://)

print(arr.shape, arr.dtype, arr.chunks)

data   = arr.compute()                    # materialize the whole array
region = arr[10:20, 50:150, 100:200].compute()   # pull just this region
```

### Write (memory-bounded streaming)

```python
from dyna_zarr import io, Codecs

io.write(arr, "out_v3.zarr", zarr_format=3)
io.write(arr, "out.zarr", chunks=(64, 64, 64), zarr_format=3)
io.write(arr, "out.zarr", dtype="float32", zarr_format=3)       # cast on write
io.write(arr, "out.zarr", compressor=Codecs(compressor="zstd", clevel=5), zarr_format=3)

# memory / parallelism controls (peak RAM ~ region_size_mb * max_workers)
io.write(arr, "out.zarr", region_size_mb=64, max_workers=4)
```

### Lazy operation chains

```python
from dyna_zarr import io, operations as ops

arr = io.read("input.zarr")

result = ops.sqrt(ops.clip(ops.abs(arr), 0, 1))   # nothing computed yet
io.write(result, "output.zarr", zarr_format=3)     # streamed, region by region
# ...or result.compute() to materialize
```

### NumPy-like interface

A `DynamicArray` behaves like a NumPy/dask array — operators, methods, and ufuncs are all lazy:

```python
import numpy as np

masked = (arr > 3) & (arr < 100)     # elementwise operators -> lazy mask
scaled = (arr.astype("float32") / 255).clip(0, 1)
out    = np.sqrt(np.abs(arr))        # NumPy ufunc protocol dispatches to lazy ops
```

## Operations catalog

All are lazy and available flat on `dyna_zarr.operations` (also grouped by category submodule).

- **Pointwise / ufuncs** — `abs`, `negative`, `sign`, `sqrt`, `square`, `exp`, `log`, `log2`, `log10`, `floor`, `ceil`, `reciprocal`, `round`, `clip`, `astype`; binary `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `mod`, `power`, `maximum`, `minimum`; comparisons `greater(_equal)`, `less(_equal)`, `equal`, `not_equal`; logical `and/or/xor/not`; `where`, `isin`, `digitize`.
- **Reductions (streaming, memory-bounded)** — `min`, `max`, `sum`, `prod`, `mean`, `any`, `all`, `var`, `std`, `argmin`, `argmax`, `median`, `histogram` (`axis=`/`keepdims=`).
- **Neighborhood (halo/overlap)** — `gaussian_filter`, `uniform_filter`, `median_filter`, `minimum_filter`, `maximum_filter`, `grey_erosion`, `grey_dilation`.
- **Structural** — `concatenate`, `stack`, `transpose`, `swap_axes`, `reshape`, `flatten`, `squeeze`, `expand_dims`, `pad`, `tile`, `roll`, `flip`, `rot90`, `slice_array`.
- **Differences** — `diff`, `gradient`.
- **Creation** — `zeros`, `ones`, `full`, `empty`, `random` (+ `*_like`). `random` is position-deterministic, so the result is independent of chunking.
- **Primitives** — `map_blocks` (pointwise), `map_overlap` (neighborhood with a halo), `reduce` (streaming) — build your own ops.

## Memory-bounded reshape / flatten / rechunk

C-order reshape/flatten conflict with n-d chunk layout, so a naive implementation blows up. dyna_zarr stages these through disk (a Rechunker-style two-phase, read-once/write-once copy), so peak RAM stays a function of the per-worker budget rather than the array size. When you `io.write` an outermost `reshape`/`flatten`, this path is used automatically:

```python
from dyna_zarr import io, operations as ops

arr = io.read("big_4d.zarr")                 # e.g. 5 GB, awkward chunks
io.write(ops.flatten(arr), "flat.zarr", region_size_mb=128, max_workers=2)
io.write(ops.reshape(arr, (a, b)), "reshaped.zarr")
```

## GPU (optional)

With a CuPy install, run a chain on the GPU. `device='cuda'` on a terminal (`compute` / `io.write`) makes device-inheriting ops run on the GPU; a single host→device transfer happens at the first CUDA op and the data stays resident up the chain (results are returned/written from the host).

```python
result = ops.gaussian_filter(arr, sigma=3)
out = result.compute(device="cuda")          # whole chain on GPU
io.write(result, "out.zarr", device="cuda")  # per-region GPU compute, streamed write
```

## Core components

- **`io.read(source)`** — read TIFF / Zarr v2 / Zarr v3 (local or remote) → `DynamicArray`.
- **`io.write(array, path, ...)`** — stream a `DynamicArray` to Zarr v2/v3 (chunks, sharding, compression, dtype cast, `region_size_mb`, `max_workers`, `device`).
- **`operations`** — the lazy op set above.
- **`DynamicArray`** — the pull-based lazy array (slicing, `.compute()`, operators, `.astype/.clip/.round`, ufunc protocol).
- **`Codecs`** — compression configuration for Zarr v2 and v3.

## Requirements

- Python ≥ 3.11
- zarr ≥ 3.0.0, numpy ≥ 1.20, scipy ≥ 1.6, tensorstore, tifffile
- Optional: CuPy (via the `gpu-cuXX` extras) for the GPU path
