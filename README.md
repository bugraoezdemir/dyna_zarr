# dyna-zarr

A lightweight, dask-free Python library for lazy, memory-bounded operations on large Zarr (and TIFF) arrays, with an optional GPU path.

## Overview

dyna-zarr is a thin, pull-based array layer over [Zarr](https://zarr-python.readthedocs.io/). Instead of building a task graph, every operation is a lazy *transform* whose `read(key)` maps an output slice back to a bounded input read, ending at a direct zarr/TensorStore read. Slicing a result pulls only that region through the whole operation chain, so no intermediates are materialized.

The practical consequence is memory-boundedness. When you stream a result to disk with `io.write`, the array is processed region by region, so peak RAM is a function of the region and worker budget rather than the array size. This makes it possible to read, transform, and write arrays far larger than memory.

## Memory-boundedness

There are two ways to run a lazy result, with different memory behavior:

- `io.write(result, path)` streams the result to disk region by region. Peak RAM is roughly `region_size_mb * max_workers`, independent of the array size. This is the memory-bounded path.
- `result.compute()` returns a single in-memory NumPy array. It materializes the whole result by design (mirroring `dask.array.compute`), so it is not memory-bounded. Use it only for results that fit in RAM.

Every operation is memory-bounded on the `io.write` path except `median`, `argmin`, and `argmax`, which are flagged in the operations catalog below.

## Features

- **Pull-based and lazy.** Operations defer until `.compute()` (materialize) or `io.write` (stream to disk).
- **Memory-bounded streaming.** Region-wise `io.write` with per-worker memory and worker-count knobs. Even reshape, flatten, and rechunk of incompatibly-chunked data stay bounded, by staging through disk.
- **NumPy-like.** Operator overloads, array methods (`.astype`, `.clip`, `.round`), and the NumPy ufunc protocol (`np.sqrt(a)`, `np.add(a, 2)`) all work on a `DynamicArray`.
- **Rich op set.** About 90 operations: pointwise ufuncs, streaming reductions, neighborhood (halo) filters, structural reshaping, differences, and array creation.
- **Multi-format I/O.** Read TIFF, Zarr v2, and Zarr v3 (local, S3/GCS, HTTP); write Zarr v2/v3 with optional sharding. An optional Rust-backed storage backend can serve reads and writes in place of the default.
- **Optional GPU.** Run an op chain on CUDA via CuPy, with a single host-to-device transfer per region.

## Installation

```bash
pip install dyna-zarr
```

Optional GPU support (pick the extra matching your CUDA toolkit from `nvidia-smi`):

```bash
pip install "dyna-zarr[gpu-cu12]"   # CUDA 12.x  ([gpu] is an alias for this)
pip install "dyna-zarr[gpu-cu11]"   # CUDA 11.x
pip install "dyna-zarr[gpu-cu13]"   # CUDA 13.x (e.g. Blackwell)
```

Optional faster storage backend (see [zarrista backend](#zarrista-backend-optional)):

```bash
pip install "dyna-zarr[zarrista]"
```

## Quick start

### Read

```python
from dyna_zarr import io

arr = io.read("image.tiff")      # TIFF via tifffile's zarr bridge
arr = io.read("array_v2.zarr")   # Zarr v2
arr = io.read("array_v3.zarr")   # Zarr v3  (also s3://, gs://, http://)

print(arr.shape, arr.dtype, arr.chunks)

data   = arr.compute()                           # materialize the whole array
region = arr[10:20, 50:150, 100:200].compute()   # pull just this region
```

With the optional `[zarrista]` extra, the same call can read through the faster
Rust-backed backend instead. Everything downstream is unchanged:

```python
arr = io.read("array.zarr", backend="zarrista")   # default is "tensorstore"
```

### Write (memory-bounded streaming)

```python
from dyna_zarr import io, Codecs

io.write(arr, "out_v3.zarr", zarr_format=3)
io.write(arr, "out.zarr", chunks=(64, 64, 64), zarr_format=3)
io.write(arr, "out.zarr", dtype="float32", zarr_format=3)       # cast on write
io.write(arr, "out.zarr", compressor=Codecs(compressor="zstd", clevel=5), zarr_format=3)

# memory and parallelism controls (peak RAM is roughly region_size_mb * max_workers)
io.write(arr, "out.zarr", region_size_mb=64, max_workers=4)
```

The optional backend writes through the same call, with the streaming pipeline,
chunking, sharding, and compression options all unchanged:

```python
io.write(arr, "out.zarr", zarr_format=3, backend="zarrista")
```

Read and write backends are independent, so they can differ:

```python
arr = io.read("s3://bucket/image.zarr", backend="zarrista")   # zarrista
io.write(arr, "out.zarr")                                     # tensorstore
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

A `DynamicArray` behaves like a NumPy or dask array. Operators, methods, and ufuncs are all lazy:

```python
import numpy as np

masked = (arr > 3) & (arr < 100)     # elementwise operators build a lazy mask
scaled = (arr.astype("float32") / 255).clip(0, 1)
out    = np.sqrt(np.abs(arr))        # NumPy ufunc protocol dispatches to lazy ops
```

### Neighborhood filters

Neighborhood (halo) filters wrap `scipy.ndimage`. Each read pulls its own halo, so results are chunk-invariant and exact, and stay memory-bounded when streamed.

```python
import numpy as np
from dyna_zarr import io, operations as ops

img = io.read("volume.zarr")   # e.g. (z, y, x)

# LoG filtering
log    = ops.gaussian_laplace(img, sigma=2)
io.write(log, "log.zarr", zarr_format=3)   # halo handled per region

# median denoise
denoised = ops.median_filter(img, size=3)
io.write(denoised, "denoised.zarr")

# a custom per-plane kernel
kernel   = np.ones((1, 3, 3), dtype="float32") / 9   # 3x3 mean within each z-plane
blurred  = ops.convolve(img, kernel)
io.write(blurred, "blurred.zarr")
```

## Operations catalog

Every operation is lazy, and memory-bounded on the `io.write` path except `median`, `argmin`, and `argmax` (see Memory-boundedness). All are available flat on `dyna_zarr.operations`, and also grouped by category submodule.

- **Pointwise / ufuncs.** `abs`, `negative`, `sign`, `sqrt`, `square`, `exp`, `log`, `log2`, `log10`, `floor`, `ceil`, `reciprocal`, `round`, `clip`, `astype`; binary `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `mod`, `power`, `maximum`, `minimum`; comparisons `greater(_equal)`, `less(_equal)`, `equal`, `not_equal`; logical `and`, `or`, `xor`, `not`; `where`, `isin`, `digitize`.
- **Reductions.** Streaming and memory-bounded: `min`, `max`, `sum`, `prod`, `mean`, `any`, `all`, `var`, `std`, `histogram` (with `axis=` and `keepdims=`). Not fully bounded (hold the full reduced axis): `median`, `argmin`, `argmax`.
- **Neighborhood (halo/overlap).** `gaussian_filter`, `uniform_filter`, `median_filter`, `minimum_filter`, `maximum_filter`, `grey_erosion`, `grey_dilation`, `convolve`, `correlate`, `laplace`, `gaussian_laplace`, `gaussian_gradient_magnitude`.
- **Structural.** `concatenate`, `stack`, `transpose`, `swap_axes`, `reshape`, `flatten`, `squeeze`, `expand_dims`, `pad`, `tile`, `roll`, `flip`, `rot90`, `slice_array`.
- **Differences.** `diff`, `gradient`.
- **Scan (prefix, along one axis).** `cumsum`, `cumprod`, `cummax`, `cummin`. Streamed with a bounded carry on the `io.write` path, so memory-bounded despite the sequential dependency.
- **Creation.** `zeros`, `ones`, `full`, `empty`, `random` (and the `*_like` variants). `random` is position-deterministic, so the result is independent of chunking.
- **Primitives.** `map_blocks` (pointwise), `map_overlap` (neighborhood with a halo), `reduce` (streaming). Use these to build your own ops.

## Memory-bounded reshape, flatten, and rechunk

C-order reshape and flatten conflict with n-dimensional chunk layout, so a naive implementation blows up. dyna-zarr stages these through disk (a Rechunker-style two-phase, read-once/write-once copy), so peak RAM stays a function of the per-worker budget rather than the array size. When you `io.write` an outermost `reshape` or `flatten`, this path is used automatically:

```python
from dyna_zarr import io, operations as ops

arr = io.read("big_4d.zarr")                 # e.g. 5 GB, awkward chunks
io.write(ops.flatten(arr), "flat.zarr", region_size_mb=128, max_workers=2)
io.write(ops.reshape(arr, (a, b)), "reshaped.zarr")   # (a, b) is any target shape of the same size
```

## GPU (optional)

With a CuPy install, run a chain on the GPU. Setting `device='cuda'` on a terminal call (`compute` or `io.write`) makes device-inheriting ops run on the GPU. A single host-to-device transfer happens at the first CUDA op and the data stays resident up the chain. Results are returned or written from the host.

```python
result = ops.gaussian_filter(arr, sigma=3)
out = result.compute(device="cuda")          # whole chain on the GPU
io.write(result, "out.zarr", device="cuda")  # per-region GPU compute, streamed write
```

## zarrista backend (optional)

[zarrista](https://github.com/zarrs/zarrista) is a Rust-backed Zarr implementation (built on [zarrs](https://zarrs.dev/)). Installed via the `[zarrista]` extra, it can serve reads and writes in place of the default TensorStore path.

Measured against that default on the same machine, with matched codecs, matched concurrency, and byte-identical output verified in both directions:

| layout | write | read |
|---|---|---|
| Zarr v3, unsharded | 2.4x | 2.0x |
| Zarr v3, sharded | 1.6x | 2.2x |
| Zarr v2 | 1.6x | 1.5x |

Single machine, local filesystem, float32 — treat these as an indication, not a guarantee, and measure on your own data and hardware. The scripts are in `benchmarks/`.

The backend is opt-in rather than the default, for reasons worth knowing before you enable it:

- zarrista 0.1.0 is beta by its own description, and its Python API may change. The extra pins an exact version, and the backend warns if it is used against a different one.
- Zarr v2 works (both directions, verified against zarr-python across dtypes and compressors) but rides on an API that upstream does not document.
- Two upstream behaviors are contained by this backend rather than fixed upstream: concurrent writes to misaligned regions silently lose data, and integer indexing does not drop the indexed axis. The wrapper restores NumPy indexing semantics, and `io.write` grows its regions to whole write units so concurrent writes never share one.

Remote stores (`s3://`, `gs://`, `az://`, `http://`) are read **and** written through zarrista's async API over [obstore](https://github.com/developmentseed/obstore), which the `[zarrista]` extra installs. Credentials come from obstore's usual environment variables; anything else is passed with `storage_options`, which goes straight to `obstore.store.from_url`:

```python
opts = {"endpoint": "https://s3.example.org",       # any non-AWS S3: MinIO, Ceph, ...
        "region": "us-east-1",
        "virtual_hosted_style_request": False,
        "skip_signature": True}                     # public bucket, no credentials

arr = io.read("s3://bucket/image.zarr", backend="zarrista", storage_options=opts)
io.write(arr, "s3://bucket/out.zarr", backend="zarrista", storage_options=opts)
```

Give an S3 endpoint as `s3://bucket/key` plus `endpoint=`, not as an `https://` URL: obstore dispatches on the scheme, so `https://host/bucket/key` builds a plain HTTP store rather than an S3 client.

Arrays written by either backend are ordinary Zarr and are readable by the other, and by zarr-python, so enabling it does not lock data in.

## Relationship to dask

dyna-zarr is not a general replacement for `dask.array`. It targets one job: memory-bounded read, transform, and write of large Zarr/TIFF arrays.

The core idea is to drop the task graph. Because every operation is a pull-based chain, where each output slice maps back to a bounded input read, there is no graph to build and no scheduler to run it. That keeps the engine small, keeps peak RAM bounded by `region_size_mb * max_workers` on the `io.write` path, and avoids scheduling overhead, which makes the read-transform-write pipeline efficient.

The tradeoff is that only operations that fit this slice-pushdown model belong in the chain: pointwise math, neighborhood/halo filters, streaming reductions, and structural reshaping. These are operations that are commonly used in image processing, which is what dyna-zarr is mainly built for. Operations that would need a global, data-dependent graph do not fit directly, and a few that do (such as non-associative reductions) trade extra reads or memory to stay correct.

Two more differences worth knowing:

- **Single machine, for now.** Parallelism today is threaded I/O within one process, plus the optional GPU path. There is no cluster or distributed execution yet; better and process-based parallelism is a possible future direction.
- **Narrower surface.** About 90 operations today, extended where the slice-pushdown model permits. Binary ops also need equal-shaped operands (no general broadcasting between differently shaped lazy arrays yet).

## Core components

- `io.read(source, backend=..., storage_options=...)` reads TIFF, Zarr v2, or Zarr v3 (local or remote) into a `DynamicArray`; `backend` is `"tensorstore"` (default) or the optional `"zarrista"`.
- `io.write(array, path, ...)` streams a `DynamicArray` to Zarr v2/v3 (chunks, sharding, compression, dtype cast, `region_size_mb`, `max_workers`, `device`, `backend`, `storage_options`).
- `operations` is the lazy op set above.
- `DynamicArray` is the pull-based lazy array (slicing, `.compute()`, operators, `.astype`/`.clip`/`.round`, ufunc protocol).
- `Codecs` is the compression configuration for Zarr v2 and v3.
- `backends.zarrista_backend` implements the optional `backend="zarrista"` path used by `io.read` and `io.write`.

## Running the tests

`[dev]` installs the tooling only. Tests for the optional backends are marked, so
they are selected explicitly rather than skipped silently:

```bash
pip install -e ".[dev]"
pytest -m "not zarrista and not gpu"    # core suite

pip install -e ".[dev,zarrista]"
pytest -m zarrista                      # the optional storage backend

pip install -e ".[dev,gpu-cu12]"
pytest -m gpu                           # needs a CUDA device
```

CI runs the core suite on Linux, Windows, and macOS across Python 3.11-3.13, the
zarrista suite on all three OSes as a separate job, and a wheel install-check on each
OS. The GPU tests are **not** run by CI — GitHub's standard runners have no CUDA
device — so run `-m gpu` locally on each platform you support before releasing a
change to the device path.

## Requirements

- Python 3.11 or newer
- zarr 3.0.0+, numpy 1.20+, scipy 1.6+, tensorstore, tifffile
- Optional: CuPy (via the `gpu-cuXX` extras) for the GPU path
- Optional: zarrista + obstore (via the `zarrista` extra) for the faster storage backend
