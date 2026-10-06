# dyna-zarr

A lightweight, dask-free Python library for lazy, memory-bounded operations on large Zarr (and TIFF) arrays, with an optional GPU path.

## Overview

dyna-zarr is a thin, pull-based array layer over [Zarr](https://zarr-python.readthedocs.io/). Instead of building a task graph, every operation is a lazy *transform* whose `read(key)` maps an output slice back to a bounded input read, ending at a direct zarr/TensorStore read. Slicing a result pulls only that region through the whole operation chain, so no intermediates are materialized.



The practical consequence is memory-boundedness. When you stream a result to disk with `io.write`, the array is processed region by region, so peak RAM is a function of the region and worker budget rather than the array size. This makes it possible to read, transform, and write arrays far larger than memory.

Segmenting a volume that does not fit in RAM, on a laptop:

```python
from dyna_zarr import io, operations as ops

vol = io.read("volume.zarr")                             # nothing read yet

smoothed = ops.gaussian_filter(vol, sigma=(1, 2, 2))     # halo filter, still lazy
mask     = (smoothed > 0.5).astype("uint8")              # lazy threshold

io.write(mask, "mask.zarr", region_size_mb=64, max_workers=4)
```

Only the last line touches the disk. The halo filter and the threshold are fused into
one pull-based chain, and `io.write` streams it region by region, so peak RAM is set
by the region budget and not by the size of the volume: a 2 GB array costs the same
as a 200 MB one. The result is identical to the equivalent eager NumPy/SciPy code,
including at region boundaries.

A threshold computed from the data itself, such as `smoothed.mean() + 3 * smoothed.std()`,
works the same way; see [Reductions and statistics](#reductions-and-statistics).

## Memory-boundedness

There are two ways to run a lazy result, with different memory behavior:

- `io.write(result, path)` streams the result to disk region by region. Peak RAM is set by `region_size_mb`, `max_workers` and the dtypes in the chain, and is **independent of the array size**. This is the memory-bounded path; see [What peak RAM actually is](#what-peak-ram-actually-is).
- `result.compute()` returns a single in-memory NumPy array. It materializes the whole result by design (mirroring `dask.array.compute`).

Every operation is memory-bounded on the `io.write` path except `median`, `argmin`, `argmax`, and `unique`, which are flagged in the operations catalog below.

## Features

- **Pull-based and lazy.** Operations defer until `.compute()` (materialize) or `io.write` (stream to disk).
- **Memory-bounded streaming.** Region-wise `io.write` with per-worker memory and worker-count knobs. Even reshape and flatten stay bounded, by staging through local disk.
- **NumPy-like.** Operator overloads, array methods (`.astype`, `.clip`, `.round`), and the NumPy ufunc protocol (`np.sqrt(a)`, `np.add(a, 2)`) all work on a `DynamicArray`.
- **Rich op set.** About 140 operations: pointwise ufuncs, streaming reductions, neighborhood (halo) filters, structural reshaping, differences, prefix scans, and array creation.
- **Multi-format I/O.** Read TIFF, Zarr v2, and Zarr v3 (local, S3/GCS, HTTP); write Zarr v2/v3 (local, S3/GCS) with optional sharding. Optionally, [zarrista](https://github.com/developmentseed/zarrista) can serve reads and writes in place of the default.
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

Optional faster storage backend, [zarrista](https://github.com/developmentseed/zarrista) (see
[zarrista backend](#zarrista-backend-optional)):

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

With the optional `[zarrista]` extra, a **Zarr** source can be read through
[zarrista](https://github.com/developmentseed/zarrista) instead, a Rust-backed Zarr
implementation built on [zarrs](https://zarrs.dev/). Everything downstream is
unchanged:

```python
arr = io.read("array.zarr", backend="zarrista")   # default is "tensorstore"
```

Note the `backend` parameter is specific for a **Zarr** storage backend. It should not be supplied with any value when the input is a TIFF. Supplying a backend with TIFF input will result in an error.

### Write (memory-bounded streaming)

```python
from dyna_zarr import io, Codecs

io.write(arr, "out_v3.zarr", zarr_format=3)
io.write(arr, "out_chunked.zarr", chunks=(64, 64, 64), zarr_format=3)
io.write(arr, "out_float32.zarr", dtype="float32", zarr_format=3)     # cast on write
io.write(arr, "out_zstd.zarr", compressor=Codecs(compressor="zstd", clevel=5), zarr_format=3)

# memory and parallelism controls (see "What peak RAM actually is" below)
io.write(arr, "out_bounded.zarr", region_size_mb=64, max_workers=4)
```

`io.write` does not replace an existing array: writing to a path that already holds a
Zarr array or group raises `OutputExistsError`. Choose a new path, or pass
`overwrite=True` to delete what is there and write it anew. This is the same for both
backends, locally and on S3. Only a path that already looks like a Zarr store is ever
removed, so a mistyped path cannot delete an unrelated directory: if it holds other
files, dyna refuses to write there at all.

Unless told otherwise, the output also keeps the input's **Zarr format, compressor and
sharding**, so reading an array and writing it back gives the same kind of store. This
holds for a TensorStore handle on a stored Zarr array too (`DynamicArray(ts_array)`).
A source that has no Zarr format of its own (a NumPy array, a TIFF, a generated array
such as `ops.zeros`, a virtual TensorStore view such as `ts.downsample`) is written as
Zarr v3 with Blosc/LZ4. Passing `zarr_format=`, `compressor=` or `shard_coefficients=`
overrides each of them.

#### Output chunks

By default, the output **keeps the input's chunks**. A read-modify-write round trip
preserves the chunk grid, and the output layout is what you already know about the
input. Inheriting the grid also inherits its problems, so dyna prints a note when the
inherited chunks are very large (over 8 MiB, which sets the memory floor of the
write, since a region always holds at least one whole chunk) or very small (under
64 KiB, which means a great many small files).

There are two ways to choose a different chunk, and passing both is an error:

- **`chunks=`** is an exact shape, used as given.
- **`chunk_size_mb=`** is a size budget. The chunk is then solved for, together with
  the read region, from sizes that divide or are multiples of the input chunks, so
  reads stay aligned to the source grid. Because only those sizes are allowed, the
  result is approximate, and dyna prints a note when it misses the budget.

What `chunk_size_mb=1` gives for a few sources:

| array | dtype | input chunks | default (kept) | `chunk_size_mb=1` |
|---|---|---|---|---|
| `(300, 1024, 1024)` | float32 | `(1, 1024, 1024)` | `(1, 1024, 1024)` | `(1, 512, 512)` |
| `(300, 1024, 1024)` | uint8 | `(1, 1024, 1024)` | `(1, 1024, 1024)` | `(4, 512, 512)` |
| `(300, 1024, 1024)` | float32 | `(64, 256, 256)` | `(64, 256, 256)` | `(4, 256, 256)` |
| `(2048, 2048)` | float32 | `(2048, 2048)` | `(2048, 2048)` | `(512, 512)` |

When the source has no chunk grid to align to (a reshaped or padded array, for
example), the budget alone decides. The chunk gets **power-of-two sides**, as close to
a cube over the **last three axes** as the budget allows (both axes for a 2D array),
with every leading axis at 1, on the assumption that the trailing axes are spatial and
the leading ones are t/c. A cube rather than a slab matters for what comes next: a
`(1, 256, 1024)` chunk of the same size would force a full XY plane to be read for
every step along Z. With no `chunk_size_mb=` in that case, the budget is 1 MiB:
`(64, 64, 64)` for float32, `(64, 128, 128)` for uint8, `(32, 64, 64)` for float64.
Generated arrays (`ops.zeros`, `ops.random`, ...) get the same grid unless you pass
`chunks=` to them.

```python
io.write(arr, "out_1mb.zarr", chunk_size_mb=1)          # budget, aligned to the source
io.write(arr, "out_exact.zarr", chunks=(32, 128, 128))  # exact
```

Zarr v2 is written the same way, with `zarr_format=2`:

```python
io.write(arr, "out_v2.zarr", zarr_format=2, chunks=(8, 64, 64))
io.write(arr, "out_v2_zstd.zarr", zarr_format=2,
         compressor=Codecs(compressor="blosc", cname="zstd", clevel=5))
```

Sharding is v3 only. `shard_coefficients` is given in units of `chunks`, not in
voxels, so the shard shape is the product of the two:

```python
# chunks (8, 64, 64) x coefficients (2, 2, 2) -> shards of (16, 128, 128)
io.write(arr, "out_sharded.zarr", zarr_format=3,
         chunks=(8, 64, 64), shard_coefficients=(2, 2, 2))
```

Sharding keeps the chunk as the unit of compression and random access while storing
many chunks per file, which suits object stores and filesystems that dislike very
large numbers of small files. Supported only with `zarr_format=3`.

A v3 array can also carry a name per axis. dyna writes them as given and attaches no
meaning to them (formats built on Zarr do, e.g. OME-Zarr 0.5 requires its axis names
here). Zarr v2 has no such field, so they are refused there:

```python
io.write(arr, "out_named.zarr", zarr_format=3, dimension_names=("z", "y", "x"))
```

#### Regions vs chunks

An important concept in dyna-zarr is the concept of regions. Regions are subsets of an array, similar to chunks. The two, however, are completely different:

- **`chunks`** is the *storage* grid: what lands on disk.
- **`region_shape`** (or `region_size_mb`) is the *read* grid: how much is pulled through the operation chain at a time, and therefore what bounds peak RAM. Typically a region contains multiple chunks. **Regions are purely a performance parameter and different region sizes supplied to the write function will lead to byte-identical output.**

By default, `region_size_mb` controls the region size (8 MiB if not given). The region shape is then chosen automatically from that budget, the output chunks, the input chunks and the widest dtype in the chain. The other parameter, `region_shape`, sets the dimensions of the region directly. The two are alternatives: passing both is an error.

When regions must be aligned to another particular grid, `region_shape` can be useful. **It must be a per-axis multiple of `chunks`, and is clamped to the array shape.**


```python
# e.g. a tile-wise producer whose processing grid is (8, 256, 256)
io.write(result, "out_tiled.zarr", chunks=(1, 64, 64), region_shape=(8, 256, 256))
```

The `zarrista` backend operates through the same function call, with the streaming pipeline, chunking, sharding, and compression options:

```python
io.write(arr, "out_zarrista.zarr", zarr_format=3, backend="zarrista")
```

An important point is that, with `backend="zarrista"` writing a **sharded output**, the region is additionally grown to whole output shards, since the shard is the contention unit there. This applies to a `region_shape` you passed as well, and dyna prints a note when it happens.


Read and write backends are independent, so they can differ:

```python
arr = io.read("s3://bucket/image.zarr", backend="zarrista")   # zarrista
io.write(arr, "out_from_s3.zarr")                             # tensorstore
```

In our hands, `backend="zarrista"` has been faster on both the read and the write paths (see [zarrista backend](#zarrista-backend-optional) below for the measurements and the caveats).

#### What peak RAM actually is

The guarantee is that peak RAM does **not** grow with the array. Measured on one
machine with a fixed 64 MB region and a single worker, while the array grew 32x:

| array | peak RAM |
|---|---|
| 64 MB | 86 MB |
| 256 MB | 233 MB |
| 1 GB | 297 MB |
| 2 GB | 278 MB |

It flattens out: past the first few regions, doubling the array changes nothing.

What peak RAM *is* set by, in decreasing order of size:

- **A fixed cost of roughly 200 MB per write**, plus about 150 MB for importing the
  stack. This dominates when regions are small and no setting reduces it.
- **Roughly the input plus the output region**, since a region in flight is held as
  source data and as the result at the same time, with copies on top. Budget for
  about 2.5x `region_size_mb` per live region rather than 1x.
- **`max_workers`**, which multiplies the second term but not the first.

Two consequences worth knowing. A chain that **narrows** the dtype (a `uint8` mask
from a `float32` volume) is the cheapest case, because `region_size_mb` is measured
in the widest dtype the chain carries; a chain that **widens** is the most expensive.
And at small region sizes the fixed cost dominates, so lowering `region_size_mb`
below about 16 MB buys much less than it appears to.

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

### Reductions and statistics

Reductions are lazy, like every other operation. `vol.mean()` returns a `DynamicArray`
and nothing is read until it is needed. A one-element result converts on demand with
`float()`, `int()` or `bool()`.

```python
vol      = io.read("volume.zarr")
smoothed = ops.gaussian_filter(vol, sigma=(1, 2, 2))
level    = smoothed.mean() + 3 * smoothed.std()          # lazy, 0-d
mask     = (smoothed > level).astype("uint8")            # nd compared with 0-d

io.write(mask, "mask_3sd.zarr", region_size_mb=64, max_workers=4)
print(float(level))                                      # already computed, no extra read
```

The threshold depends on every voxel, so the volume has to be read twice: once for the
statistics and once for the write. That is what dyna does. Before the first region is
written, `io.write` computes the small reductions in the chain, in one pass for those
that read the same input (`mean` and `std` above share it). Results of up to 1 MiB,
such as a 0-d statistic or per-channel values, are then kept in memory, so every region
reuses them and a later `float(level)` costs nothing.

These results are kept with the array they were computed from, so a later
`smoothed.max()` does not read anything either. When dyna computes any of `min`, `max`,
`sum` or `mean`, it computes the other three in the same pass, since they cost very
little next to reading the data. `var` and `std` need a float64 sum of squares, which
costs roughly as much as the read itself, so they are only computed when you ask for
them. If you do, the rest comes along. This works for reductions along any axis, as
long as the whole set still fits in 1 MiB.

The cache assumes the data does not change underneath. If you rewrite a source in
place, call `io.clear_cache()` (or `arr.clear_cache()` for one array) and the next read
recomputes.

Shapes broadcast like in NumPy, so a reduction with `keepdims=True` can be combined
with the full array:

```python
centred = vol - vol.mean(axis=0, keepdims=True)
```

An important point is that a larger result like this one, broadcast back over the axis
it reduced, is needed in full by every region along that axis. When it fits the region
budget, `io.write` shapes the regions to span that axis, so it is still read once. When
it does not, dyna recomputes it for each region and warns with the number of repeats
(`RepeatedReductionWarning`). The result is correct either way. To compute it only
once, persist it first:

```python
m = vol.mean(axis=0, keepdims=True).persist()    # computed once, into a temporary store
io.write(vol - m, "centred.zarr")
```

`persist()` writes to a temporary Zarr store that is removed when nothing uses it
anymore, or to a path you give it (`persist("mean.zarr")`). A result of up to 1 MiB is
simply kept in memory.

`histogram` is lazy as well and returns `(counts, edges)` like `numpy.histogram`.
Without `range=`, finding the range takes one extra pass. `unique` is lazy too, with
one catch: the length of its result depends on the data. So it reads nothing until
something needs its shape or values (`.shape`, `.size`, `compute()`, a write, or
building another operation on it), and then computes both in one pass and keeps them,
like the small reductions above.

## Operations catalog

About 140 operations, plus the three primitives they are built from. All are available flat on `dyna_zarr.operations` (`ops.gaussian_filter`) and also grouped by category submodule (`ops.neighborhood.gaussian_filter`). Every operation is lazy. Every operation is memory-bounded on the `io.write` path except `median`, `argmin`, and `argmax` (see Memory-boundedness), and `unique`, whose memory grows with the number of distinct values.

- **Pointwise, unary.**
  - Arithmetic and rounding: `abs`, `fabs`, `negative`, `positive`, `sign`, `sqrt`, `cbrt`, `square`, `reciprocal`, `floor`, `ceil`, `trunc`, `rint`, `round`, `clip`, `astype`, `conjugate`.
  - Exponents and logarithms: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`.
  - Trigonometric and hyperbolic: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `deg2rad`, `rad2deg`, `degrees`, `radians`.
  - Floating-point tests: `isfinite`, `isinf`, `isnan`, `signbit`, `spacing`.
- **Pointwise, binary.**
  - Arithmetic: `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `mod`, `remainder`, `fmod`, `power`, `float_power`, `maximum`, `minimum`, `fmax`, `fmin`, `hypot`, `arctan2`, `copysign`, `nextafter`, `logaddexp`, `logaddexp2`, `heaviside`, `ldexp`.
  - Integer and bitwise: `gcd`, `lcm`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`, `left_shift`, `right_shift`.
  - Comparisons: `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal`.
  - Logical: `logical_and`, `logical_or`, `logical_xor`, `logical_not`.
  - Selection and binning: `where`, `isin`, `digitize`.
- **Reductions.** Streaming and memory-bounded, with `axis=` and `keepdims=`: `min`, `max`, `sum`, `prod`, `mean`, `any`, `all`, `var`, `std`. Not fully bounded (hold the full reduced axis): `median`, `argmin`, `argmax`. Whole-array: `histogram` (lazy, returns `(counts, bin_edges)` like `numpy.histogram`, memory-bounded) and `unique` (lazy until its length is needed, a sorted array of the distinct values, bounded by how many there are). See [Reductions and statistics](#reductions-and-statistics).
- **Neighborhood (halo/overlap).** `gaussian_filter`, `uniform_filter`, `median_filter`, `minimum_filter`, `maximum_filter`, `grey_erosion`, `grey_dilation`, `convolve`, `correlate`, `laplace`, `gaussian_laplace`, `gaussian_gradient_magnitude`.
- **Structural.** `concatenate`, `stack`, `transpose`, `swap_axes`, `reshape`, `flatten`, `squeeze`, `expand_dims`, `pad`, `tile`, `roll`, `flip`, `rot90`, `slice_array`. Slicing follows NumPy, including negative steps (`arr[::-1]`). `pad` reads only what it needs at the borders, except for the statistic modes (`mean`, `median`, `maximum`, `minimum`, `linear_ramp`), which need the whole axis.
- **Differences.** `diff`, `gradient`.
- **Scan (prefix, along one axis).** `cumsum`, `cumprod`, `cummax`, `cummin`. Streamed with a bounded carry on the `io.write` path, so memory-bounded despite the sequential dependency.
- **Creation.** `zeros`, `ones`, `full`, `empty`, `random`, and `zeros_like`, `ones_like`, `full_like`, `empty_like`. `random` is position-deterministic, so the result is independent of chunking.
- **Primitives.** `map_blocks` (pointwise), `map_overlap` (neighborhood with a halo), `reduce` (streaming). Use these to build your own ops.

## Memory-bounded reshape and flatten

C-order reshape and flatten conflict with n-dimensional chunk layout, so a naive implementation blows up. dyna-zarr stages these through disk (a Rechunker-style two-phase, read-once/write-once copy), so peak RAM stays a function of the per-worker budget rather than the array size. When you `io.write` an outermost `reshape` or `flatten`, this path is used automatically:

```python
from dyna_zarr import io, operations as ops

arr = io.read("big_4d.zarr")                 # e.g. 5 GB, awkward chunks
io.write(ops.flatten(arr), "flat.zarr", region_size_mb=128, max_workers=2)
io.write(ops.reshape(arr, (a, b)), "reshaped.zarr")   # (a, b) is any target shape of the same size
```

The staging happens in the system temp directory (set `TMPDIR`, or `TEMP` on Windows,
to move it), never next to the output, so the output can also be remote. It needs free
local disk of up to about 2x the uncompressed array for `flatten` and 3x for `reshape`.
dyna checks this before anything is written and stops with an error that says how much
is needed and where. An outermost scan (`cumsum` and friends) is streamed with a bounded
carry and needs no staging. Everything else about the output (format, chunks,
compression, sharding, backend, `overwrite`) works as for any other write. Note that
`region_shape`, `memory_budget_mb` and `num_readers` belong to the region pipeline and
are refused for these writes.

Changing only the chunk grid needs no special call: `io.write(arr, path, chunks=...)`
streams it like any other write.

## GPU (optional)

With a CuPy install, run a chain on the GPU. Setting `device='cuda'` on a terminal call (`compute` or `io.write`) makes device-inheriting ops run on the GPU. A single host-to-device transfer happens at the first CUDA op and the data stays resident up the chain. Results are returned or written from the host.

```python
result = ops.gaussian_filter(arr, sigma=3)
out = result.compute(device="cuda")          # whole chain on the GPU
io.write(result, "out_gpu.zarr", device="cuda")  # per-region GPU compute, streamed write
```

## Remote storage

Both backends read and write `s3://` and `gs://` arrays, and read `http(s)://` ones,
with the same `storage_options`. Credentials come from the usual environment variables
(and `~/.aws` for S3). For any non-AWS S3 (MinIO, Ceph, ...), pass the endpoint:

```python
opts = {"endpoint": "https://s3.example.org",
        "region": "us-east-1",
        "virtual_hosted_style_request": False}

arr = io.read("s3://bucket/image.zarr", storage_options=opts)
io.write(arr, "s3://bucket/out.zarr", storage_options=opts)
io.write(arr, "s3://bucket/out2.zarr", backend="zarrista", storage_options=opts)
```

Add `"skip_signature": True` for a public bucket without credentials. Note that
TensorStore's AWS client may log that `~/.aws/config` or `~/.aws/credentials` cannot be
found. That is harmless when the credentials come from the environment.

The default TensorStore backend understands `endpoint`, `region`, `skip_signature`,
`virtual_hosted_style_request` and `client_options={"allow_http": ...}`, and refuses
anything else with an error rather than ignoring it. It takes no credentials in
`storage_options`, and `skip_signature` needs tensorstore 0.1.72 or newer. It cannot
write over plain `http(s)://` (its HTTP store is read-only) and has no store for `az://`.
`backend="zarrista"` covers both; see below.

## zarrista backend (optional)

[zarrista](https://github.com/developmentseed/zarrista) is a Rust-backed Zarr implementation (built on [zarrs](https://zarrs.dev/)). Installed via the `[zarrista]` extra, it can serve reads and writes in place of the default TensorStore path.

Measured against that default on the same machine, with matched codecs, matched concurrency, and byte-identical output verified in both directions:

| layout | write | read |
|---|---|---|
| Zarr v3, unsharded | 2.4x | 2.0x |
| Zarr v3, sharded | 1.6x | 2.2x |
| Zarr v2 | 1.6x | 1.5x |

Single machine, local filesystem, float32, so treat these as an indication, not a guarantee, and measure on your own data and hardware. The scripts are in `benchmarks/`.

Note that the default backend is tensorstore. Zarrista backend needs to be explicitly supplied. Simply pass the `backend="zarrista"` to the reader:

```python
arr = io.read("image.zarr", backend = "zarrista")
```

With zarrista, remote stores (`s3://`, `gs://`, `az://`, `http://`) are read **and** written through zarrista's async API over [obstore](https://github.com/developmentseed/obstore), which the `[zarrista]` extra installs. Credentials come from obstore's usual environment variables. It is possible to pass custom options with `storage_options`, which then goes directly to `obstore.store.from_url`:

```python
opts = {"endpoint": "https://s3.example.org",       # any non-AWS S3: MinIO, Ceph, ...
        "region": "us-east-1",
        "virtual_hosted_style_request": False,
        "skip_signature": True}                     # public bucket, no credentials

arr = io.read("s3://bucket/image.zarr", backend="zarrista", storage_options=opts)
io.write(arr, "s3://bucket/out.zarr", backend="zarrista", storage_options=opts)
```

A public bucket also works addressed directly by its `https://` URL, with no options at all:

```python
arr = io.read("https://s3.example.org/bucket/image.zarr/0", backend="zarrista")
```

obstore picks the store from the scheme: `https://` gives an HTTP store, which serves
ordinary key GETs and PUTs, enough to read and write an array, while `s3://` with
`endpoint=` gives a real S3 client. Prefer the `s3://` form when you need credentials,
signing, or anything that lists keys, since an HTTP store cannot list an S3 bucket (it
would need WebDAV `PROPFIND`).


## Relationship to dask

dyna-zarr is not a general replacement for `dask.array`. It targets one job: memory-bounded read, transform, and write of large Zarr/TIFF arrays.

The core idea is to drop the task graph. Because every operation is a pull-based chain, where each output slice maps back to a bounded input read, there is no graph to build and no scheduler to run it. That keeps the engine small, keeps peak RAM bounded by the region budget rather than the array size on the `io.write` path, and avoids scheduling overhead, which makes the read-transform-write pipeline efficient.

The tradeoff is that only operations that fit this slice-pushdown model belong in the chain: pointwise math, neighborhood/halo filters, streaming reductions, and structural reshaping. These are operations that are commonly used in image processing, which is what dyna-zarr is mainly built for. Operations that would need a global, data-dependent graph do not fit directly, and a few that do (such as non-associative reductions) trade extra reads or memory to stay correct.

Two more differences worth knowing:

- **Single machine, for now.** Parallelism today is threaded I/O within one process, plus the optional GPU path. There is no cluster or distributed execution yet; better and process-based parallelism is a possible future direction.
- **Narrower surface.** About 140 operations today, extended where the slice-pushdown model permits.

### Writing a dask array

dyna does not pull from a dask graph, but dask can push into a dyna output.
`io.create_sink` creates the output exactly as `io.write` would (format, chunks,
codecs, shards, dimension names, overwrite rule, local or remote, either backend) and
returns a sink that blocks are written into. Rechunk the dask array to
`sink.write_unit` first: that is the chunk, or the shard when sharded, and it keeps
concurrent writes from sharing one.

```python
import dask.array as da

x = da.from_zarr("in.zarr") + 1
sink = io.create_sink("out.zarr", x.shape, x.dtype, chunks=(8, 64, 64))
da.store(x.rechunk(sink.write_unit), sink, lock=False)
```

On a 1.2 GB array this was about 3x faster than storing into a zarr-python array.
`io.write` on a dyna array is still faster (0–30% here, depending on backend and
operation), so prefer it when you can build the array with dyna.

## Core components

- `io.read(source, backend=..., storage_options=...)` reads TIFF, Zarr v2, or Zarr v3 (local or remote) into a `DynamicArray`; `backend` is `"tensorstore"` (default) or the optional `"zarrista"`.
- `io.write(array, path, ...)` streams a `DynamicArray` to Zarr v2/v3, locally or remotely. Output: `zarr_format`, `chunks` or `chunk_size_mb`, `shard_coefficients`, `compressor`, `dtype`, `dimension_names`, `overwrite`. Pipeline: `region_size_mb` or `region_shape`, `max_workers`, `num_readers`, `memory_budget_mb`, `device`. Storage: `backend`, `storage_options`.
- `io.create_sink(path, shape, dtype, ...)` creates an output with the same output and storage options as `io.write` and returns a sink to push blocks into (`sink[key] = block`, e.g. from `dask.array.store`).
- `io.clear_cache()` forgets the cached small reductions (see [Reductions and statistics](#reductions-and-statistics)).
- `operations` is the lazy op set above.
- `DynamicArray` is the pull-based lazy array (slicing, `.compute()`, `.persist()`, operators with NumPy broadcasting, lazy reductions such as `.mean()`, `.astype`/`.clip`/`.round`, ufunc protocol, `.shape`/`.dtype`/`.chunks`/`.size`/`.nbytes`).
- `Codecs` is the compression configuration for Zarr v2 and v3.
- `backends.zarrista_backend` implements the optional `backend="zarrista"` path used by `io.read` and `io.write`.

## Running the tests

`[dev]` installs the tooling only, including [moto](https://github.com/getmoto/moto),
an in-process S3 emulator for the remote-storage tests, so they never touch a real
cloud store. Tests for the optional backends are marked, so they are selected
explicitly rather than skipped silently:

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
OS. A separate job installs every dependency at its **minimum** version on Python 3.11
and runs the suite there, so the version floors below are tested, not guessed. The
GPU tests are **not** run by CI, since GitHub's standard runners have no CUDA device,
so run `-m gpu` locally on each platform you support before releasing a change to the
device path.

## Requirements

- Python 3.11 or newer
- zarr 3.1.5+, numpy 1.26+, scipy 1.10+, tensorstore 0.1.62+, tifffile 2025.5.21+
- tensorstore 0.1.77 to 0.1.84 are excluded: after any S3 access they crash when the
  Python process exits (seen on Windows), which turns a finished job into a failure.
- Optional: CuPy (via the `gpu-cuXX` extras) for the GPU path
- Optional: zarrista + obstore (via the `zarrista` extra) for the faster storage backend;
  this extra also needs numpy 2.1+
