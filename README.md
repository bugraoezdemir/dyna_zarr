# dyna-zarr

**Lazy, memory-bounded processing for large Zarr and TIFF arrays.**

`dyna-zarr` is a lightweight Python array layer for reading, transforming, and writing arrays that may be much larger than RAM. Operations are lazy and pull-based: when a slice is requested, dyna-zarr works backwards through the operation chain and reads only the input regions needed. It does not build a task graph or materialize intermediate arrays.

The result is a simple read → transform → write workflow for large images and other N-dimensional data, with optional GPU execution.

## Why dyna-zarr?

- **Memory-bounded streaming.** `io.write` processes the result region by region, so peak working memory is controlled mainly by the region size and number of threads, and not by the total array size.
- **Lazy operation chains.** Compose transformations without reading the data until you compute or write a result.
- **Image-processing operations.** About 150 operations, including pointwise math, reductions, neighborhood filters, reshaping, scans, and downsampling.
- **NumPy-like interface.** Use operators, array methods, broadcasting, and NumPy ufuncs such as `np.sqrt` on a `DynamicArray`.
- **Flexible storage.** Read TIFF and Zarr v2/v3; work with local or remote Zarr stores; choose output chunking, compression, and Zarr v3 sharding.
- **Optional GPU and storage backend.** Run operation chains on CUDA with CuPy, or read and write Zarr through the alternative Rust-based `zarrista` backend.

## A first example: process a volume larger than RAM

```python
from dyna_zarr import io, operations as ops

volume = io.read("volume.zarr")                 # nothing read yet
smoothed = ops.gaussian_filter(volume, sigma=(1, 2, 2))
mask = (smoothed > 0.5).astype("uint8")          # still lazy

io.write(mask, "mask.zarr", region_size_mb=64, max_threads=4)
```

The first three lines describe the computation, and the final line executes it. The filter and threshold form one pull-based chain, and `io.write` streams the result to disk region by region. Neighborhood filters read the required halo around each region, so results remain correct at region boundaries.

Peak memory depends on the selected region size, the number of threads, the dtypes and operations in the chain; see [Chunks versus regions](#chunks-versus-regions).

## Installation

```bash
pip install dyna-zarr
```

Optional GPU support: install the extra matching your CUDA setup.

```bash
pip install "dyna-zarr[gpu-cu12]"  # CUDA 12.x; [gpu] is an alias
pip install "dyna-zarr[gpu-cu11]"  # CUDA 11.x
pip install "dyna-zarr[gpu-cu13]"  # CUDA 13.x
```

Optional Rust-backed Zarr I/O:

```bash
pip install "dyna-zarr[zarrista]"
```

The `zarrista` extra installs `zarrista-bounded`, our forked build from the [zarrista package](https://github.com/developmentseed/zarrista). The default storage backend is TensorStore. The alternative backend can be selected explicitly with `backend="zarrista"`.

## Quick start

### Read and slice

```python
from dyna_zarr import io

image = io.read("image.tiff")       # TIFF
zarr2 = io.read("array_v2.zarr")    # Zarr v2
volume = io.read("volume_v3.zarr")  # Zarr v3

print(volume.shape, volume.dtype, volume.chunks)

small_region = volume[10:20, 50:150, 100:200]
mask = small_region > 128          # still lazy; only this region will be read
io.write(mask, "mask.zarr")        # streams the masked region to disk
```

Remote Zarr stores on `s3://` and `gs://` can be both read and written. `http(s)://` stores are read-only with the default backend. See [Remote storage](#remote-storage).

### Wrap an array or reader you already have

`from_array` makes an array-like source lazy, including NumPy, Zarr, TensorStore, Dask, `micro_reader.Image`, or a custom reader. A custom reader needs `shape`, `dtype`, and `__getitem__` that returns a NumPy array.

```python
from dyna_zarr import from_array

lazy = from_array(existing_array)
lazy = from_array(reader, chunks=(1, 256, 256))
lazy = from_array(reader, lock=True)  # serialize reads for readers that need it
```

By default, dyna-zarr uses the source's chunk grid where available. You can supply `chunks=` to choose a different grid. Slicing pulls only the requested region; strided slices are read chunk by chunk to keep memory bounded.

### Compose NumPy-like operations

Operators, methods, broadcasting, and NumPy ufuncs build lazy operations rather than immediately computing an array.

```python
import numpy as np
from dyna_zarr import io, operations as ops

arr = io.read("input.zarr")
result = np.sqrt(ops.clip(ops.abs(arr), 0, 1))

io.write(result, "output.zarr", zarr_format=3)  # stream to disk
# Or: result.compute() to materialize the whole result
```

### Neighborhood filters

Neighborhood filters wrap `scipy.ndimage`. Each requested region includes the halo needed by the filter, so streamed output matches the corresponding whole-array operation, including at region boundaries.

```python
from dyna_zarr import io, operations as ops

image = io.read("volume.zarr")
log = ops.gaussian_laplace(image, sigma=2)
denoised = ops.median_filter(image, size=3)

io.write(log, "log.zarr")
io.write(denoised, "denoised.zarr")
```

### Lazy reductions and caching mechanism

Reductions such as `mean`, `std`, `min`, and `max` are lazy too. Small reduction results are automatically cached with the array, so they can be reused by later operations.

```python
from dyna_zarr import io, operations as ops

volume = io.read("volume.zarr")
smoothed = ops.gaussian_filter(volume, sigma=(1, 2, 2))
threshold = smoothed.mean() + 3 * smoothed.std()  # mean and std in one pass
mask = (smoothed > threshold).astype("uint8")

io.write(mask, "mask_3sd.zarr", region_size_mb=64, max_threads=4)
```

In this example, `mean` and `std` are computed together in one pass over the filtered volume and kept in memory, so the chain does not run once for each of them. Writing the mask then runs the chain a second time, region by region: the threshold depends on the whole volume, so it has to be known before the first region can be masked. To run the filter only once, `.persist()` the smoothed volume first, which stores it in a temporary Zarr store on disk.

Results up to 1 MiB (a scalar, per-channel statistics, a small projection) are kept in memory like this. To keep a larger result, call `.persist()` on it, e.g. m = vol.mean(axis=0).persist(), so it is computed once.

Most operations are memory-bounded when streamed with `io.write`. Exceptions include `median`, `argmin`, and `argmax`, which may need a full reduced axis, and `unique`, whose output size depends on the number of distinct values.

## Writing arrays and controlling memory

`io.write` streams a `DynamicArray` to a Zarr v2 or v3 store. It can also write Dask arrays; see [Relationship to Dask](#relationship-to-dask).

```python
from dyna_zarr import io, Codecs

io.write(arr, "out.zarr", zarr_format=3)
io.write(arr, "out_chunked.zarr", chunks=(8, 64, 64), zarr_format=3)
io.write(arr, "out_1mb_chunks.zarr", chunk_size_mb=1, zarr_format=3)
io.write(arr, "out_zstd.zarr",
         compressor=Codecs(compressor="zstd", clevel=5), zarr_format=3)
io.write(arr, "out_sharded.zarr", chunks=(8, 64, 64),
         shard_coefficients=(2, 2, 2))  # Zarr v3 shards of 2x2x2 chunks

# Control working memory and parallelism
io.write(arr, "out_bounded.zarr", region_size_mb=64, max_threads=4)
io.write(arr, "out_budgeted.zarr", memory_budget_mb=1024)
```

By default, output inherits the input's Zarr format, compressor, chunking, and sharding where applicable. For sources without a Zarr layout of their own, dyna-zarr chooses default values for all storage parameters.

### Chunks versus regions

These parameters control different things:

- **`chunks` / `chunk_size_mb`** set the output's storage layout. `chunks` is an exact shape; `chunk_size_mb` asks dyna-zarr to choose a suitable shape while aligning reads to the source grid where possible.
- **`region_shape` / `region_size_mb`** set the read-and-process region size used during a streamed write. Regions usually contain multiple chunks and primarily affect performance and peak memory. They **do not** affect the resulting array values.
- **`memory_budget_mb`** limits the additional memory the write may use by reducing concurrency when needed. It never raises `max_threads`. On local stores, shards larger than the budget are still written within it, chunk by chunk.
- **`max_threads`** controls concurrency. More threads can improve throughput, but also increase memory use.

A practical starting point is to set a moderate region size and increase `max_threads` while watching RAM and throughput. As a rough guide, budget for about 2.5–3 times `region_size_mb` per thread for region data and intermediate copies, plus fixed process overhead. The actual amount depends on the operation chain and dtypes.

Writing to an existing Zarr array or group raises `OutputExistsError`. Use `overwrite=True` to replace it; dyna-zarr checks that the target looks like a Zarr store before removing it.

### Write multiscale levels in one pass

`io.write_levels` can write an array and its downsampled levels from one source read, which is useful for image pyramids.

```python
from dyna_zarr import io

outputs = [
    {"path": "pyramid/0", "factors": (1, 1, 1)},
    {"path": "pyramid/1", "factors": (2, 2, 2)},
    {"path": "pyramid/2", "factors": (4, 4, 4)},
]

io.write_levels(arr, outputs, method="mean")
```

Factors are per-axis and relative to the input. Each output can specify its own `chunks` or `chunk_size_mb`; other write options apply to all outputs. If all requested levels cannot be produced in one pass, the return value indicates how many were written.

## Remote storage

The default TensorStore backend reads and writes Zarr on S3-compatible stores and Google Cloud Storage, and reads HTTP(S) stores. For a non-AWS S3 service such as MinIO or Ceph, pass endpoint options as needed:

```python
opts = {
    "endpoint": "https://s3.example.org",
    "region": "us-east-1",
    "virtual_hosted_style_request": False,
}

arr = io.read("s3://bucket/image.zarr", storage_options=opts)
io.write(arr, "s3://bucket/out.zarr", storage_options=opts)
```

Credentials normally come from the backend's supported environment configuration. TensorStore has no Azure (`az://`) support, and its HTTP store is read-only. The optional `zarrista` backend uses `obstore` for remote reads and writes and supports a different set of schemes and options. Select it with `backend="zarrista"` and consult the backend's documentation for provider-specific settings.

Read and write backends are independent, so an array can be read with one backend and written with the other.

## GPU (optional)

With the matching CuPy extra installed, set `device="cuda"` on a terminal operation such as `.compute()` or `io.write`. Device-capable operations in the chain then run on the GPU, with a host-to-device transfer at the first CUDA operation for each region.

```python
from dyna_zarr import io, operations as ops

arr = io.read("volume.zarr")
result = ops.gaussian_filter(arr, sigma=3)

io.write(result, "filtered.zarr", device="cuda")  # streamed GPU processing
# Or: result.compute(device="cuda") for an in-memory result
```

## Available operations

The operation set contains around **150 operations**, available as `dyna_zarr.operations` (often imported as `ops`) and grouped into category submodules. The table below shows several examples.

| Category | Examples |
|---|---|
| Pointwise math and comparisons | `sqrt`, `clip`, `add`, `where`, `isfinite` |
| Reductions and statistics | `min`, `max`, `sum`, `mean`, `std`, `histogram`, `unique` |
| Neighborhood filters | `gaussian_filter`, `median_filter`, `convolve`, `gaussian_laplace` |
| Structural operations | `concatenate`, `stack`, `transpose`, `reshape`, `flatten`, `pad`, `flip` |
| Differences and scans | `diff`, `gradient`, `cumsum`, `cummax` |
| Downsampling | `coarsen` |
| Array creation | `zeros`, `ones`, `full`, `random`, `zeros_like` |
| Custom operations | `map_blocks`, `map_overlap`, `reduce` |


C-order `reshape` and `flatten` are streamed through temporary local storage to avoid materializing the full array in RAM. They may need free temporary disk space of up to about 2× the uncompressed array size for `flatten` and 3× for `reshape`. Set `TMPDIR` (or `TEMP` on Windows) to choose the location.

## Relationship to Dask

`dyna-zarr` is **not a general replacement for `dask.array`**. It focuses on a specific task: memory-bounded read–transform–write pipelines that suit its pull-based operation model. This model avoids building and scheduling a task graph, which makes it suitable for many image-processing chains that can be evaluated from bounded input regions.

`dyna-zarr` currently runs on a single machine, using threaded I/O and an optional GPU path; it does not yet provide a built-in distributed execution.

Existing Dask arrays can be wrapped and incorporated into dyna-zarr operation chains using `from_array`. Additionally, `io.write` accepts a Dask array directly.

```python
from dyna_zarr import io

x = y + 1  # where y is a dask array
io.write(x, "out.zarr", chunks=(8, 64, 64), max_threads=8)  # the dask array, written directly
```

For new processing chains that fit dyna-zarr's slice-based model, using dyna-zarr operations directly usually avoids the overhead of building a Dask graph. For existing Dask workflows, passing the resulting Dask array to `io.write` also works. In that case, the underlying computation is owned by Dask, while dyna-zarr writes the results. 


## Core API

- **`io.read(source, ...)`** — read TIFF or Zarr into a lazy `DynamicArray`.
- **`from_array(source, chunks=None, *, lock=None)`** — wrap an array-like object or reader.
- **`io.write(array, path, ...)`** — stream an array to Zarr v2/v3, with controls for chunking, sharding, compression, regions, memory budget, threads, backend, and device.
- **`io.write_levels(array, outputs, ...)`** — write multiscale outputs from a shared source read when possible.
- **`io.create_sink(path, shape, dtype, ...)`** — create a Zarr destination and return a sink for pushing blocks from an external producer.
- **`io.clear_cache()`** — clear cached small reduction results.
- **`operations`** (`from dyna_zarr import operations as ops`) — about 150 lazy operations: pointwise math, reductions, neighborhood filters, structural and scan operations, downsampling, array creation, and `map_blocks` / `map_overlap` / `reduce` for your own; see [Available operations](#available-operations).
- **`DynamicArray`** — lazy slicing, `.compute()`, `.persist()`, NumPy-like operators and ufuncs, broadcasting, and reductions.
- **`Codecs`** — configure compression for Zarr v2 and v3.

## Requirements and tests

- Python 3.11 or newer.
- Core dependencies include Zarr, NumPy, SciPy, TensorStore, and tifffile. The optional `zarrista` extra also requires NumPy 2.1 or newer.
- TensorStore versions 0.1.77–0.1.84 are excluded because they have a known process-exit crash after S3 access in affected environments. The package dependency constraints enforce the supported range.

Run the core tests locally:

```bash
pip install -e ".[dev]"
pytest -m "not zarrista and not gpu"
```

Run optional backend tests with the corresponding extras:

```bash
pip install -e ".[dev,zarrista]"
pytest -m zarrista

pip install -e ".[dev,gpu-cu12]"  # choose the CUDA extra for your setup
pytest -m gpu                     # requires a CUDA device
```

CI tests the core suite across Linux, Windows, and macOS, as well as the optional Zarrista backend. GPU tests require CUDA hardware and are run separately.
