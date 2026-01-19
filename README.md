# dyna_zarr

A lightweight Python library for lazy operations on Zarr arrays.

## Overview

**dyna_zarr** provides a thin abstraction layer on top of
[zarr](https://zarr-python.readthedocs.io/) for lazy and dynamic array processing. It is designed to simplify working with large, multidimensional datasets by enabling memory-efficient, region-wise I/O and computation.

## Features

- **Lazy evaluation** – Operations are deferred until explicitly computed
- **Multi-format I/O** – Read from TIFF, Zarr v2, and Zarr v3; write to Zarr v2 or Zarr v3
- **Efficient region-wise processing** – Data is processed in regions, where each region may span multiple chunks
- **Minimal dependencies** – Requires only `zarr`, `numpy`, `tensorstore`, and `tifffile`

## Installation

```bash
pip install dyna_zarr
```

For development:

```bash
git clone https://github.com/bugraoezdemir/dyna_zarr.git
cd dyna_zarr
pip install -e ".[dev]"
```

## Quick Start

### Reading Arrays

```python
from dyna_zarr import io

# Read a TIFF file as a DynamicArray
arr = io.read("image.tiff")
print(arr.shape)   # (100, 256, 256)
print(arr.dtype)   # dtype('uint16')

# Read Zarr v2
arr = io.read("array_v2.zarr")

# Read Zarr v3
arr = io.read("array_v3.zarr")

# Compute the full array into memory
data = arr.compute()

# Compute a region of interest
region = arr[10:20, 50:150, 100:200].compute()
```

### Writing Arrays

```python
from dyna_zarr import io, Codecs

# Write to Zarr v3
io.write(arr, "output_v3.zarr", zarr_format=3)

# Write to Zarr v2
io.write(arr, "output_v2.zarr", zarr_format=2)

# Specify custom chunks
io.write(arr, "output.zarr", chunks=(64, 64, 64), zarr_format=3)

# Enable compression
codecs = Codecs(compressor="zstd", clevel=5)
io.write(arr, "output.zarr", compressor=codecs, zarr_format=3)

# Convert dtype during write
io.write(arr, "output.zarr", dtype="float32", zarr_format=3)
```

### Lazy Operations

```python
from dyna_zarr import io, operations

# Read array
arr = io.read("input.zarr")

# Apply lazy transformations (no computation yet)
result = operations.abs(arr)
result = operations.clip(result, 0, 1)
result = operations.sqrt(result)

# Write result using region-wise processing
io.write(result, "output.zarr", zarr_format=3)

# Or fully materialize the result
final_data = result.compute()
```

### Multi-source Operations

```python
from dyna_zarr import io, operations

# Read multiple sources
arr1 = io.read("input1.zarr")
arr2 = io.read("input2.zarr")

# Chain operations
result = operations.concatenate([arr1, arr2], axis=0)
result = operations.clip(result, -1, 1)

# Write result
io.write(result, "concatenated_output.zarr", zarr_format=3)
```

## Core Components

- **`io.read()`** – Read TIFF, Zarr v2, or Zarr v3 sources and return a
  `DynamicArray`
- **`io.write()`** – Write a `DynamicArray` to Zarr v2 or v3 with optional
  region-wise execution
- **`operations`** – Lazy transformation functions such as `abs`,
  `clip`, `sqrt`, `concatenate`, `reshape`, and `slice`
- **`DynamicArray`** – Core lazy array abstraction supporting slicing,
  shape/dtype inspection, and `.compute()`
- **`Codecs`** – Compression configuration for Zarr v2 and v3

## Further Examples

### Manual Configuration of Region Size

```python
from dyna_zarr import io

arr = io.read("large_array.zarr")

# Process approximately 64 MB regions at a time
io.write(arr, "output.zarr", region_size_mb=64)
```

### Zarr Sharding (Zarr v3 only)

```python
io.write(
    arr,
    "sharded_output.zarr",
    chunks=(64, 64, 64),            # Inner chunk size
    shard_coefficients=(4, 4, 4),   # Shard size = 4x chunks in each dimension
    zarr_format=3,
)
```

## Requirements

- Python >= 3.11
- zarr >= 3.0.0
- numpy >= 1.20.0
- tensorstore
- tifffile

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src/dyna_zarr --cov-report=html
```

