# dyna_zarr

A lightweight library for lazy operations on Zarr arrays.

## Overview

**dyna_zarr** provides a thin layer around [zarr](https://zarr-python.readthedocs.io/) enabling lazy/dynamic array processing. It is intended to facilitate large, multidimensional array operations with memory-efficient region-wise I/O & computation. 


### Features

- **Lazy evaluation** - Operations don't execute until explicitly computed
- **Multi-format I/O** - Read from TIFF, Zarr v2, and Zarr v3; write to Zarr v2 or v3
- **Efficient region-wise processing** - Processes data
region-wise (where each single region may consist of multiple chunks)
- **Minimal dependencies** - Only requires zarr, numpy, tensorstore, and tifffile

## Installation

\`\`\`bash
pip install dyna_zarr
\`\`\`

Or for development:

\`\`\`bash
git clone https://github.com/bugraoezdemir/dyna_zarr.git
cd dyna_zarr
pip install -e ".[dev]"
\`\`\`

## Quick Start

### Reading Arrays

\`\`\`python
from dyna_zarr import io

# Read TIFF file as DynamicArray
arr = io.read("image.tiff")
print(arr.shape)      # (100, 256, 256)
print(arr.dtype)      # dtype('uint16')

# Read Zarr v2
arr = io.read("array_v2.zarr")

# Read Zarr v3
arr = io.read("array_v3.zarr")

# Compute full array into memory
data = arr.compute()

# Compute a region of interest
region = arr[10:20, 50:150, 100:200].compute()
\`\`\`

### Writing Arrays

\`\`\`python
# Write to Zarr v3 
io.write(arr, "output_v3.zarr", zarr_format=3)

# Write to Zarr v2
io.write(arr, "output_v2.zarr", zarr_format=2)

# Custom chunks
io.write(arr, "output.zarr", chunks=(64, 64, 64), zarr_format=3)

# With compression
from dyna_zarr import Codecs
codecs = Codecs(compressor='zstd', clevel=5)
io.write(arr, "output.zarr", compressor=codecs, zarr_format=3)

# Convert dtype during write
io.write(arr, "output.zarr", dtype='float32', zarr_format=3)
\`\`\`

### Lazy Operations

\`\`\`python
from dyna_zarr import operations

# Read array
arr = io.read("input.zarr")

# Apply lazy transformations (no computation yet)
result = operations.abs(arr)
result = operations.clip(result, 0, 1)
result = operations.sqrt(result)

# Write region-wise
io.write(result, "output.zarr", zarr_format=3)

# Or fully materialize the result
final_data = result.compute()
\`\`\`

### Multi-source Operations

\`\`\`python
from dyna_zarr import operations

# Read multiple sources
arr1 = io.read("input1.zarr")
arr2 = io.read("input2.zarr")

# Chain operations
result = operations.concatenate([arr1, arr2], axis=0)
result = operations.clip(result, -1, 1)

# Write result
io.write(result, "concatenated_output.zarr", zarr_format=3)
\`\`\`

## Core Components

- **`io.read()`** - Read from TIFF, Zarr v2, or Zarr v3 files → returns `DynamicArray`
- **`io.write()`** - Write `DynamicArray` to Zarr v2 or v3 with optional region-wise processing
- **`operations`** - Lazy transformation functions: `abs`, `clip`, `sqrt`, `concatenate`, `reshape`, `slice`, etc.
- **`DynamicArray`** - Core lazy array class supporting slicing, shape/dtype properties, and `.compute()`
- **`Codecs`** - Compression configuration for Zarr v2 and v3

## Further Examples

### Manual Configuration of Region Size

\`\`\`python
# Specify region size
arr = io.read("large_array.zarr")  
io.write(arr, "output.zarr", region_size_mb=64) # Process 64 MB regions at a time
\`\`\`

### Zarr Sharding (v3 only)

\`\`\`python
# Create sharded Zarr v3 array
io.write(
    arr,
    "sharded_output.zarr",
    chunks=(64, 64, 64),           # Inner chunk size
    shard_coefficients=(4, 4, 4),  # 4x multiplier for shard size
    zarr_format=3
)
\`\`\`

## Requirements

- Python >= 3.11
- zarr >= 3.0.0
- numpy >= 1.20.0
- tensorstore
- tifffile

## Testing

Run the test suite:

\`\`\`bash
pytest tests/ -v
\`\`\`

With coverage:

\`\`\`bash
pytest tests/ --cov=src/dyna_zarr --cov-report=html
\`\`\`

