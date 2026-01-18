# dyna_zarr

A lightweight library for lazy operations on Zarr arrays without task graph overhead.

## Overview

dyna_zarr provides a thin layer around [zarr.Array](https://zarr-python.readthedocs.io/) that enables lazy/dynamic array processing, similar to [dask.array](https://docs.dask.org/en/stable/array.html) but without the task graph overhead. It is optimized for large-scale (terabyte+) Zarr datasets.

### Features

- **Lazy evaluation** without task graph complexity
- **Efficient TIFF reading** via tifffile's zarr bridge with concurrent access support
- **Minimal dependencies** (zarr, numpy)
- **Optimized for large datasets** (TB+ scale)

## Installation

```bash
pip install dyna_zarr
```

Or for development:

```bash
pip install -e .
```

## Quick Start

```python
from dyna_zarr import DynamicArray, operations

# Open or create a Zarr array
array = DynamicArray.from_zarr("path/to/array.zarr")

# Perform lazy operations
result = operations.slice_and_compute(array, slices)

# Read TIFF files as Zarr arrays
from dyna_zarr import read_tiff_lazy
tiff_array = read_tiff_lazy("path/to/image.tiff")
```

## Core Components

- **DynamicArray**: Main class for lazy array operations
- **operations**: Namespace with array manipulation functions
- **io**: Input/output utilities for reading and writing arrays
- **codecs**: Compression and encoding support
- **tiff_reader**: Specialized utilities for TIFF file handling

## Requirements

- Python >= 3.8
- zarr >= 3.0.0
- numpy >= 1.20.0

## License

MIT

## Author

Bugra Oezdemir (bugraa.ozdemir@gmail.com)

## Links

- [GitHub](https://github.com/bugraoezdemir/dyna_zarr)
- [Documentation](https://dyna-zarr.readthedocs.io)
