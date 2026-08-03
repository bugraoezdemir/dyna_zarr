"""Streaming reindex writer for reshape / flatten (a C-order flat re-index).

These conflict with nd chunk layout, so the region writer either materializes the whole
input or thrashes chunks. This engine instead is **output-chunk driven with per-input-chunk
streaming**:

  for each OUTPUT chunk O:
      buffer = empty(O)
      for each INPUT chunk that feeds O (read ONE AT A TIME, then released):
          scatter its overlapping elements into buffer
      write O once

So each output chunk is produced by a single task (no write race, at any parallelism), and
memory is hard-bounded to **one input chunk + one output chunk** (+ the O-sized index
arrays), regardless of how many input chunks feed O or how far apart they are. The cost is
read amplification -- an input chunk is re-read once per output chunk that needs it -- which
an optional bounded LRU cache (``cache_size`` input chunks) trades back for memory.

reshape/flatten are the flat-identity case: output flat index == input flat index.
"""

from itertools import product

import numpy as np
import zarr

from .utils import parse_dtype
from .operations._backend import asnumpy


def _c_strides(shape):
    strides = [1] * len(shape)
    for a in range(len(shape) - 2, -1, -1):
        strides[a] = strides[a + 1] * shape[a + 1]
    return strides


def reindex_write(input_array, out_shape, output_path, output_chunks,
                  dtype=None, zarr_format=2, cache_size=0):
    """Write ``input_array`` reshaped (C-order) to ``out_shape`` at ``output_path``,
    streaming per output chunk. Memory ~ one input chunk + one output chunk + O-sized index
    arrays. ``cache_size`` (input chunks) optionally caches recent input chunks to cut
    re-reads."""
    in_shape = tuple(int(s) for s in input_array.shape)
    out_shape = tuple(int(s) for s in out_shape)
    if int(np.prod(in_shape)) != int(np.prod(out_shape)):
        raise ValueError(f"reindex needs equal size: {in_shape} vs {out_shape}")
    in_chunks = tuple(int(c) for c in (input_array.chunks or in_shape))
    output_chunks = tuple(int(c) for c in output_chunks)
    dt = parse_dtype(dtype if dtype is not None else input_array.dtype)[0]

    out = zarr.open(str(output_path), mode="w", shape=out_shape,
                    chunks=output_chunks, dtype=dt, zarr_format=zarr_format)

    in_ndim, out_ndim = len(in_shape), len(out_shape)
    out_strides = _c_strides(out_shape)
    grid = tuple((in_shape[d] + in_chunks[d] - 1) // in_chunks[d] for d in range(in_ndim))

    cache = {} if cache_size > 0 else None
    order = []

    def read_input_chunk(cid, cc):
        if cache is not None and cid in cache:
            return cache[cid]
        sl = tuple(slice(cc[d] * in_chunks[d],
                         min(cc[d] * in_chunks[d] + in_chunks[d], in_shape[d]))
                   for d in range(in_ndim))
        block = np.asarray(asnumpy(input_array._read_direct(sl)))
        if cache is not None:
            cache[cid] = block
            order.append(cid)
            while len(order) > cache_size:
                cache.pop(order.pop(0), None)
        return block

    for origin in product(*[range(0, out_shape[d], output_chunks[d]) for d in range(out_ndim)]):
        o_sl = tuple(slice(origin[d], min(origin[d] + output_chunks[d], out_shape[d]))
                     for d in range(out_ndim))
        o_shape = tuple(s.stop - s.start for s in o_sl)

        # flat index (C-order in out_shape) of every element of this output chunk
        flat = np.zeros(o_shape, dtype=np.int64)
        for d in range(out_ndim):
            coord = (o_sl[d].start + np.arange(o_shape[d], dtype=np.int64)) * np.int64(out_strides[d])
            shp = [1] * out_ndim
            shp[d] = o_shape[d]
            flat += coord.reshape(shp)
        flat = flat.ravel()

        # same flat index in the INPUT -> input coords + which input chunk each belongs to
        in_coords = np.unravel_index(flat, in_shape)
        chunk_lin = np.ravel_multi_index(
            tuple(in_coords[d] // in_chunks[d] for d in range(in_ndim)), grid)

        obuf = np.empty(flat.shape[0], dtype=dt)
        for cid in np.unique(chunk_lin):
            cc = np.unravel_index(int(cid), grid)
            block = read_input_chunk(int(cid), cc)          # ONE input chunk at a time
            mask = chunk_lin == cid
            local = tuple(in_coords[d][mask] - cc[d] * in_chunks[d] for d in range(in_ndim))
            obuf[mask] = block[local]
        out[o_sl] = obuf.reshape(o_shape)

    return output_path
