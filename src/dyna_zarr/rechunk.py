"""Two-phase disk-staged rechunk (Rechunker's algorithm, dask-free).

Rechunking source chunks -> target chunks directly can blow memory when the chunk grids
don't align. Rechunker's fix: stage through an on-disk intermediate whose chunks divide
BOTH grids, so each hop is chunk-aligned -> each source chunk read once, each target chunk
written once, memory bounded to one chunk.

  int_chunks[d] = gcd(source_chunks[d], target_chunks[d])   # divides both grids

  - source divides target (source finer)  -> one pass, consolidate (iterate target chunks).
  - target divides source (target finer)   -> one pass, split (iterate source chunks).
  - otherwise                              -> two passes via a disk intermediate (int_chunks):
        phase 1  source -> intermediate   (iterate source chunks; int|source -> aligned)
        phase 2  intermediate -> target   (iterate target chunks; int|target -> aligned)

Same-shape only (it's a re-chunk, not a re-shape). reshape/flatten build on top of this by
rechunking to a flat-contiguous layout, then relabeling the shape.
"""

import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import zarr

from .utils import parse_dtype
from .operations._backend import asnumpy

DEFAULT_MAX_MEM = 256 * 1024 * 1024     # per-worker region-buffer budget, in bytes
DEFAULT_MAX_WORKERS = 4                  # concurrent read->write regions


def _read(src, sl):
    from .dynamic_array import DynamicArray
    if isinstance(src, DynamicArray):
        return asnumpy(src._read_direct(sl))
    return np.asarray(src[sl])


def _expand_region(base_chunk, shape, budget):
    """Grow a copy region outward from one ``base_chunk`` (the coarser grid's chunk) by WHOLE
    chunk multiples -- innermost axis first (C-order, keeps reads contiguous) -- while the
    element count stays within ``budget``. Result is chunk-aligned on both grids (so read-once
    / write-once holds) and bounds each worker's buffer to ~budget. For tiny chunks this
    batches many into one big I/O; for a chunk already >= budget it stays at one chunk (floor).
    """
    ndim = len(shape)
    region = [min(int(base_chunk[d]), int(shape[d])) for d in range(ndim)]
    for ax in range(ndim - 1, -1, -1):
        while region[ax] < shape[ax]:
            nxt = min(region[ax] + int(base_chunk[ax]), int(shape[ax]))
            prod = 1
            for d in range(ndim):
                prod *= nxt if d == ax else region[d]
            if prod <= budget:
                region[ax] = nxt
            else:
                break
    return tuple(region)


def _copy(src, dst, base_chunk, shape, budget, max_workers):
    """Copy src->dst region by region. Each region is a multiple of ``base_chunk`` (the COARSER
    grid) sized to ``budget`` -> chunk-aligned both sides (read-once / write-once), disjoint ->
    parallel-safe (distinct chunk files). Up to ``max_workers`` regions in flight => peak
    ~= max_workers * region_bytes."""
    ndim = len(shape)
    region = _expand_region(base_chunk, shape, budget)
    origins = list(product(*[range(0, shape[d], region[d]) for d in range(ndim)]))

    def task(origin):
        sl = tuple(slice(o, min(o + region[d], shape[d]))
                   for d, o in zip(range(ndim), origin))
        dst[sl] = _read(src, sl)

    if max_workers <= 1 or len(origins) <= 1:
        for o in origins:
            task(o)
    else:
        # ThreadPoolExecutor runs <= max_workers tasks concurrently; reads happen INSIDE the
        # task, so at most max_workers regions are resident at once. list() surfaces exceptions.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(task, origins))


def rechunk(source, target_chunks, output_path, dtype=None, zarr_format=2,
            intermediate_path=None, max_mem=DEFAULT_MAX_MEM, max_workers=DEFAULT_MAX_WORKERS):
    """Rechunk ``source`` (a DynamicArray or zarr, same output shape) to ``target_chunks`` at
    ``output_path``, disk-staged and memory-bounded. Returns output_path.

    Memory model (per-worker, matching io.write's region_size_mb + max_workers): each worker
    buffers up to ``max_mem`` bytes (one region = as many whole aligned chunks as fit, >= one
    chunk); ``max_workers`` regions run concurrently, so peak ~= ``max_workers * max_mem``.
    """
    shape = tuple(int(s) for s in source.shape)
    ndim = len(shape)
    sc = tuple(int(c) for c in (source.chunks or shape))
    tc = tuple(int(c) for c in target_chunks)
    if len(tc) != ndim:
        raise ValueError(f"target_chunks {tc} rank != source rank {ndim}")
    dt = parse_dtype(dtype if dtype is not None else source.dtype)[0]
    budget = max(1, int(max_mem) // dt.itemsize)     # region budget in elements

    out = zarr.open(str(output_path), mode="w", shape=shape, chunks=tc,
                    dtype=dt, zarr_format=zarr_format)

    src_divides_tgt = all(tc[d] % sc[d] == 0 for d in range(ndim))   # source finer
    tgt_divides_src = all(sc[d] % tc[d] == 0 for d in range(ndim))   # target finer

    if src_divides_tgt:
        _copy(source, out, tc, shape, budget, max_workers)   # consolidate: base = target chunk
    elif tgt_divides_src:
        _copy(source, out, sc, shape, budget, max_workers)   # split: base = source chunk
    else:
        ic = tuple(math.gcd(sc[d], tc[d]) for d in range(ndim))
        ipath = intermediate_path or (str(output_path) + ".rechunk_int.zarr")
        interm = zarr.open(ipath, mode="w", shape=shape, chunks=ic, dtype=dt,
                           zarr_format=zarr_format)
        _copy(source, interm, sc, shape, budget, max_workers)   # phase 1 (int | source)
        _copy(interm, out, tc, shape, budget, max_workers)      # phase 2 (int | target)
        try:
            shutil.rmtree(ipath)
        except Exception:
            pass
    return output_path


def flatten_write(source, output_path, output_chunks=None, max_mem=DEFAULT_MAX_MEM,
                  max_workers=DEFAULT_MAX_WORKERS, dtype=None, zarr_format=2):
    """Flatten ``source`` (nd) to 1D at ``output_path``, memory-bounded + read-once.

    Two steps: (1) rechunk source to a FLAT-CONTIGUOUS layout (read-once source via the
    rechunk engine); (2) relabel those flat-contiguous slabs to the 1D output (read-once).
    The nd-chunk -> flat-order reorder happens inside the disk-staged rechunk, so RAM stays
    bounded to ~one flat-contiguous unit regardless of array size.

    Sizing the flat-contiguous unit to ``max_mem``: a chunk is a single C-order contiguous
    flat run iff every axis AFTER some split axis ``a`` is full and every axis BEFORE is 1,
    i.e. ``(1,..,1, ca, D_{a+1}..D_last)`` -> ``ca * suffix[a]`` contiguous elements
    (``suffix[a] = prod(shape[a+1:])``). Pick the shallowest ``a`` whose full-trailing block
    ``suffix[a]`` already fits the budget, then take as many ``ca`` steps as fit. This bounds
    the unit to ``max_mem`` even when a whole trailing row (split axis 0) would blow it.
    """
    shape = tuple(int(s) for s in source.shape)
    ndim = len(shape)
    dt = parse_dtype(dtype if dtype is not None else source.dtype)[0]
    budget = max(1, int(max_mem) // dt.itemsize)              # in elements

    suffix = [1] * ndim                                       # suffix[d] = prod(shape[d+1:])
    for d in range(ndim - 2, -1, -1):
        suffix[d] = suffix[d + 1] * shape[d + 1]

    a = ndim - 1                                              # split axis (deepest = last)
    for d in range(ndim):
        if suffix[d] <= budget:                              # trailing block already fits
            a = d
            break
    ca = max(1, min(shape[a], budget // suffix[a]))           # steps of the split axis that fit
    ff_chunks = tuple(1 if d < a else (ca if d == a else shape[d]) for d in range(ndim))

    ff_path = str(output_path) + ".flatten_ff.zarr"
    rechunk(source, ff_chunks, ff_path, dtype=dt, zarr_format=zarr_format,
            max_mem=max_mem, max_workers=max_workers)        # read-once source, parallel
    ff = zarr.open(ff_path, mode="r")

    N = suffix[0] * shape[0]
    unit = ca * suffix[a]                                     # elements per flat-contiguous slab
    oc = tuple(output_chunks) if output_chunks is not None else (min(N, unit),)
    out = zarr.open(str(output_path), mode="w", shape=(N,), chunks=oc, dtype=dt,
                    zarr_format=zarr_format)

    # Relabel flat-contiguous units to the 1D output, in C-order: leading axes one index at a
    # time, split axis in steps of ca, trailing axes whole -> each unit is a contiguous output
    # range. Kept SEQUENTIAL: user output_chunks need not align to units, so parallel writes
    # could collide on a shared output chunk. Memory here = one unit (<= max_mem); it's mostly
    # I/O, and the expensive reorder already happened (parallel) in the rechunk above.
    lead_ranges = [range(shape[d]) for d in range(a)] + [range(0, shape[a], ca)]
    for origin in product(*lead_ranges):
        lead, ka = origin[:a], origin[a]
        da = min(ca, shape[a] - ka)
        sl = tuple([slice(i, i + 1) for i in lead] + [slice(ka, ka + da)]
                   + [slice(None)] * (ndim - a - 1))
        block = np.asarray(ff[sl])                            # one ff slab (read-once)
        f0 = sum(lead[d] * suffix[d] for d in range(a)) + ka * suffix[a]
        out[f0:f0 + da * suffix[a]] = block.reshape(-1)
    shutil.rmtree(ff_path, ignore_errors=True)
    return output_path


def reshape_write(source, target_shape, output_path, output_chunks=None,
                  max_mem=DEFAULT_MAX_MEM, max_workers=DEFAULT_MAX_WORKERS,
                  dtype=None, zarr_format=2):
    """Reshape ``source`` -> ``target_shape`` (C-order) at ``output_path``, memory-bounded.

    Two staged steps: (1) flatten source to a 1D CONTIGUOUS ``F`` (read-once source, via the
    disk-staged rechunk engine); (2) reindex ``F`` -> ``target_shape`` streaming per output
    chunk. Feeding reindex a 1D-contiguous input is the key: reindex was only slow on nd
    sources because a flat range forced whole nd chunks -- over contiguous ``F`` each output
    chunk maps to contiguous ``F`` ranges, so there's no chunk-amplification. Peak memory ~
    max(one flatten unit [<= max_mem], one F chunk + one output chunk of the reindex).

    Shape change happens only at the F -> target flat-identity relabel (output flat index ==
    input flat index); everything else is same-shape staging.
    """
    from .dynamic_array import DynamicArray
    from .reindex import reindex_write

    target_shape = tuple(int(s) for s in target_shape)
    src_shape = tuple(int(s) for s in source.shape)
    if int(np.prod(src_shape)) != int(np.prod(target_shape)):
        raise ValueError(f"reshape needs equal size: {src_shape} -> {target_shape}")
    dt = parse_dtype(dtype if dtype is not None else source.dtype)[0]

    N = int(np.prod(target_shape))
    if output_chunks is None:
        output_chunks = ((min(N, 1 << 20),) if len(target_shape) == 1
                         else tuple(min(s, 256) for s in target_shape))
    else:
        output_chunks = tuple(int(c) for c in output_chunks)

    f_path = str(output_path) + ".reshape_1d.zarr"
    f_chunk = (min(N, 1 << 22),)                              # 1D F chunks (~16MB f32), lean reindex input
    flatten_write(source, f_path, output_chunks=f_chunk, max_mem=max_mem,
                  max_workers=max_workers, dtype=dt, zarr_format=zarr_format)
    F = DynamicArray(zarr.open(f_path, mode="r"))
    try:
        reindex_write(F, target_shape, output_path, output_chunks,
                      dtype=dt, zarr_format=zarr_format)
    finally:
        shutil.rmtree(f_path, ignore_errors=True)
    return output_path
