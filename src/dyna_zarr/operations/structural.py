"""Structural / coordinate transforms: concat, stack, slice, transpose, reshape,
squeeze, flatten, pad, tile, roll, flip, expand_dims, swap_axes (+ slice_array)."""

import numpy as np
from typing import Tuple, Union, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dyna_zarr.dynamic_array import DynamicArray


from ._base import Transform, _is_int_index, _perm_on_surviving
from ._backend import array_namespace


def _reshape_read(array, new_shape, key):
    """Windowed read for reshape/flatten (a C-order flat re-index).

    An output region's elements occupy flat indices [fmin, fmax] (same in input and output
    C-order). We read only the covering rectangle of input rows spanning that flat range,
    then gather the region's flat positions from it. Memory ~= region + one inner slab, so
    contiguous reads (io.write / compute) are memory-bound; only scatter patterns (a stepped
    output) can widen the flat span toward the whole array.
    """
    axinfo = _norm_key(key, new_shape)
    nnd = len(new_shape)
    nstride = [1] * nnd
    for a in range(nnd - 2, -1, -1):
        nstride[a] = nstride[a + 1] * new_shape[a + 1]

    kept_axes = [a for a, (is_int, *_r) in enumerate(axinfo) if not is_int]
    kept_shape = tuple(len(range(axinfo[a][1], axinfo[a][2], axinfo[a][3])) for a in kept_axes)
    flat = np.zeros(kept_shape, dtype=np.int64)             # nd flat-index of each output elem
    for a, (is_int, start, stop, step) in enumerate(axinfo):
        st = int(nstride[a])
        if is_int:
            flat = flat + start * st
        else:
            coord = np.arange(start, stop, step, dtype=np.int64) * st
            shp = [1] * len(kept_shape)
            shp[kept_axes.index(a)] = coord.shape[0]
            flat = flat + coord.reshape(shp)

    in_shape = array.shape
    in_stride0 = 1
    for s in in_shape[1:]:
        in_stride0 *= s
    if flat.size == 0:
        block = array._read_direct((slice(0, 0),) + (slice(None),) * (len(in_shape) - 1))
        return array_namespace(block).reshape(block, kept_shape)
    fmin, fmax = int(flat.min()), int(flat.max())
    c0, c1 = fmin // in_stride0, fmax // in_stride0
    box = array._read_direct((slice(c0, c1 + 1),) + (slice(None),) * (len(in_shape) - 1))
    xp = array_namespace(box)
    flatbox = box.reshape(-1)
    idx = xp.asarray(flat - c0 * in_stride0)                # into flatbox, on the box's device
    return flatbox[idx]


def _norm_key(key, shape):
    """Per output axis -> (is_int, start, stop, step) with concrete non-negative values.
    Integer indices become (True, idx, idx+1, 1); the caller squeezes those axes."""
    if not isinstance(key, tuple):
        key = (key,)
    key = key + (slice(None),) * (len(shape) - len(key))
    out = []
    for a, k in enumerate(key):
        size = shape[a]
        if _is_int_index(k):
            idx = int(k) if k >= 0 else size + int(k)
            out.append((True, idx, idx + 1, 1))
        else:
            start, stop, step = k.indices(size)
            out.append((False, start, stop, step))
    return out


def _common_grid(arrays):
    """The chunk grid ALL ``arrays`` share, or None.

    A combined array has one grid only when its inputs agree; with differing grids, or
    any grid-less input, there is no single source grid to carry forward, and picking one
    input's (it used to be ``arrays[0]``'s) made the result depend on argument order.
    None lets io.write choose its default instead.
    """
    grids = [a.chunks for a in arrays]
    first = grids[0]
    if first is None or any(g is None or tuple(g) != tuple(first) for g in grids[1:]):
        return None
    return tuple(first)


class ConcatenateTransform(Transform):
    """
    Lazy concatenation of multiple arrays along an axis.
    """

    def __init__(self, arrays: Tuple['DynamicArray', ...], axis: int):
        super().__init__()
        self.arrays = arrays
        self.axis = axis

        # Validate shapes
        ref_shape = list(arrays[0].shape)
        for arr in arrays[1:]:
            for i, (s1, s2) in enumerate(zip(ref_shape, arr.shape)):
                if i != axis and s1 != s2:
                    raise ValueError(f"All arrays must have same shape except on axis {axis}")

        # Compute output shape
        self.shape = tuple(
            sum(arr.shape[axis] for arr in arrays) if i == axis else ref_shape[i]
            for i in range(len(ref_shape))
        )

        # A grid only if every input shares it (order-independent; see _common_grid).
        self.chunks = _common_grid(arrays)
        self.dtype = arrays[0].dtype

        # Precompute cumulative sizes for fast lookup
        self._cumulative_sizes = [0]
        for arr in arrays:
            self._cumulative_sizes.append(self._cumulative_sizes[-1] + arr.shape[axis])
        
        # OPTIMIZATION: Pre-compute optimal read strategies
        # Store whether arrays can be read sequentially (no interleaving)
        self._sequential_friendly = len(arrays) <= 4

    def read(self, key):
        """
        Read data by routing to appropriate source arrays.
        Optimized with efficient memory allocation and direct writing.
        """
        # Normalize key to tuple of slices
        if not isinstance(key, tuple):
            key = (key,)

        # Pad with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Convert to slices
        normalized_key = []
        for k, size in zip(key, self.shape):
            if isinstance(k, int):
                normalized_key.append(slice(k, k + 1))
            elif isinstance(k, slice):
                normalized_key.append(k)
            else:
                raise NotImplementedError(f"Indexing with {type(k)} not supported")

        axis_slice = normalized_key[self.axis]
        start, stop, step = axis_slice.indices(self.shape[self.axis])

        if step < 0:
            raise NotImplementedError("Negative-step slicing on the concat axis not supported yet")
        # Route the contiguous span [start, stop); the step is re-applied to the
        # assembled result below so we read each source array only once.

        # Find which arrays we need to read from
        arrays_to_read = []

        for i, arr in enumerate(self.arrays):
            arr_start = self._cumulative_sizes[i]
            arr_stop = self._cumulative_sizes[i + 1]

            # Check if this array overlaps with requested range
            if start < arr_stop and stop > arr_start:
                # Compute slice within this array
                local_start = max(0, start - arr_start)
                local_stop = min(arr.shape[self.axis], stop - arr_start)

                # Build the key for this array
                local_key = list(normalized_key)
                local_key[self.axis] = slice(local_start, local_stop)
                local_key = tuple(local_key)

                arrays_to_read.append((arr, local_key, local_stop - local_start))

        # OPTIMIZATION: Handle based on number of arrays
        if len(arrays_to_read) == 0:
            # Empty selection on the concat axis: return a correctly-shaped empty array
            # (reading the first source with a zero-length concat-axis slice fixes the
            # other axis sizes and the dtype without materialising any data).
            empty_key = list(normalized_key)
            empty_key[self.axis] = slice(0, 0)
            result = self.arrays[0]._read_direct(tuple(empty_key))
        elif len(arrays_to_read) == 1:
            # Single array - no concatenation needed
            result = arrays_to_read[0][0]._read_direct(arrays_to_read[0][1])
        else:
            # Multiple arrays - concatenate
            # Collect data from all arrays in order
            result_parts = []
            for arr, local_key, _ in arrays_to_read:
                # Use _read_direct to avoid creating more SliceTransforms
                data = arr._read_direct(local_key)
                result_parts.append(data)
            
            result = np.concatenate(result_parts, axis=self.axis)

        # Re-apply the step on the concat axis (routing above read the contiguous span).
        if step != 1:
            step_idx = (slice(None),) * self.axis + (slice(None, None, step),)
            result = result[step_idx]

        # Remove dimensions that were indexed with int
        squeeze_axes = [i for i, k in enumerate(key) if _is_int_index(k)]
        for ax in reversed(squeeze_axes):
            result = np.squeeze(result, axis=ax)

        return result


class StackTransform(Transform):
    """
    Lazy stacking of arrays along a new axis.
    """

    def __init__(self, arrays: Tuple['DynamicArray', ...], axis: int):
        super().__init__()
        self.arrays = arrays
        self.axis = axis

        # Validate all arrays have same shape
        ref_shape = arrays[0].shape
        for arr in arrays[1:]:
            if arr.shape != ref_shape:
                raise ValueError("All arrays must have the same shape for stacking")

        # Compute output shape (insert new dimension)
        self.shape = ref_shape[:axis] + (len(arrays),) + ref_shape[axis:]
        # No grid in, no grid out (a grid-less input used to crash here); a grid only if
        # every input shares it (order-independent; see _common_grid).
        src = _common_grid(arrays)
        self.chunks = (src[:axis] + (1,) + src[axis:]) if src is not None else None
        self.dtype = arrays[0].dtype

    def read(self, key):
        # Normalize key
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Extract index/slice for new axis
        new_axis_key = key[self.axis]

        # Build key for source arrays (remove new axis)
        source_key = key[:self.axis] + key[self.axis + 1:]

        # Determine which arrays to read
        if _is_int_index(new_axis_key):
            # Single array - the new axis is dropped
            return self.arrays[new_axis_key]._read_direct(source_key)

        # Slice on the new axis. Insert the stacked axis at its position among the
        # *surviving* source axes (earlier integer indices dropped their axes).
        idxs = range(*new_axis_key.indices(len(self.arrays)))
        insert_pos = sum(1 for j in range(self.axis) if not _is_int_index(key[j]))

        if len(idxs) == 0:
            # Empty selection: build an empty array of the correct per-part shape.
            sample = self.arrays[0]._read_direct(source_key)
            empty_shape = sample.shape[:insert_pos] + (0,) + sample.shape[insert_pos:]
            return np.empty(empty_shape, dtype=sample.dtype)

        parts = [self.arrays[i]._read_direct(source_key) for i in idxs]
        return np.stack(parts, axis=insert_pos)


def _positive_steps(key):
    """Rewrite every negative-step slice in ``key`` as the positive-step slice over the
    same elements; return ``(key, axes_to_flip)``, axes counted in the READ RESULT
    (integer indices drop theirs).

    Upstream reads never see a negative step. zarr refuses one outright
    (NegativeStepError) and TensorStore/zarrista are not asked to support one. It also
    removes an ambiguity: a composed/normalised negative slice marks "before index 0" as
    ``stop=-1``, which is right as a ``range()`` bound but means "the LAST element" in
    Python slice syntax - so passing it on could read the wrong data, not just fail.
    ``key`` elements here are concrete ints/slices (range semantics for the bounds).
    """
    out, flip, res_axis = [], set(), 0
    for k in key:
        if isinstance(k, slice) and k.step is not None and k.step < 0:
            n = len(range(k.start, k.stop, k.step))
            if n == 0:
                out.append(slice(0, 0, 1))
            else:
                last = k.start + (n - 1) * k.step            # the smallest index read
                out.append(slice(last, k.start + 1, -k.step))
                flip.add(res_axis)
        else:
            out.append(k)
        if not isinstance(k, (int, np.integer)):
            res_axis += 1
    return out, flip


class SliceTransform(Transform):
    """
    Lazy slicing of an array with support for np.newaxis.
    """

    def __init__(self, array: 'DynamicArray', key):
        super().__init__()
        self.array = array

        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # The grid is derived only when the source has one (no grid in, no grid out).
        chunks = array.chunks
        new_shape = []
        new_chunks = [] if chunks is not None else None
        normalized = []          # store CONCRETE, non-negative keys so shape/read math is
        original_dim = 0         # simple (negatives + None resolved once, here)

        for k in key:
            if k is np.newaxis:
                normalized.append(k)
                new_shape.append(1)
                if new_chunks is not None:
                    new_chunks.append(1)
                continue
            if original_dim >= array.ndim:
                raise IndexError("Too many indices for array")
            size = array.shape[original_dim]
            if isinstance(k, (int, np.integer)):
                ki = int(k) if k >= 0 else size + int(k)     # normalize negative index
                if not (0 <= ki < size):
                    raise IndexError(
                        f"index {k} out of bounds for axis {original_dim} of size {size}")
                normalized.append(ki)                         # int removes this dimension
            elif isinstance(k, slice):
                s = slice(*k.indices(size))                   # resolves None + negatives
                normalized.append(s)
                new_shape.append(len(range(s.start, s.stop, s.step)))
                if new_chunks is not None:
                    new_chunks.append(chunks[original_dim])
            else:
                raise TypeError(f"Invalid index type: {type(k)}")
            original_dim += 1

        # Add remaining (untouched) dimensions
        for i in range(original_dim, array.ndim):
            new_shape.append(array.shape[i])
            if new_chunks is not None:
                new_chunks.append(chunks[i])

        self.key = tuple(normalized)
        self.shape = tuple(new_shape)
        # If the source carries a read-alignment grid (a map_overlap(align=cell)), it stays
        # valid under this slice - just shifted. Expose the cell together with the OFFSET of
        # our first element inside it, so a consumer (io.write) can cut regions on the
        # underlying cell boundaries instead of a 0-based grid of its own. Without this the
        # consumer sees no alignment and falls back to small regions, and the producer
        # recomputes each cell once per region that touches it.
        self.align, self.align_offset = self._derive_align()
        # No grid in, no grid out. A grid-less source used to get a made-up (1, ..., 1)
        # grid here, which io.write then PRESERVED - one-voxel chunks on disk. With None,
        # the writer chooses its default chunk instead.
        self.chunks = tuple(new_chunks) if new_chunks is not None else None
        self.dtype = array.dtype
        self.new_axes = [i for i, k in enumerate(self.key) if k is np.newaxis]


    def _derive_align(self):
        """(cell, offset) inherited from the source's alignment, or (None, None).

        Only unit-step, rank-preserving slices inherit: a step or an added/dropped axis
        breaks the correspondence between our coordinates and the source's cells.
        """
        src = getattr(self.array, "_transform", None)
        cell = getattr(src, "align", None)
        if cell is None:
            return None, None
        off = getattr(src, "align_offset", None)
        ndim = self.array.ndim
        try:
            cell = tuple(int(c) for c in cell)
        except TypeError:
            return None, None
        if len(cell) != ndim or any(c <= 0 for c in cell):
            return None, None
        if len(self.key) != ndim or any(not isinstance(k, slice) for k in self.key):
            return None, None            # int index drops an axis, newaxis adds one
        if any((k.step or 1) != 1 for k in self.key):
            return None, None
        base = off if off is not None else (0,) * ndim
        # our element i maps to source element k.start + i; its phase in the source grid
        return cell, tuple((k.start + b) % c for k, b, c in zip(self.key, base, cell))

    def read(self, read_key):
        """
        Read data by composing the stored slice with a new read key.
        """
        if not isinstance(read_key, tuple):
            read_key = (read_key,)

        # Pad read_key to match output dimensions
        read_key = read_key + (slice(None),) * (len(self.shape) - len(read_key))

        # Build the key to apply to the underlying array
        full_key = []
        output_dim = 0  # Track dimension in the output (sliced) array
        input_dim = 0  # Track dimension in the input (underlying) array

        for stored_key_elem in self.key:
            if stored_key_elem is np.newaxis:
                # New axis - just track that we're moving to next output dimension
                # but don't consume an input dimension
                output_dim += 1
                continue

            # This is a real dimension in the underlying array
            if isinstance(stored_key_elem, int):
                # Dimension was removed by integer indexing in the stored slice
                full_key.append(stored_key_elem)
                input_dim += 1
                # Don't increment output_dim since this dimension is gone

            elif isinstance(stored_key_elem, slice):
                # Dimension was sliced in the stored slice
                read_elem = read_key[output_dim]

                # Normalize the incoming read element (resolve None + negatives) against
                # this output axis, so the composition below sees concrete non-negatives.
                osize = self.shape[output_dim]
                if isinstance(read_elem, (int, np.integer)):
                    read_elem = int(read_elem) if read_elem >= 0 else osize + int(read_elem)
                elif isinstance(read_elem, slice):
                    read_elem = slice(*read_elem.indices(osize))

                # Get parameters of the (already concrete) stored slice
                orig_size = self.array.shape[input_dim]
                stored_start = stored_key_elem.start
                stored_stop = stored_key_elem.stop
                stored_step = stored_key_elem.step

                if isinstance(read_elem, (int, np.integer)):
                    # Compose integer index with slice
                    new_idx = stored_start + read_elem * stored_step
                    if new_idx < 0 or new_idx >= orig_size:
                        raise IndexError(f"Index {new_idx} is out of bounds for axis with size {orig_size}")
                    full_key.append(new_idx)

                elif isinstance(read_elem, slice):
                    # Compose two slices
                    read_start = read_elem.start if read_elem.start is not None else 0
                    read_stop = read_elem.stop if read_elem.stop is not None else self.shape[output_dim]
                    read_step = read_elem.step if read_elem.step is not None else 1

                    new_start = stored_start + read_start * stored_step
                    new_stop = stored_start + read_stop * stored_step
                    new_step = stored_step * read_step

                    # Clamp to original array bounds
                    if new_step > 0:
                        new_start = max(0, min(new_start, orig_size))
                        new_stop = max(0, min(new_stop, orig_size))
                    else:
                        new_start = min(orig_size - 1, max(-1, new_start))
                        new_stop = min(orig_size - 1, max(-1, new_stop))

                    full_key.append(slice(new_start, new_stop, new_step))
                else:
                    raise TypeError(f"Invalid index type: {type(read_elem)}")

                input_dim += 1
                output_dim += 1
            else:
                raise TypeError(f"Invalid stored key type: {type(stored_key_elem)}")

        # Add any remaining dimensions that weren't in the stored key (normalize the
        # read element against the underlying axis so no negatives reach the base array)
        while input_dim < self.array.ndim:
            if output_dim < len(read_key):
                elem = read_key[output_dim]
                isize = self.array.shape[input_dim]
                if isinstance(elem, (int, np.integer)):
                    elem = int(elem) if elem >= 0 else isize + int(elem)
                elif isinstance(elem, slice):
                    elem = slice(*elem.indices(isize))
                full_key.append(elem)
                output_dim += 1
            else:
                full_key.append(slice(None))
            input_dim += 1

        # Read from underlying array - with positive steps only (see _positive_steps).
        full_key, flip_axes = _positive_steps(full_key)
        result = self.array._read_direct(tuple(full_key))
        if flip_axes:
            result = result[tuple(slice(None, None, -1) if ax in flip_axes else slice(None)
                                  for ax in range(result.ndim))]

        # Add back any newaxis dimensions at the correct positions
        for i, stored_key_elem in enumerate(self.key):
            if stored_key_elem is np.newaxis:
                # Count how many non-newaxis, non-int dimensions come before this
                axis_pos = sum(1 for j, k in enumerate(self.key[:i])
                               if k is not np.newaxis and not isinstance(k, int))
                result = np.expand_dims(result, axis=axis_pos)

        return result


class ExpandDimsTransform(Transform):
    """
    Lazy addition of a singleton dimension to an array.
    """

    def __init__(self, array: 'DynamicArray', axis: int):
        super().__init__()
        self.array = array
        self.axis = axis if axis >= 0 else array.ndim + axis + 1

        # Compute new shape and chunks
        self.shape = array.shape[:self.axis] + (1,) + array.shape[self.axis:]

        # Handle chunks - if chunks is None, keep it as None
        if array.chunks is not None:
            self.chunks = array.chunks[:self.axis] + (1,) + array.chunks[self.axis:]
        else:
            self.chunks = None

        self.dtype = array.dtype

    def read(self, key):
        """Read with the singleton dimension inserted at the correct axis."""
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        # Pad key with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Element that applies to the inserted (singleton) axis, and the key for the
        # underlying array (which lacks that axis).
        key_ins = key[self.axis]
        underlying_key = key[:self.axis] + key[self.axis + 1:]

        # Read from underlying array
        result = self.array._read_direct(underlying_key)

        # Insert the singleton axis at its position among the *surviving* axes: earlier
        # axes indexed by an integer were dropped by the read, shifting the position.
        insert_pos = sum(1 for j in range(self.axis) if not _is_int_index(key[j]))
        result = np.expand_dims(result, axis=insert_pos)

        # Apply the key element to the inserted axis (int drops it, slice sizes it 0/1).
        idx = (slice(None),) * insert_pos + (key_ins,) + \
              (slice(None),) * (result.ndim - insert_pos - 1)
        return result[idx]


class SwapAxesTransform(Transform):
    """
    Lazy swapping of two axes in an array.
    """

    def __init__(self, array: 'DynamicArray', axis1: int, axis2: int):
        super().__init__()
        self.array = array
        self.axis1 = axis1 if axis1 >= 0 else array.ndim + axis1
        self.axis2 = axis2 if axis2 >= 0 else array.ndim + axis2

        # Validate axes
        if self.axis1 >= array.ndim or self.axis2 >= array.ndim:
            raise ValueError(f"Axis out of bounds for {array.ndim}-D array")

        # Compute new shape and chunks (no grid in, no grid out - this used to crash)
        shape = list(array.shape)
        shape[self.axis1], shape[self.axis2] = shape[self.axis2], shape[self.axis1]
        self.shape = tuple(shape)
        if array.chunks is not None:
            chunks = list(array.chunks)
            chunks[self.axis1], chunks[self.axis2] = chunks[self.axis2], chunks[self.axis1]
            self.chunks = tuple(chunks)
        else:
            self.chunks = None
        self.dtype = array.dtype

    def read(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Pad key with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))
        key = list(key)
        
        # Swap the key indices to match the original array's axes
        original_key = key[:]
        original_key[self.axis1], original_key[self.axis2] = original_key[self.axis2], original_key[self.axis1]

        # Read from underlying array with unswapped key
        result = self.array._read_direct(tuple(original_key))

        # A swap is a permutation; apply it on the surviving axes so integer indices
        # that dropped an axis don't leave np.swapaxes with a stale axis index.
        axes = list(range(self.array.ndim))
        axes[self.axis1], axes[self.axis2] = axes[self.axis2], axes[self.axis1]
        return _perm_on_surviving(result, tuple(key), tuple(axes))


class TransposeTransform(Transform):
    """
    Lazy transposition (reordering) of array axes.
    """

    def __init__(self, array: 'DynamicArray', axes: tuple):
        super().__init__()
        self.array = array
        self.axes = tuple(ax if ax >= 0 else array.ndim + ax for ax in axes)

        # Validate axes
        if len(self.axes) != array.ndim:
            raise ValueError(f"axes don't match array (expected {array.ndim} dimensions, got {len(axes)})")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("repeated axis in transpose")
        if any(ax >= array.ndim or ax < 0 for ax in self.axes):
            raise ValueError("axis out of bounds for array")

        # Compute new shape and chunks
        self.shape = tuple(array.shape[ax] for ax in self.axes)
        if array.chunks is not None:
            self.chunks = tuple(array.chunks[ax] for ax in self.axes)
        else:
            self.chunks = None
        self.dtype = array.dtype

    def read(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Pad key with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Compute inverse permutation to map from transposed axes back to original
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i

        # Reorder the key according to the inverse permutation
        reordered_key = tuple(key[inv_axes[i]] for i in range(len(self.axes)))

        # Read from underlying array with reordered key
        result = self.array._read_direct(reordered_key)

        # Transpose to output order, honouring any integer indices that dropped an
        # axis (a plain np.transpose(result, self.axes) would use stale axis indices).
        return _perm_on_surviving(result, key, self.axes)


# Extended operations from extended_operations.py

class ReshapeTransform(Transform):
    """Lazy reshape without data copying."""
    
    def __init__(self, array: 'DynamicArray', shape: Tuple[int, ...]):
        super().__init__()
        self.array = array
        self.new_shape = shape
        
        # Validate reshape is possible
        if np.prod(array.shape) != np.prod(shape):
            raise ValueError(f"Cannot reshape array of size {np.prod(array.shape)} into shape {shape}")
        
        self.shape = tuple(shape)
        self.chunks = None  # Reshaping invalidates chunk alignment
        self.dtype = array.dtype
    
    def read(self, key):
        return _reshape_read(self.array, self.new_shape, key)


class SqueezeTransform(Transform):
    """Remove singleton dimensions.
    
    Supports NumPy-compatible squeeze with axis as None, int, or tuple of ints.
    """
    
    def __init__(self, array: 'DynamicArray', axis: Union[int, Tuple[int, ...], None] = None):
        super().__init__()
        self.array = array
        self.axis = axis
        
        # Normalize axis to tuple of indices to squeeze
        if axis is None:
            # Remove all singleton dimensions
            self.squeeze_axes = [i for i, s in enumerate(array.shape) if s == 1]
        elif isinstance(axis, int):
            # Single axis
            axis = axis if axis >= 0 else array.ndim + axis
            if axis < 0 or axis >= array.ndim:
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {array.ndim}")
            if array.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze axis {axis} with size {array.shape[axis]}")
            self.squeeze_axes = [axis]
        elif isinstance(axis, (tuple, list)):
            # Multiple axes
            squeeze_axes = []
            for ax in axis:
                ax = ax if ax >= 0 else array.ndim + ax
                if ax < 0 or ax >= array.ndim:
                    raise ValueError(f"axis {ax} is out of bounds for array of dimension {array.ndim}")
                if array.shape[ax] != 1:
                    raise ValueError(f"Cannot squeeze axis {ax} with size {array.shape[ax]}")
                squeeze_axes.append(ax)
            self.squeeze_axes = sorted(set(squeeze_axes))  # Remove duplicates and sort
        else:
            raise TypeError(f"axis must be None, int, or tuple of ints, got {type(axis)}")
        
        # Compute output shape
        self.shape = tuple(s for i, s in enumerate(array.shape) if i not in self.squeeze_axes)
        
        # Handle chunks
        if array.chunks:
            # Squeezing every axis leaves a 0-d array, whose grid is (); it used to be
            # reported as (1,) - a grid of the wrong rank.
            self.chunks = tuple(c for i, c in enumerate(array.chunks)
                                if i not in self.squeeze_axes)
        else:
            self.chunks = None
        
        self.dtype = array.dtype
    
    def read(self, key):
        """
        Lazy read that only accesses the required region of the underlying array.
        """
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        # Build unsqueezed key by inserting slice(None) for squeezed axes.
        unsqueezed_key = []
        squeezed_input_axes = []
        output_idx = 0
        for input_idx in range(self.array.ndim):
            if input_idx in self.squeeze_axes:
                # This axis was squeezed - insert full slice (it is size 1 upstream).
                unsqueezed_key.append(slice(None))
                squeezed_input_axes.append(input_idx)
            else:
                # This axis is preserved - use key from read operation
                if output_idx < len(key):
                    unsqueezed_key.append(key[output_idx])
                else:
                    unsqueezed_key.append(slice(None))
                output_idx += 1

        # Read only the required region from underlying array
        result = self.array._read_direct(tuple(unsqueezed_key))

        # A squeezed axis lands in the result at the position given by the number of
        # *surviving* (non-integer-indexed) axes before it - integer indices earlier in
        # the key drop their axes and shift everything left.
        to_remove = [
            sum(1 for j in range(input_idx) if not _is_int_index(unsqueezed_key[j]))
            for input_idx in squeezed_input_axes
        ]
        for ax in sorted(to_remove, reverse=True):
            result = np.squeeze(result, axis=ax)

        return result


class FlattenTransform(Transform):
    """Flatten array to 1D."""
    
    def __init__(self, array: 'DynamicArray'):
        super().__init__()
        self.array = array
        self.original_shape = array.shape
        # int(), not the np.int64 np.prod returns: the shape tuple is compared and
        # formatted all over the writer, and a numpy scalar there prints as
        # "np.int64(1048576)".
        self.shape = (int(np.prod(array.shape)),)
        self.chunks = None
        self.dtype = array.dtype
    
    def read(self, key):
        return _reshape_read(self.array, self.shape, key)


#: numpy.pad modes whose padded values are COPIES of input values (or one scalar
#: constant): every output position maps to one input index, so a window's input is
#: computable exactly. The statistic modes (mean/median/maximum/minimum/linear_ramp)
#: depend on whole axes and are read that way.
_PAD_REMAP_MODES = frozenset({"constant", "empty", "edge", "reflect", "symmetric", "wrap"})


class PadTransform(Transform):
    """Lazy padding (materializes on read). Supports numpy.pad ``mode`` (constant/reflect/
    edge/symmetric/wrap/...).

    Memory model. An axis whose read window lies entirely inside the core reads just that
    sub-slice. For the copy modes (see _PAD_REMAP_MODES; ``constant`` with a scalar value,
    ``reflect``/``symmetric`` with the default ``reflect_type='even'``) a window touching a
    padded border reads exactly the span of input indices it maps to - found by padding an
    INDEX array with the same mode - so an edge region costs about its own size. (Only a
    ``wrap`` window reaching both ends of an axis spans that whole axis.) The statistic
    modes, ``reflect_type='odd'`` and per-axis constants depend on whole axes: a bordered
    window reads the whole input axis, pads it and crops, as before.
    """

    def __init__(self, array: 'DynamicArray', pad_width: Union[int, Tuple],
                 mode: str = "constant", **pad_kwargs):
        super().__init__()
        self.array = array
        if isinstance(pad_width, int):
            self.pad_width = [(pad_width, pad_width)] * array.ndim
        else:
            self.pad_width = [(int(b), int(a)) for (b, a) in pad_width]
        self.mode = mode
        self.pad_kwargs = pad_kwargs      # e.g. constant_values=, reflect_type=
        self.shape = tuple(s + b + a for s, (b, a) in zip(array.shape, self.pad_width))
        self.chunks = None
        self.dtype = array.dtype

        # Exact bounded reads need a pure index remap (see the class docstring).
        fill = pad_kwargs.get("constant_values", 0)
        self._remap = (
            mode in _PAD_REMAP_MODES
            and pad_kwargs.get("reflect_type", "even") == "even"
            and np.ndim(fill) == 0
            and 0 not in tuple(array.shape)
        )
        self._fill = fill if np.ndim(fill) == 0 else None
        self._index_maps = {}

    def _index_map(self, axis):
        """Padded position -> input index along ``axis`` (-1 = constant fill)."""
        imap = self._index_maps.get(axis)
        if imap is None:
            n = self.array.shape[axis]
            idx = np.arange(n, dtype=np.int64)
            if self.mode in ("constant", "empty"):
                imap = np.pad(idx, self.pad_width[axis], mode="constant", constant_values=-1)
            else:
                imap = np.pad(idx, self.pad_width[axis], mode=self.mode)
            self._index_maps[axis] = imap
        return imap

    def read(self, key):
        if self._remap:
            return self._read_remap(key)
        return self._read_whole_axis(key)

    def _read_remap(self, key):
        """Bounded read for the copy modes: per bordered axis, read only the span of input
        indices the window maps to, then gather (and fill constants) along that axis."""
        input_slices, gathers, fills, squeeze_axes = [], [], [], []
        for a, (is_int, start, stop, step) in enumerate(_norm_key(key, self.shape)):
            before, _after = self.pad_width[a]
            in_size = self.array.shape[a]
            if start >= before and stop <= before + in_size:
                input_slices.append(slice(start - before, stop - before))
                gathers.append(slice(0, stop - start, step))
                fills.append(None)
            else:
                # range semantics for the positions (never a slice: stop=-1 would mean
                # "the last element" there)
                window = self._index_map(a)[np.arange(start, stop, step, dtype=np.int64)]
                real = window[window >= 0]
                lo = int(real.min()) if real.size else 0
                hi = int(real.max()) + 1 if real.size else 1     # >=1 element so take() works
                input_slices.append(slice(lo, hi))
                gathers.append(np.where(window >= 0, window - lo, 0))
                fills.append(window < 0 if (window < 0).any() else None)
            if is_int:
                squeeze_axes.append(a)

        out = self.array._read_direct(tuple(input_slices))
        xp = array_namespace(out)
        for a, g in enumerate(gathers):
            if isinstance(g, slice):
                out = out[(slice(None),) * a + (g,)]
            else:
                out = xp.take(out, xp.asarray(g), axis=a)
        if any(f is not None for f in fills):
            mask = None
            for a, f in enumerate(fills):
                if f is None:
                    continue
                shape = [1] * out.ndim
                shape[a] = f.shape[0]
                m = xp.asarray(f).reshape(shape)
                mask = m if mask is None else (mask | m)
            fill = 0 if self._fill is None else self._fill
            out = xp.where(mask, xp.asarray(fill, dtype=out.dtype), out)
        for a in sorted(squeeze_axes, reverse=True):
            out = xp.squeeze(out, axis=a)
        return out

    def _read_whole_axis(self, key):
        """Statistic modes: a bordered window needs the whole input axis."""
        input_slices, pad_widths, crop, squeeze_axes = [], [], [], []
        for a, (is_int, start, stop, step) in enumerate(_norm_key(key, self.shape)):
            before, after = self.pad_width[a]
            in_size = self.array.shape[a]
            if start >= before and stop <= before + in_size:
                # window entirely in the core: read just this sub-slice, no padding this axis
                input_slices.append(slice(start - before, stop - before))
                pad_widths.append((0, 0))
                crop.append(slice(0, stop - start, step))
            else:
                # window touches a padded border: read the whole input axis, pad it fully
                # (so the mode sees the real edge/interior), then crop the requested window
                input_slices.append(slice(0, in_size))
                pad_widths.append((before, after))
                crop.append(slice(start, stop, step))
            if is_int:
                squeeze_axes.append(a)
        core = self.array._read_direct(tuple(input_slices))
        xp = array_namespace(core)
        out = xp.pad(core, pad_widths, mode=self.mode, **self.pad_kwargs)
        out = out[tuple(crop)]
        for a in sorted(squeeze_axes, reverse=True):
            out = xp.squeeze(out, axis=a)
        return out


class TileTransform(Transform):
    """Repeat array along dimensions."""
    
    def __init__(self, array: 'DynamicArray', reps: Union[int, Tuple]):
        super().__init__()
        self.array = array
        
        if isinstance(reps, int):
            reps = (reps,) * array.ndim
        self.reps = reps
        
        # Calculate output shape
        self.shape = tuple(s * r for s, r in zip(array.shape, reps))
        # Tiling repeats the whole array, so a source chunk still tiles the output
        # exactly whenever it tiled the input: the grid carries over unchanged.
        src = array.chunks
        self.chunks = (tuple(int(c) for c in src)
                       if src is not None and len(src) == len(self.shape) else None)
        self.dtype = array.dtype
    
    def read(self, key):
        # Bounded by the INPUT (the tile), not the full tiled OUTPUT: read the tile once,
        # then gather the region via modulo indexing (out[i] = in[i % N] per axis).
        axinfo = _norm_key(key, self.shape)
        data = self.array._read_direct(tuple(slice(None) for _ in range(self.array.ndim)))
        xp = array_namespace(data)
        idx, squeeze_axes = [], []
        for a, (is_int, start, stop, step) in enumerate(axinfo):
            idx.append(xp.arange(start, stop, step) % self.array.shape[a])
            if is_int:
                squeeze_axes.append(a)
        result = data[xp.ix_(*idx)]
        for a in sorted(squeeze_axes, reverse=True):
            result = xp.squeeze(result, axis=a)
        return result


class RollTransform(Transform):
    """Roll array elements along an axis."""
    
    def __init__(self, array: 'DynamicArray', shift: int, axis: Optional[int] = None):
        super().__init__()
        self.array = array
        self.shift = shift
        self.axis = axis
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        if self.axis is None:
            # flatten-roll: genuinely global. Rare; keep the whole-input fallback.
            data = self.array._read_direct(tuple(slice(None) for _ in range(self.array.ndim)))
            return np.roll(data, self.shift)[key]
        # Memory-bound: out[i] = in[(i-shift) % N] along the axis, so a contiguous output
        # run maps to 1 or 2 wrapped input segments. Read those, other axes contiguous,
        # then crop step + squeeze ints.
        A = self.axis if self.axis >= 0 else self.array.ndim + self.axis
        N = self.array.shape[A]
        shift = (self.shift % N) if N else 0
        axinfo = _norm_key(key, self.shape)
        base = [slice(start, stop) for (_ii, start, stop, _st) in axinfo]
        _iiA, startA, stopA, _stA = axinfo[A]
        L = stopA - startA
        s0 = (startA - shift) % N if N else 0
        if N == 0 or s0 + L <= N:
            base[A] = slice(s0, s0 + L)
            block = self.array._read_direct(tuple(base))
        else:
            k1, k2 = list(base), list(base)
            k1[A] = slice(s0, N)
            k2[A] = slice(0, s0 + L - N)
            b1 = self.array._read_direct(tuple(k1))
            b2 = self.array._read_direct(tuple(k2))
            block = array_namespace(b1).concatenate([b1, b2], axis=A)
        xp = array_namespace(block)
        block = block[tuple(slice(0, stop - start, step) for (_ii, start, stop, step) in axinfo)]
        for a in sorted((i for i, (ii, *_r) in enumerate(axinfo) if ii), reverse=True):
            block = xp.squeeze(block, axis=a)
        return block


class FlipTransform(Transform):
    """Flip/reverse array along an axis."""
    
    def __init__(self, array: 'DynamicArray', axis: int):
        super().__init__()
        self.array = array
        self.axis = axis
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        # Memory-bound: read only the mirrored input window (positive-step slice, since
        # zarr/tensorstore reject negative steps), flip it in-memory, then crop step +
        # squeeze ints. out[start:stop) along the axis mirrors input[N-stop:N-start).
        A = self.axis if self.axis >= 0 else self.array.ndim + self.axis
        axinfo = _norm_key(key, self.shape)
        input_slices = []
        for a, (_is_int, start, stop, step) in enumerate(axinfo):
            if a == A:
                N = self.array.shape[a]
                input_slices.append(slice(N - stop, N - start))
            else:
                input_slices.append(slice(start, stop))
        block = self.array._read_direct(tuple(input_slices))
        xp = array_namespace(block)
        block = xp.flip(block, axis=A)
        block = block[tuple(slice(0, stop - start, step) for (_ii, start, stop, step) in axinfo)]
        for a in sorted((i for i, (ii, *_r) in enumerate(axinfo) if ii), reverse=True):
            block = xp.squeeze(block, axis=a)
        return block



def slice_array(array: 'DynamicArray', key) -> 'DynamicArray':
    """Create a lazy slice of an array."""
    transform = SliceTransform(array, key)
    return array._with_transform(transform)


# --------------------------------------------------------------------------- #
# Public ops -- structural / coordinate (this module owns both transforms + ops)
# --------------------------------------------------------------------------- #

def expand_dims(array, axis):
    """Add a new axis of length 1."""
    return array._with_transform(ExpandDimsTransform(array, axis))


def concatenate(arrays, axis=0):
    """Concatenate arrays along an existing axis."""
    if not arrays:
        raise ValueError("Need at least one array to concatenate")
    return arrays[0]._with_transform(ConcatenateTransform(tuple(arrays), axis))


def stack(arrays, axis=0):
    """Stack arrays along a new axis."""
    if not arrays:
        raise ValueError("Need at least one array to stack")
    return arrays[0]._with_transform(StackTransform(tuple(arrays), axis))


def swap_axes(array, axis1, axis2):
    """Swap two axes."""
    return array._with_transform(SwapAxesTransform(array, axis1, axis2))


def transpose(array, axes):
    """Permute array dimensions."""
    return array._with_transform(TransposeTransform(array, axes))


def reshape(array, shape):
    """Reshape array to a new shape (C-order).

    Memory: ``io.write`` streams this via the reindex writer (per output chunk, one input
    chunk at a time), so writing a reshape is hard-bounded to ~1 input chunk + 1 output
    chunk regardless of array size. A lazy ``compute()`` / sub-slice still uses a covering
    read (heavier, since a flat re-index conflicts with nd chunk layout).
    """
    return array._with_transform(ReshapeTransform(array, shape))


def squeeze(array, axis=None):
    """Remove singleton dimensions."""
    return array._with_transform(SqueezeTransform(array, axis))


def flatten(array):
    """Flatten array to 1D (C-order).

    Memory: ``io.write`` streams this via the reindex writer, hard-bounded to ~1 input
    chunk + 1 output chunk regardless of array size (verified flat: 170MB->679MB arrays
    all peak ~135MB). A lazy ``compute()`` still materializes.
    """
    return array._with_transform(FlattenTransform(array))


def pad(array, pad_width, mode="constant", **kwargs):
    """Pad ``array`` (like numpy.pad). ``mode`` is any numpy.pad mode (constant/reflect/edge/
    symmetric/wrap/...); extra kwargs (e.g. ``constant_values=``) pass through. Memory-bounded:
    interior reads touch only the core; border reads read the whole bordered axis."""
    return array._with_transform(PadTransform(array, pad_width, mode=mode, **kwargs))


def tile(array, reps):
    """Repeat array along dimensions."""
    return array._with_transform(TileTransform(array, reps))


def roll(array, shift, axis=None):
    """Roll array elements along an axis."""
    return array._with_transform(RollTransform(array, shift, axis))


def flip(array, axis):
    """Flip array along an axis."""
    return array._with_transform(FlipTransform(array, axis))


def rot90(array, k=1, axes=(0, 1)):
    """Rotate array by 90 degrees ``k`` times in the plane of ``axes`` (like numpy.rot90).
    Composed from the validated flip/transpose transforms, so it stays lazy + backend-agnostic."""
    ndim = array.ndim
    a0 = axes[0] if axes[0] >= 0 else ndim + axes[0]
    a1 = axes[1] if axes[1] >= 0 else ndim + axes[1]
    if a0 == a1 or not (0 <= a0 < ndim and 0 <= a1 < ndim):
        raise ValueError(f"invalid rotation axes {axes} for ndim {ndim}")
    k %= 4
    if k == 0:
        return array
    if k == 2:
        return flip(flip(array, a0), a1)
    perm = list(range(ndim))
    perm[a0], perm[a1] = perm[a1], perm[a0]
    if k == 1:
        return transpose(flip(array, a1), tuple(perm))
    return flip(transpose(array, tuple(perm)), a1)   # k == 3


__all__ = [
    "ConcatenateTransform", "StackTransform", "SliceTransform", "ExpandDimsTransform",
    "SwapAxesTransform", "TransposeTransform", "ReshapeTransform", "SqueezeTransform",
    "FlattenTransform", "PadTransform", "TileTransform", "RollTransform", "FlipTransform",
    "slice_array",
    "expand_dims", "concatenate", "stack", "swap_axes", "transpose", "reshape",
    "squeeze", "flatten", "pad", "tile", "roll", "flip", "rot90",
]
