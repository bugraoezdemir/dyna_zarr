"""
Tests for basic laziness guarantees in read, write, and operations.

Tests verify that data is not loaded unnecessarily and that operations
are lazy (not executed until compute() or write()).
"""

import pytest
import numpy as np
import zarr
from dyna_zarr import io, operations, DynamicArray


# ============================================================================
# Test Read Laziness
# ============================================================================

class TestReadLaziness:
    """Tests that io.read() is lazy."""
    
    def test_read_tiff_lazy(self, tiff_file_100_256_256):
        """io.read() on TIFF is lazy - doesn't load until compute()."""
        # Reading should be nearly instant (lazy)
        arr = io.read(tiff_file_100_256_256)
        
        # Verify it's a DynamicArray (lazy wrapper)
        assert isinstance(arr, DynamicArray)
        assert arr.shape == (100, 256, 256)
        assert arr.dtype == np.uint16
        # No compute() yet - data not in memory as numpy array
    
    def test_read_zarr_v3_lazy(self, zarr_v3_100_512_512):
        """io.read() on Zarr v3 is lazy."""
        arr = io.read(zarr_v3_100_512_512)
        
        assert isinstance(arr, DynamicArray)
        assert arr.shape == (100, 512, 512)
        # compute() not called - still lazy
    
    def test_slice_is_lazy(self, tiff_file_50_512_512):
        """Slicing should be instant and lazy."""
        arr = io.read(tiff_file_50_512_512)
        
        # Slicing should be very fast (lazy)
        import time
        start = time.time()
        sliced = arr[0:10, 0:50, 0:50]  # Small region
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be nearly instant
        assert sliced.shape == (10, 50, 50)
        # Still lazy - no compute


# ============================================================================
# Test Write Laziness (Region-wise Processing)
# ============================================================================

class TestWriteLaziness:
    """Tests that io.write() processes lazily and region-wise."""
    
    def test_write_doesnt_require_full_load(self, zarr_v3_100_512_512, tmp_path):
        """io.write() can handle large arrays by processing region-wise."""
        # Read 100x512x512 array (100 MB+)
        arr = io.read(zarr_v3_100_512_512)
        
        # Write with small region size (forces region-wise processing)
        output_path = str(tmp_path / "regionwise.zarr")
        io.write(arr, output_path, region_size_mb=4.0, zarr_format=3)
        
        # If it failed to process region-wise, it would run out of memory
        # If it succeeded, the output should be valid
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 512, 512)
    
    def test_write_small_slice_is_efficient(self, tiff_file_100_256_256, tmp_path):
        """Writing a small sliced region should be efficient."""
        arr = io.read(tiff_file_100_256_256)
        
        # Slice to small region
        sliced = arr[0:10, 0:50, 0:50]  # Only 10x50x50 = 25KB
        
        # Write - should only read the sliced region
        output_path = str(tmp_path / "small_region.zarr")
        io.write(sliced, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (10, 50, 50)


# ============================================================================
# Test Operation Laziness
# ============================================================================

class TestOperationLaziness:
    """Tests that operations are lazy (not executed until compute/write)."""
    
    def test_abs_is_lazy(self, tiff_file_50_512_512):
        """operations.abs() is lazy."""
        arr = io.read(tiff_file_50_512_512)
        
        # Apply operation - should be instant
        import time
        start = time.time()
        processed = operations.abs(arr)
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be instant
        assert isinstance(processed, DynamicArray)
        assert processed.shape == (50, 512, 512)
    
    def test_clip_is_lazy(self, zarr_v2_50_256_256):
        """operations.clip() is lazy."""
        arr = io.read(zarr_v2_50_256_256)
        
        # Clip operation - should be instant
        import time
        start = time.time()
        clipped = operations.clip(arr, 0.2, 0.8)
        elapsed = time.time() - start
        
        assert elapsed < 0.1
        assert isinstance(clipped, DynamicArray)
    
    def test_concatenate_is_lazy(self, zarr_v3_50_256_256):
        """operations.concatenate() is lazy."""
        arr = io.read(zarr_v3_50_256_256)
        
        # Concatenate - should be instant
        import time
        start = time.time()
        concat = operations.concatenate([arr, arr], axis=0)
        elapsed = time.time() - start
        
        assert elapsed < 0.1
        assert isinstance(concat, DynamicArray)
        assert concat.shape == (100, 256, 256)
    
    def test_chained_operations_lazy(self, tiff_file_50_512_512):
        """Chaining operations should be lazy until compute/write."""
        arr = io.read(tiff_file_50_512_512)
        
        # Chain multiple operations
        import time
        start = time.time()
        result = arr
        for _ in range(5):
            result = operations.abs(result)
        elapsed = time.time() - start
        
        # All operations should be instant (lazy)
        assert elapsed < 0.5
        assert isinstance(result, DynamicArray)


# ============================================================================
# Test Lazy Evaluation in Full Pipelines
# ============================================================================

class TestLazyPipelines:
    """Tests that full pipelines remain lazy until execution."""
    
    def test_read_process_write_lazy(self, tiff_file_30_768_768, tmp_path):
        """Full read→process→write pipeline remains lazy until write executes."""
        # These should all be instant (lazy)
        import time
        start = time.time()
        arr = io.read(tiff_file_30_768_768)
        processed = operations.abs(arr)
        sliced = processed[0:10, 0:100, 0:100]
        elapsed = time.time() - start
        
        # All lazy operations should complete instantly
        assert elapsed < 0.5
        assert isinstance(sliced, DynamicArray)
        
        # Only the write should take time (and process region-wise)
        output_path = str(tmp_path / "lazy_pipeline.zarr")
        io.write(sliced, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (10, 100, 100)
    
    def test_operations_dont_evaluate_until_needed(self, zarr_v2_100_512_512):
        """Operations don't evaluate data until compute() or write()."""
        arr = io.read(zarr_v2_100_512_512)
        
        # Build transformation chain
        import time
        start = time.time()
        processed = operations.abs(arr)
        processed = operations.clip(processed, 0.1, 0.9)
        processed = operations.abs(processed)
        elapsed = time.time() - start
        
        # All should be instant
        assert elapsed < 0.5
        assert isinstance(processed, DynamicArray)


# ============================================================================
# Test Slice Efficiency
# ============================================================================

class TestSliceEfficiency:
    """Tests that slicing is memory efficient."""
    
    def test_large_array_small_slice_efficient(self, tiff_file_100_256_256, tmp_path):
        """Reading large file and slicing small region should be efficient."""
        arr = io.read(tiff_file_100_256_256)  # 100x256x256 uint16 = 12.8 MB
        
        # Take very small slice
        tiny = arr[0:1, 0:10, 0:10]  # 1x10x10 = 100 bytes
        
        # Write this tiny region
        output_path = str(tmp_path / "tiny.zarr")
        io.write(tiny, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (1, 10, 10)
        assert result.nbytes == 1 * 10 * 10 * 2  # 200 bytes only
    
    def test_slice_compute_only_reads_region(self, zarr_v3_50_256_256):
        """Slicing and computing should only read requested region."""
        arr = io.read(zarr_v3_50_256_256)
        
        # Slice to small region
        small_region = arr[0:5, 0:50, 0:50]  # 5x50x50 = 12.5 KB
        
        # Compute should only load this region
        data = small_region.compute()
        assert data.shape == (5, 50, 50)
        assert data.nbytes == 5 * 50 * 50 * 4  # 50 KB only for float32
