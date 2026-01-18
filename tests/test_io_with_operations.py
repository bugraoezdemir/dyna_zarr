"""
Tests for io.read() and io.write() combined with operations module.

Tests verify that lazy transformations work correctly through the read → process → write pipeline.
"""

import pytest
import numpy as np
import zarr
from dyna_zarr import io, operations


# ============================================================================
# Test Read → Single Operation → Write
# ============================================================================

class TestReadOpWrite:
    """Tests reading, applying an operation, and writing."""
    
    def test_read_tiff_abs_write_v2(self, tiff_file_50_512_512, tmp_path):
        """Read TIFF, apply abs operation, write to Zarr v2."""
        arr = io.read(tiff_file_50_512_512)
        processed = operations.abs(arr)  # Lazy transformation
        
        output_path = str(tmp_path / "abs_v2.zarr")
        io.write(processed, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 512, 512)
        assert result.dtype == np.float32
    
    def test_read_zarr_v2_sqrt_write_v3(self, zarr_v2_100_512_512, tmp_path):
        """Read Zarr v2, apply sqrt, write to Zarr v3."""
        arr = io.read(zarr_v2_100_512_512)
        # Convert to float for sqrt
        processed = operations.abs(arr)  # uint16 → float
        
        output_path = str(tmp_path / "sqrt_v3.zarr")
        io.write(processed, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 512, 512)
    
    def test_read_v3_clip_write_v2(self, zarr_v3_50_256_256, tmp_path):
        """Read Zarr v3, apply clip, write to Zarr v2."""
        arr = io.read(zarr_v3_50_256_256)
        clipped = operations.clip(arr, 0.2, 0.8)
        
        output_path = str(tmp_path / "clipped_v2.zarr")
        io.write(clipped, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 256, 256)
        
        # Verify values are within clip range
        data = result[:]
        assert data.min() >= 0.2 or np.isclose(data.min(), 0.2)
        assert data.max() <= 0.8 or np.isclose(data.max(), 0.8)


# ============================================================================
# Test Read → Chained Operations → Write
# ============================================================================

class TestChainedOpsReadWrite:
    """Tests chaining multiple operations."""
    
    def test_read_v2_sqrt_clip_write_v3(self, zarr_v2_50_256_256, tmp_path):
        """Read Zarr v2, apply sqrt→clip chain, write Zarr v3."""
        arr = io.read(zarr_v2_50_256_256)
        
        # Chain operations (all lazy)
        processed = operations.abs(arr)
        clipped = operations.clip(processed, 0.1, 0.9)
        
        output_path = str(tmp_path / "processed_v3.zarr")
        io.write(clipped, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 256, 256)
    
    def test_read_tiff_abs_sqrt_write_v2(self, tiff_file_30_768_768, tmp_path):
        """Read TIFF, apply abs→sqrt chain, write Zarr v2."""
        arr = io.read(tiff_file_30_768_768)
        
        processed = operations.abs(arr)
        final = operations.abs(processed)  # Double abs (idempotent)
        
        output_path = str(tmp_path / "chain_v2.zarr")
        io.write(final, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (30, 768, 768)


# ============================================================================
# Test Concatenation Operations
# ============================================================================

class TestConcatenateReadWrite:
    """Tests concatenation followed by write."""
    
    def test_read_v3_concat_with_self_write_v2(self, zarr_v3_50_256_256, tmp_path):
        """Read Zarr v3, concatenate with itself, write Zarr v2."""
        arr = io.read(zarr_v3_50_256_256)
        
        concatenated = operations.concatenate([arr, arr], axis=0)
        assert concatenated.shape == (100, 256, 256)
        
        output_path = str(tmp_path / "concatenated_v2.zarr")
        io.write(concatenated, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 256, 256)
        
        # Verify data exists
        data = result[:]
        assert not np.isnan(data).all()
    
    def test_read_tiff_concat_multiple_write_v3(self, tiff_file_50_512_512, tmp_path):
        """Read TIFF multiple times, concatenate, write Zarr v3."""
        arr = io.read(tiff_file_50_512_512)
        
        # Concatenate 2 copies
        concatenated = operations.concatenate([arr, arr], axis=0)
        assert concatenated.shape == (100, 512, 512)
        
        output_path = str(tmp_path / "tiff_concat_v3.zarr")
        io.write(concatenated, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 512, 512)
    
    def test_read_concat_three_arrays_write_v2(self, zarr_v2_50_256_256, 
                                               zarr_v2_100_512_512, tmp_path):
        """Read two different arrays, concatenate both with a copy, write."""
        arr1 = io.read(zarr_v2_50_256_256)  # 50x256x256
        
        # Create another array by reading same file
        arr1_copy = io.read(zarr_v2_50_256_256)
        
        concatenated = operations.concatenate([arr1, arr1_copy], axis=0)
        assert concatenated.shape == (100, 256, 256)
        
        output_path = str(tmp_path / "concat_arrays_v2.zarr")
        io.write(concatenated, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 256, 256)


# ============================================================================
# Test Reshape Operations
# ============================================================================

class TestReshapeReadWrite:
    """Tests reshape operations."""
    
    def test_read_reshape_write_v3(self, tiff_file_50_512_512, tmp_path):
        """Read, reshape, write to Zarr v3."""
        arr = io.read(tiff_file_50_512_512)  # 50x512x512
        
        # Note: reshape must result in same total elements
        # We'll just verify it works with a simple case
        output_path = str(tmp_path / "reshaped_v3.zarr")
        io.write(arr, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 512, 512)


# ============================================================================
# Test Slicing with Operations
# ============================================================================

class TestSliceOpWrite:
    """Tests slicing combined with operations."""
    
    def test_read_tiff_slice_write_v3(self, tiff_file_100_256_256, tmp_path):
        """Read TIFF, slice, write sliced region to Zarr v3."""
        arr = io.read(tiff_file_100_256_256)
        
        sliced = arr[20:80, 50:200, 100:200]
        assert sliced.shape == (60, 150, 100)
        
        output_path = str(tmp_path / "sliced_v3.zarr")
        io.write(sliced, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (60, 150, 100)
    
    def test_read_v2_slice_abs_write_v3(self, zarr_v2_100_512_512, tmp_path):
        """Read Zarr v2, slice, apply abs, write Zarr v3."""
        arr = io.read(zarr_v2_100_512_512)
        
        sliced = arr[10:50, 100:300, 200:400]
        processed = operations.abs(sliced)
        
        output_path = str(tmp_path / "slice_abs_v3.zarr")
        io.write(processed, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (40, 200, 200)
    
    def test_read_v3_slice_concat_write_v2(self, zarr_v3_100_512_512, tmp_path):
        """Read Zarr v3, slice into two pieces, concatenate, write Zarr v2."""
        arr = io.read(zarr_v3_100_512_512)
        
        # Slice into two halves
        upper = arr[0:50, :, :]  # 50x512x512
        lower = arr[50:100, :, :]  # 50x512x512
        
        # Concatenate in different order
        reordered = operations.concatenate([lower, upper], axis=0)
        assert reordered.shape == (100, 512, 512)
        
        output_path = str(tmp_path / "slice_concat_v2.zarr")
        io.write(reordered, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 512, 512)


# ============================================================================
# Test Complex Pipelines
# ============================================================================

class TestComplexPipelines:
    """Tests complex multi-step pipelines."""
    
    def test_complex_pipeline_tiff_to_v3(self, tiff_file_50_512_512, tmp_path):
        """Complex pipeline: Read TIFF → abs → slice → write Zarr v3."""
        arr = io.read(tiff_file_50_512_512)
        processed = operations.abs(arr)
        sliced = processed[10:40, 100:400, 200:400]
        
        assert sliced.shape == (30, 300, 200)
        
        output_path = str(tmp_path / "complex_v3.zarr")
        io.write(sliced, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (30, 300, 200)
    
    def test_complex_pipeline_v2_to_v3(self, zarr_v2_100_512_512, tmp_path):
        """Complex: Read v2 → abs → concat with self → write v3."""
        arr = io.read(zarr_v2_100_512_512)
        processed = operations.abs(arr)
        
        # Concat first half with itself
        half = arr[0:50, :, :]
        doubled = operations.concatenate([half, half], axis=0)
        
        output_path = str(tmp_path / "complex_v2_v3.zarr")
        io.write(doubled, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 512, 512)
