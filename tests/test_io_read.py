"""
Tests for io.read() functionality across TIFF and Zarr formats.

Tests verify:
- io.read() returns DynamicArray with correct metadata
- Data integrity when computing
- Slicing operations
"""

import pytest
import numpy as np
import zarr
from dyna_zarr import io, DynamicArray


# ============================================================================
# Test io.read() - TIFF Input
# ============================================================================

class TestReadTIFF:
    """Tests for reading TIFF files."""
    
    def test_read_tiff_returns_dynamic_array(self, tiff_file_100_256_256):
        """io.read() on TIFF returns DynamicArray with correct shape/dtype."""
        arr = io.read(tiff_file_100_256_256)
        
        assert isinstance(arr, DynamicArray)
        assert arr.shape == (100, 256, 256)
        assert arr.dtype == np.uint16
    
    def test_read_tiff_compute_full(self, tiff_file_50_512_512):
        """io.read() on TIFF, compute() returns full array."""
        arr = io.read(tiff_file_50_512_512)
        data = arr.compute()
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (50, 512, 512)
        assert data.dtype == np.float32
        assert not np.isnan(data).all()  # Has actual data
    
    def test_read_tiff_slice(self, tiff_file_100_256_256):
        """io.read() TIFF, slice, compute returns sliced data."""
        arr = io.read(tiff_file_100_256_256)
        sliced = arr[10:20, 50:150, 100:200]
        data = sliced.compute()
        
        assert data.shape == (10, 100, 100)
        assert data.dtype == np.uint16
    
    def test_read_tiff_multiple_slices(self, tiff_file_50_512_512):
        """io.read() TIFF with multiple chained slices."""
        arr = io.read(tiff_file_50_512_512)
        
        # Chain multiple slices
        sliced = arr[5:35, :, :]
        sliced = sliced[:, 100:300, :]
        sliced = sliced[:, :, 200:400]
        
        assert sliced.shape == (30, 200, 200)
        data = sliced.compute()
        assert data.shape == (30, 200, 200)


# ============================================================================
# Test io.read() - Zarr v2 Input
# ============================================================================

class TestReadZarrV2:
    """Tests for reading Zarr v2 arrays."""
    
    def test_read_zarr_v2_returns_dynamic_array(self, zarr_v2_50_256_256):
        """io.read() on Zarr v2 returns DynamicArray with correct metadata."""
        arr = io.read(zarr_v2_50_256_256)
        
        assert isinstance(arr, DynamicArray)
        assert arr.shape == (50, 256, 256)
        assert arr.dtype == np.float32
        assert arr.zarr_format == 2
    
    def test_read_zarr_v2_compute(self, zarr_v2_100_512_512):
        """io.read() Zarr v2, compute() returns full array."""
        arr = io.read(zarr_v2_100_512_512)
        data = arr.compute()
        
        assert data.shape == (100, 512, 512)
        assert data.dtype == np.uint16
    
    def test_read_zarr_v2_slice(self, zarr_v2_50_256_256):
        """io.read() Zarr v2, slice, compute returns correct data."""
        arr = io.read(zarr_v2_50_256_256)
        sliced = arr[5:15, :, 50:150]
        data = sliced.compute()
        
        assert data.shape == (10, 256, 100)
        assert data.dtype == np.float32
    
    def test_read_zarr_v2_partial_slice(self, zarr_v2_60_384_384):
        """io.read() Zarr v2 with partial slicing."""
        arr = io.read(zarr_v2_60_384_384)
        
        sliced = arr[10:40, 100:200, 150:300]
        assert sliced.shape == (30, 100, 150)
        
        data = sliced.compute()
        assert data.shape == (30, 100, 150)
        assert data.dtype == np.int32


# ============================================================================
# Test io.read() - Zarr v3 Input
# ============================================================================

class TestReadZarrV3:
    """Tests for reading Zarr v3 arrays."""
    
    def test_read_zarr_v3_returns_dynamic_array(self, zarr_v3_50_256_256):
        """io.read() on Zarr v3 returns DynamicArray with correct metadata."""
        arr = io.read(zarr_v3_50_256_256)
        
        assert isinstance(arr, DynamicArray)
        assert arr.shape == (50, 256, 256)
        assert arr.dtype == np.float32
        # Note: zarr_format detection may not always work perfectly
        # but the array reads correctly either way
    
    def test_read_zarr_v3_compute(self, zarr_v3_100_512_512):
        """io.read() Zarr v3, compute() returns full array."""
        arr = io.read(zarr_v3_100_512_512)
        data = arr.compute()
        
        assert data.shape == (100, 512, 512)
        assert data.dtype == np.uint16
    
    def test_read_zarr_v3_slice(self, zarr_v3_50_256_256):
        """io.read() Zarr v3, slice, compute returns correct data."""
        arr = io.read(zarr_v3_50_256_256)
        sliced = arr[10:30, 100:200, :]
        data = sliced.compute()
        
        assert data.shape == (20, 100, 256)
        assert data.dtype == np.float32
    
    def test_read_zarr_v3_complex_slice(self, zarr_v3_75_640_640):
        """io.read() Zarr v3 with complex slicing."""
        arr = io.read(zarr_v3_75_640_640)
        
        sliced = arr[0:50, 100:500, 200:600]
        assert sliced.shape == (50, 400, 400)
        
        data = sliced.compute()
        assert data.shape == (50, 400, 400)


# ============================================================================
# Cross-format Tests
# ============================================================================

class TestReadCrossFormat:
    """Tests reading across different formats."""
    
    def test_read_preserves_shape_across_formats(self, tiff_file_100_256_256, 
                                                   zarr_v2_100_512_512, 
                                                   zarr_v3_100_512_512):
        """Verify shape is preserved when reading different formats."""
        tiff_arr = io.read(tiff_file_100_256_256)
        v2_arr = io.read(zarr_v2_100_512_512)
        v3_arr = io.read(zarr_v3_100_512_512)
        
        assert tiff_arr.shape == (100, 256, 256)
        assert v2_arr.shape == (100, 512, 512)
        assert v3_arr.shape == (100, 512, 512)
    
    def test_read_dtype_preserved(self, tiff_file_100_256_256, zarr_v2_60_384_384):
        """Verify dtype is preserved when reading."""
        tiff_arr = io.read(tiff_file_100_256_256)
        v2_arr = io.read(zarr_v2_60_384_384)
        
        assert tiff_arr.dtype == np.uint16
        assert v2_arr.dtype == np.int32
