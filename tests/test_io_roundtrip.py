"""
Tests for roundtrip operations (read → write → read → verify).

Tests verify data integrity through format conversions and multi-step pipelines.
"""

import pytest
import numpy as np
import zarr
from dyna_zarr import io


# ============================================================================
# Test Roundtrips
# ============================================================================

class TestRoundtrips:
    """Tests reading, writing, and re-reading data."""
    
    def test_roundtrip_tiff_to_v2_to_v3(self, tiff_file_50_512_512, tmp_path):
        """Read TIFF, write to Zarr v2, read, write to Zarr v3, verify."""
        # Read TIFF
        arr1 = io.read(tiff_file_50_512_512)
        data1 = arr1.compute()
        
        # Write to Zarr v2
        v2_path = str(tmp_path / "step1.zarr")
        io.write(arr1, v2_path, zarr_format=2)
        
        # Read from Zarr v2
        arr2 = io.read(v2_path)
        data2 = arr2.compute()
        
        # Write to Zarr v3
        v3_path = str(tmp_path / "step2.zarr")
        io.write(arr2, v3_path, zarr_format=3)
        
        # Read from Zarr v3
        arr3 = io.read(v3_path)
        data3 = arr3.compute()
        
        # Verify data integrity through pipeline
        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(data2, data3)
    
    def test_roundtrip_v2_to_v3_to_v2(self, zarr_v2_100_512_512, tmp_path):
        """Read Zarr v2, write v3, read v3, write v2, verify data integrity."""
        arr1 = io.read(zarr_v2_100_512_512)
        data1 = arr1.compute()
        
        v3_path = str(tmp_path / "temp_v3.zarr")
        io.write(arr1, v3_path, zarr_format=3)
        
        arr2 = io.read(v3_path)
        
        v2_path = str(tmp_path / "roundtrip_v2.zarr")
        io.write(arr2, v2_path, zarr_format=2)
        
        arr3 = io.read(v2_path)
        data3 = arr3.compute()
        
        np.testing.assert_array_equal(data1, data3)
    
    def test_roundtrip_v3_to_v2(self, zarr_v3_75_640_640, tmp_path):
        """Read Zarr v3, write to Zarr v2, read back, verify."""
        arr1 = io.read(zarr_v3_75_640_640)
        data1 = arr1.compute()
        
        v2_path = str(tmp_path / "from_v3.zarr")
        io.write(arr1, v2_path, zarr_format=2)
        
        arr2 = io.read(v2_path)
        data2 = arr2.compute()
        
        np.testing.assert_allclose(data1, data2, rtol=1e-5)
    
    def test_roundtrip_with_slicing(self, tiff_file_100_256_256, tmp_path):
        """Read TIFF, slice, write, read, verify slice matches."""
        arr = io.read(tiff_file_100_256_256)
        
        # Extract specific region
        region = arr[20:80, 50:200, 100:200]
        region_data = region.compute()
        
        # Write sliced region to Zarr v2
        v2_path = str(tmp_path / "sliced_v2.zarr")
        io.write(region, v2_path, zarr_format=2)
        
        # Read back and verify
        arr2 = io.read(v2_path)
        data2 = arr2.compute()
        
        np.testing.assert_array_equal(region_data, data2)
    
    def test_roundtrip_multiple_formats(self, zarr_v2_50_256_256, tmp_path):
        """Test TIFF → v2 → v3 → v2 chain."""
        arr1 = io.read(zarr_v2_50_256_256)
        data1 = arr1.compute()
        
        # v2 → v3
        v3_path = str(tmp_path / "inter_v3.zarr")
        io.write(arr1, v3_path, zarr_format=3)
        arr2 = io.read(v3_path)
        
        # v3 → v2
        v2_path = str(tmp_path / "final_v2.zarr")
        io.write(arr2, v2_path, zarr_format=2)
        arr3 = io.read(v2_path)
        data3 = arr3.compute()
        
        # Verify
        np.testing.assert_allclose(data1, data3, rtol=1e-5)
    
    def test_roundtrip_with_dtype_conversion(self, tiff_file_100_256_256, tmp_path):
        """Read uint16 TIFF, convert to float32, write, read, verify dtype."""
        arr1 = io.read(tiff_file_100_256_256)  # uint16
        assert arr1.dtype == np.uint16
        
        # Write as float32 to Zarr v3
        v3_path = str(tmp_path / "float_v3.zarr")
        io.write(arr1, v3_path, dtype=np.float32, zarr_format=3)
        
        # Read back
        arr2 = io.read(v3_path)
        assert arr2.dtype == np.float32
        
        # Write back to uint16
        v2_path = str(tmp_path / "back_uint16_v2.zarr")
        io.write(arr2, v2_path, dtype=np.uint16, zarr_format=2)
        
        # Verify
        arr3 = io.read(v2_path)
        assert arr3.dtype == np.uint16


# ============================================================================
# Test Sharding Roundtrips
# ============================================================================

class TestShardingRoundtrips:
    """Tests roundtrips with Zarr v3 sharding."""
    
    def test_roundtrip_sharded_v3_to_v2(self, zarr_v3_50_256_256, tmp_path):
        """Write sharded Zarr v3, read, write to v2 (shards stripped)."""
        arr1 = io.read(zarr_v3_50_256_256)
        data1 = arr1.compute()
        
        # Write as sharded v3
        sharded_path = str(tmp_path / "sharded.zarr")
        io.write(
            arr1,
            sharded_path,
            chunks=(16, 64, 64),
            shard_coefficients=(2, 2, 2),
            zarr_format=3
        )
        
        # Read sharded v3
        arr2 = io.read(sharded_path)
        
        # Write to v2 (loses sharding)
        v2_path = str(tmp_path / "from_sharded.zarr")
        io.write(arr2, v2_path, zarr_format=2)
        
        # Verify data integrity
        arr3 = io.read(v2_path)
        data3 = arr3.compute()
        
        np.testing.assert_allclose(data1, data3, rtol=1e-5)
    
    def test_roundtrip_v2_to_sharded_v3(self, zarr_v2_100_512_512, tmp_path):
        """Write v2 to sharded v3, read back, verify."""
        arr1 = io.read(zarr_v2_100_512_512)
        data1 = arr1.compute()
        
        # Write as sharded v3
        sharded_path = str(tmp_path / "v2_as_sharded.zarr")
        io.write(
            arr1,
            sharded_path,
            chunks=(25, 128, 128),
            shard_coefficients=(2, 2, 2),
            zarr_format=3
        )
        
        # Read and verify
        arr2 = io.read(sharded_path)
        data2 = arr2.compute()
        
        np.testing.assert_array_equal(data1, data2)


# ============================================================================
# Test Data Integrity Through Pipelines
# ============================================================================

class TestDataIntegrity:
    """Tests that verify data is not corrupted through operations."""
    
    def test_roundtrip_preserves_values(self, tiff_file_50_512_512, tmp_path):
        """Verify actual values (not just shape) through roundtrip."""
        arr1 = io.read(tiff_file_50_512_512)
        data1 = arr1.compute()
        
        # Write and read back
        v2_path = str(tmp_path / "integrity_v2.zarr")
        io.write(arr1, v2_path, zarr_format=2)
        arr2 = io.read(v2_path)
        data2 = arr2.compute()
        
        # Check specific values
        np.testing.assert_allclose(data1, data2, rtol=1e-5)
        assert data1.min() == data2.min() or np.isclose(data1.min(), data2.min())
        assert data1.max() == data2.max() or np.isclose(data1.max(), data2.max())
    
    def test_roundtrip_with_sliced_region(self, zarr_v3_100_512_512, tmp_path):
        """Verify sliced region maintains values."""
        arr = io.read(zarr_v3_100_512_512)
        
        # Extract region
        region = arr[10:50, 100:300, 200:400]
        region_data = region.compute()
        
        # Write region
        v3_path = str(tmp_path / "region_v3.zarr")
        io.write(region, v3_path, zarr_format=3)
        
        # Read and compare
        arr2 = io.read(v3_path)
        data2 = arr2.compute()
        
        np.testing.assert_array_equal(region_data, data2)
