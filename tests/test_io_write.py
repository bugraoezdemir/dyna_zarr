"""
Tests for io.write() functionality to Zarr v2 and v3.

Tests verify:
- io.write() creates valid Zarr arrays
- Metadata (shape, dtype, chunks) is preserved
- Custom parameters (chunks, compression, dtype conversion)
- Both Zarr v2 and v3 output formats
"""

import pytest
import numpy as np
import zarr
from pathlib import Path
from dyna_zarr import io, Codecs


# ============================================================================
# Test io.write() - Write to Zarr v2
# ============================================================================

class TestWriteZarrV2:
    """Tests for writing to Zarr v2 format."""
    
    def test_write_tiff_to_zarr_v2_basic(self, tiff_file_50_512_512, tmp_path):
        """io.write() creates valid Zarr v2 array with correct metadata."""
        arr = io.read(tiff_file_50_512_512)
        
        output_path = str(tmp_path / "output_v2.zarr")
        io.write(arr, output_path, zarr_format=2)
        
        # Verify output is valid Zarr v2
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 512, 512)
        assert result.dtype == np.float32
        assert (Path(output_path) / ".zarray").exists()
    
    def test_write_zarr_v3_to_zarr_v2(self, zarr_v3_50_256_256, tmp_path):
        """io.write() Zarr v3 to Zarr v2 preserves data."""
        arr = io.read(zarr_v3_50_256_256)
        
        output_path = str(tmp_path / "v3_to_v2.zarr")
        io.write(arr, output_path, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 256, 256)
        assert result.dtype == np.float32
    
    def test_write_to_zarr_v2_custom_chunks(self, zarr_v3_100_512_512, tmp_path):
        """io.write() respects custom chunk size for Zarr v2."""
        arr = io.read(zarr_v3_100_512_512)
        
        output_path = str(tmp_path / "custom_chunks_v2.zarr")
        io.write(arr, output_path, chunks=(25, 128, 128), zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 512, 512)
        assert result.chunks == (25, 128, 128)
    
    def test_write_to_zarr_v2_with_compression(self, tiff_file_100_256_256, tmp_path):
        """io.write() applies compression to Zarr v2."""
        arr = io.read(tiff_file_100_256_256)
        
        output_path = str(tmp_path / "compressed_v2.zarr")
        codecs = Codecs(compressor='zstd', clevel=3)
        io.write(arr, output_path, compressor=codecs, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 256, 256)
        assert result.compressor is not None
    
    def test_write_to_zarr_v2_dtype_conversion(self, tiff_file_50_512_512, tmp_path):
        """io.write() converts dtype if specified."""
        arr = io.read(tiff_file_50_512_512)  # float32
        
        output_path = str(tmp_path / "dtype_v2.zarr")
        io.write(arr, output_path, dtype=np.float64, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.dtype == np.float64
        assert result.shape == (50, 512, 512)
    
    def test_write_to_zarr_v2_uint8(self, zarr_v2_100_512_512, tmp_path):
        """io.write() uint16 to uint8 conversion."""
        arr = io.read(zarr_v2_100_512_512)  # uint16
        
        output_path = str(tmp_path / "uint8_v2.zarr")
        io.write(arr, output_path, dtype=np.uint8, zarr_format=2)
        
        result = zarr.open(output_path, mode='r')
        assert result.dtype == np.uint8
        assert result.shape == (100, 512, 512)


# ============================================================================
# Test io.write() - Write to Zarr v3
# ============================================================================

class TestWriteZarrV3:
    """Tests for writing to Zarr v3 format."""
    
    def test_write_tiff_to_zarr_v3_basic(self, tiff_file_50_512_512, tmp_path):
        """io.write() creates valid Zarr v3 array with correct metadata."""
        arr = io.read(tiff_file_50_512_512)
        
        output_path = str(tmp_path / "output_v3.zarr")
        io.write(arr, output_path, zarr_format=3)
        
        # Verify output is valid Zarr v3
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 512, 512)
        assert result.dtype == np.float32
        assert (Path(output_path) / "zarr.json").exists()
    
    def test_write_zarr_v2_to_zarr_v3(self, zarr_v2_50_256_256, tmp_path):
        """io.write() Zarr v2 to Zarr v3 preserves data."""
        arr = io.read(zarr_v2_50_256_256)
        
        output_path = str(tmp_path / "v2_to_v3.zarr")
        io.write(arr, output_path, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 256, 256)
        assert result.dtype == np.float32
    
    def test_write_to_zarr_v3_custom_chunks(self, tiff_file_100_256_256, tmp_path):
        """io.write() respects custom chunk size for Zarr v3."""
        arr = io.read(tiff_file_100_256_256)
        
        output_path = str(tmp_path / "custom_chunks_v3.zarr")
        io.write(arr, output_path, chunks=(20, 64, 128), zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 256, 256)
        assert result.chunks == (20, 64, 128)
    
    def test_write_to_zarr_v3_with_sharding(self, zarr_v3_50_256_256, tmp_path):
        """io.write() creates sharded Zarr v3 when shard_coefficients specified."""
        arr = io.read(zarr_v3_50_256_256)
        
        output_path = str(tmp_path / "sharded_v3.zarr")
        io.write(
            arr, 
            output_path, 
            chunks=(16, 64, 64),
            shard_coefficients=(2, 2, 2),
            zarr_format=3
        )
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (50, 256, 256)
        
        # Verify sharding in metadata
        import json
        with open(Path(output_path) / "zarr.json") as f:
            metadata = json.load(f)
        assert 'codecs' in metadata
    
    def test_write_to_zarr_v3_with_compression(self, tiff_file_100_256_256, tmp_path):
        """io.write() applies compression to Zarr v3."""
        arr = io.read(tiff_file_100_256_256)
        
        output_path = str(tmp_path / "compressed_v3.zarr")
        codecs = Codecs(compressor='blosc', clevel=5, cname='zstd')
        io.write(arr, output_path, compressor=codecs, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.shape == (100, 256, 256)
        assert result.dtype == np.uint16
    
    def test_write_to_zarr_v3_dtype_conversion(self, zarr_v2_60_384_384, tmp_path):
        """io.write() int32 to float32 conversion for Zarr v3."""
        arr = io.read(zarr_v2_60_384_384)  # int32
        
        output_path = str(tmp_path / "dtype_v3.zarr")
        io.write(arr, output_path, dtype=np.float32, zarr_format=3)
        
        result = zarr.open(output_path, mode='r')
        assert result.dtype == np.float32
        assert result.shape == (60, 384, 384)


# ============================================================================
# Cross-format Write Tests
# ============================================================================

class TestWriteCrossFormat:
    """Tests writing across different format combinations."""
    
    def test_write_preserves_shape(self, tiff_file_50_512_512, tmp_path):
        """Verify shape is preserved when writing to both formats."""
        arr = io.read(tiff_file_50_512_512)
        
        v2_path = str(tmp_path / "v2.zarr")
        v3_path = str(tmp_path / "v3.zarr")
        
        io.write(arr, v2_path, zarr_format=2)
        io.write(arr, v3_path, zarr_format=3)
        
        v2_result = zarr.open(v2_path, mode='r')
        v3_result = zarr.open(v3_path, mode='r')
        
        assert v2_result.shape == (50, 512, 512)
        assert v3_result.shape == (50, 512, 512)
    
    def test_write_with_different_compressions(self, zarr_v2_50_256_256, tmp_path):
        """Verify different compression codecs work."""
        arr = io.read(zarr_v2_50_256_256)
        
        compressions = [
            ('blosc', {'clevel': 5, 'cname': 'lz4'}),
            ('zstd', {'clevel': 3}),
        ]
        
        for comp_name, comp_kwargs in compressions:
            output_path = str(tmp_path / f"compressed_{comp_name}.zarr")
            codecs = Codecs(compressor=comp_name, **comp_kwargs)
            io.write(arr, output_path, compressor=codecs, zarr_format=2)
            
            result = zarr.open(output_path, mode='r')
            assert result.shape == (50, 256, 256)
