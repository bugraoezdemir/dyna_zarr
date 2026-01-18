"""
Shared test fixtures for dyna-zarr tests.

Creates on-the-fly test data (TIFF and Zarr arrays) with automatic cleanup.
"""

import pytest
import numpy as np
import zarr
import tifffile
from pathlib import Path


# ============================================================================
# TIFF Fixtures
# ============================================================================

@pytest.fixture
def tiff_file_100_256_256(tmp_path):
    """Create 100x256x256 uint16 TIFF file (~12.8 MB)."""
    tiff_path = tmp_path / "test_100_256_256.tiff"
    # Create test data
    data = np.random.randint(0, 65535, size=(100, 256, 256), dtype=np.uint16)
    # Write TIFF
    tifffile.imwrite(str(tiff_path), data)
    yield str(tiff_path)
    # Cleanup (automatic via tmp_path)


@pytest.fixture
def tiff_file_50_512_512(tmp_path):
    """Create 50x512x512 float32 TIFF file (~51.2 MB)."""
    tiff_path = tmp_path / "test_50_512_512.tiff"
    data = np.random.rand(50, 512, 512).astype(np.float32)
    tifffile.imwrite(str(tiff_path), data)
    yield str(tiff_path)


@pytest.fixture
def tiff_file_30_768_768(tmp_path):
    """Create 30x768x768 float32 TIFF file (~70.2 MB)."""
    tiff_path = tmp_path / "test_30_768_768.tiff"
    data = np.random.rand(30, 768, 768).astype(np.float32)
    tifffile.imwrite(str(tiff_path), data)
    yield str(tiff_path)


# ============================================================================
# Zarr v2 Fixtures
# ============================================================================

@pytest.fixture
def zarr_v2_50_256_256(tmp_path):
    """Create Zarr v2 array (50, 256, 256) float32."""
    zarr_path = tmp_path / "zarr_v2_50_256_256.zarr"
    data = np.random.rand(50, 256, 256).astype(np.float32)
    z = zarr.open(str(zarr_path), mode='w', zarr_format=2, shape=(50, 256, 256), dtype=np.float32)
    z[:] = data
    yield str(zarr_path)


@pytest.fixture
def zarr_v2_100_512_512(tmp_path):
    """Create Zarr v2 array (100, 512, 512) uint16."""
    zarr_path = tmp_path / "zarr_v2_100_512_512.zarr"
    data = np.random.randint(0, 65535, size=(100, 512, 512), dtype=np.uint16)
    z = zarr.open(str(zarr_path), mode='w', zarr_format=2, shape=(100, 512, 512), dtype=np.uint16)
    z[:] = data
    yield str(zarr_path)


@pytest.fixture
def zarr_v2_60_384_384(tmp_path):
    """Create Zarr v2 array (60, 384, 384) int32."""
    zarr_path = tmp_path / "zarr_v2_60_384_384.zarr"
    data = np.random.randint(-100, 100, size=(60, 384, 384), dtype=np.int32)
    z = zarr.open(str(zarr_path), mode='w', zarr_format=2, shape=(60, 384, 384), dtype=np.int32)
    z[:] = data
    yield str(zarr_path)


# ============================================================================
# Zarr v3 Fixtures
# ============================================================================

@pytest.fixture
def zarr_v3_50_256_256(tmp_path):
    """Create Zarr v3 array (50, 256, 256) float32."""
    zarr_path = tmp_path / "zarr_v3_50_256_256.zarr"
    data = np.random.rand(50, 256, 256).astype(np.float32)
    z = zarr.open(str(zarr_path), mode='w', zarr_format=3, shape=(50, 256, 256), dtype=np.float32)
    z[:] = data
    yield str(zarr_path)


@pytest.fixture
def zarr_v3_100_512_512(tmp_path):
    """Create Zarr v3 array (100, 512, 512) uint16."""
    zarr_path = tmp_path / "zarr_v3_100_512_512.zarr"
    data = np.random.randint(0, 65535, size=(100, 512, 512), dtype=np.uint16)
    z = zarr.open(str(zarr_path), mode='w', zarr_format=3, shape=(100, 512, 512), dtype=np.uint16)
    z[:] = data
    yield str(zarr_path)


@pytest.fixture
def zarr_v3_75_640_640(tmp_path):
    """Create Zarr v3 array (75, 640, 640) float32."""
    zarr_path = tmp_path / "zarr_v3_75_640_640.zarr"
    data = np.random.rand(75, 640, 640).astype(np.float32)
    z = zarr.open(str(zarr_path), mode='w', zarr_format=3, shape=(75, 640, 640), dtype=np.float32)
    z[:] = data
    yield str(zarr_path)


# ============================================================================
# Helper Functions
# ============================================================================

def assert_arrays_equal(arr1, arr2, rtol=1e-5):
    """Compare two arrays with tolerance for floating point errors."""
    assert arr1.shape == arr2.shape, f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
    assert arr1.dtype == arr2.dtype, f"Dtype mismatch: {arr1.dtype} vs {arr2.dtype}"
    
    if np.issubdtype(arr1.dtype, np.floating):
        # Use allclose for floating point
        np.testing.assert_allclose(arr1, arr2, rtol=rtol)
    else:
        # Exact equality for integers
        np.testing.assert_array_equal(arr1, arr2)
