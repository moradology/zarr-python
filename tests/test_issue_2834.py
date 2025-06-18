"""Regression test for issue #2834: oindex with sharding bug"""
import numpy as np
import pytest
import zarr


def test_oindex_with_sharding():
    """Test case from issue #2834 - oindex with sharding should work"""
    group = zarr.group(store={}, zarr_format=3)
    
    array = group.create_array(
        name="zoo",
        shape=(1, 2, 1),
        chunks=(1, 2, 1),
        shards=(1, 2, 1),
        dtype=np.int32,
    )
    
    zindexer = (np.array([0]), np.array([0, 0]), np.array([0]))
    new_data = np.full(array.oindex[zindexer].shape, fill_value=1)
    
    # This was raising ValueError before the fix
    array.oindex[zindexer] = new_data
    
    # Verify the assignment worked
    # Note: since we're indexing [0,0,0] twice, the value should be 1
    assert array[0, 0, 0] == 1
    

def test_oindex_with_sharding_different_configuration():
    """Test oindex with sharding using different array/chunk/shard configuration"""
    group = zarr.group(store={}, zarr_format=3)
    
    # Larger array with multiple chunks per shard
    array = group.create_array(
        name="test_array",
        shape=(10, 20, 5),
        chunks=(2, 4, 5),   # 5x5x1 chunks
        shards=(6, 12, 5),  # 3x3x1 chunks per shard
        dtype=np.float64,
    )
    
    # Initialize with some data
    array[:] = np.arange(10 * 20 * 5).reshape(10, 20, 5)
    
    # Test orthogonal indexing across multiple shards
    indices = (
        np.array([4, 3, 7]),      # rows across different shards
        np.array([2, 8, 14, 19]), # columns across different shards  
        np.array([0, 4])          # depths
    )
    
    selection = array.oindex[indices]
    assert selection.shape == (3, 4, 2)  # orthogonal indexing gives outer product
    
    new_values = np.ones((3, 4, 2)) * 42.0
    array.oindex[indices] = new_values
    
    result = array.oindex[indices]
    np.testing.assert_array_equal(result, new_values)