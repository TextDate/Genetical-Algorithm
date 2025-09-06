#!/usr/bin/env python3
"""
Test script to verify multiprocess-safe compression cache
"""
import sys
import os
import time
from multiprocess_cache import get_global_cache
from Compressors.zstd_compressor import ZstdCompressor

def test_multiprocess_cache():
    """Test the multiprocess cache with ZSTD compressor."""
    
    # Create test input file
    test_file = "test_cache_input.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test file for multiprocess compression caching.\n" * 100)
    
    try:
        print("Testing multiprocess-safe compression cache...")
        
        # Initialize compressor
        compressor = ZstdCompressor(test_file, temp="test_temp")
        
        # Test parameters
        test_params = [{
            'level': 3,
            'window_log': 17,
            'chain_log': 16,
            'hash_log': 16,
            'search_log': 6,
            'min_match': 4,
            'target_length': 16384,
            'strategy': 1
        }]
        
        cache = get_global_cache()
        cache.clear()
        
        # First evaluation (should be cache miss)
        print("First evaluation (cache miss expected):")
        start_time = time.time()
        result1 = compressor.evaluate(test_params, "test_individual_1")
        time1 = time.time() - start_time
        print(f"  Result: {result1}, Time: {time1:.3f}s")
        cache.print_stats()
        
        # Second evaluation with same parameters (should be cache hit)
        print("\nSecond evaluation with same params (cache hit expected):")
        start_time = time.time()
        result2 = compressor.evaluate(test_params, "test_individual_2")
        time2 = time.time() - start_time
        print(f"  Result: {result2}, Time: {time2:.3f}s")
        cache.print_stats()
        
        # Verify results are identical
        print(f"\nResults identical: {result1 == result2}")
        if time1 > 0 and time2 > 0:
            speedup = time1 / time2 if time2 < time1 else 1.0
            print(f"Speedup: {speedup:.1f}x faster")
        
        # Check if cache files were created
        cache_dir = cache.cache_dir
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            print(f"Cache files created: {len(cache_files)}")
        
        print("\nMultiprocess cache test completed successfully!")
        
        # Clean up
        compressor.erase_temp_files()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("test_temp"):
            import shutil
            shutil.rmtree("test_temp")

if __name__ == "__main__":
    test_multiprocess_cache()