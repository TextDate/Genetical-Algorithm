#!/usr/bin/env python3
"""
Basic cache test to verify file-based caching works correctly.
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_cache():
    """Test basic cache functionality."""
    try:
        from cache import get_global_cache
        
        print("Testing basic cache functionality...")
        
        # Get cache instance
        cache = get_global_cache()
        print(f"Cache directory: {cache.cache_dir}")
        
        # Clear cache first
        cache._cleanup_old_entries()
        
        # Test parameters - use unique values with timestamp
        import time
        timestamp = int(time.time())
        test_params = {'level': 99, 'window_log': 99, 'unique_id': timestamp}
        test_file = f'test_file_{timestamp}.txt'
        compressor_type = f'test_compressor_{timestamp}'
        
        # Test cache miss
        result = cache.get(compressor_type, test_params, test_file)
        if result is not None:
            print(f"FAIL: Cache should return None for non-existent entry, but got: {result}")
            return False
        print("PASS: Cache miss working correctly")
        
        # Test cache set
        cache.set(compressor_type, test_params, test_file, 2.5, 1.23)
        
        # Test cache hit
        result = cache.get(compressor_type, test_params, test_file)
        if result is None:
            print("FAIL: Cache should return stored value")
            return False
        
        if isinstance(result, tuple) and len(result) == 2:
            fitness, compression_time = result
            if abs(fitness - 2.5) > 0.001 or abs(compression_time - 1.23) > 0.001:
                print(f"FAIL: Cached values incorrect: {result} vs (2.5, 1.23)")
                return False
        else:
            print(f"FAIL: Invalid cached result format: {result}")
            return False
        
        print("PASS: Cache hit working correctly")
        
        # Test cache stats
        stats = cache.get_stats()
        hits, misses, hit_rate, total_entries = stats
        print(f"Cache stats: {hits} hits, {misses} misses, {hit_rate:.1%} hit rate, {total_entries} entries")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"FAIL: Cache test failed with exception: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("="*60)
    print("BASIC CACHE FUNCTIONALITY TEST")
    print("="*60)
    
    success = test_basic_cache()
    
    print("="*60)
    if success:
        print("PASS: Basic cache test passed")
    else:
        print("FAIL: Basic cache test failed")
    print("="*60)
    
    sys.exit(0 if success else 1)