#!/usr/bin/env python3
"""
Test script to verify multiprocess cache consistency.
This ensures that identical parameters return identical compression times from cache.
"""

import sys
import os
import time
import concurrent.futures
from multiprocessing import Process, Queue

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Compressors.zstd_compressor import ZstdCompressor
from cache import get_global_cache

def test_worker(worker_id, test_file, params, results_queue):
    """Worker function to test cache consistency across processes."""
    try:
        # Create compressor instance
        compressor = ZstdCompressor(test_file)
        
        # Perform compression multiple times with same parameters
        times = []
        for i in range(3):
            start_time = time.time()
            result = compressor.evaluate(params, f"worker_{worker_id}_test_{i}")
            end_time = time.time()
            
            if isinstance(result, tuple):
                fitness, compression_time = result
            else:
                fitness = result
                compression_time = end_time - start_time
            
            times.append(compression_time)
            print(f"Worker {worker_id}, Test {i}: Fitness={fitness:.4f}, Time={compression_time:.3f}s")
        
        results_queue.put((worker_id, times))
        
    except Exception as e:
        import traceback
        error_msg = f"Worker {worker_id} exception: {e}\n{traceback.format_exc()}"
        print(error_msg)
        results_queue.put((worker_id, None))

def main():
    """Main test function."""
    # Use a small test file - try different paths
    possible_paths = [
        "data/chemistry_samples/1mb/Compound_000000001_000500000_1mb.sdf",
        "../data/chemistry_samples/1mb/Compound_000000001_000500000_1mb.sdf"
    ]
    
    test_file = None
    for path in possible_paths:
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file is None:
        print(f"Test file not found in any of these locations: {possible_paths}")
        print("SKIP: Chemistry sample file not available - test cannot run")
        return True  # Return True to skip test gracefully
    
    # Test parameters (identical for all workers)
    test_params = {
        'level': 13,
        'window_log': 23,
        'chain_log': 28,
        'hash_log': 26, 
        'search_log': 6,
        'min_match': 4,
        'target_length': 4096,
        'strategy': 0
    }
    
    print("="*60)
    print("MULTIPROCESS CACHE CONSISTENCY TEST")
    print("="*60)
    print(f"Test file: {test_file}")
    print(f"Test parameters: {test_params}")
    print("-"*60)
    
    # Clear cache first
    cache = get_global_cache()
    cache._cleanup_old_entries()  # Clean old entries
    
    # Start multiple worker processes
    num_workers = 3
    results_queue = Queue()
    processes = []
    
    for worker_id in range(num_workers):
        p = Process(target=test_worker, args=(worker_id, test_file, test_params, results_queue))
        p.start()
        processes.append(p)
    
    # Wait for all processes and collect results
    worker_results = {}
    for _ in range(num_workers):
        try:
            worker_id, times = results_queue.get(timeout=120)  # 2 minute timeout
            if times is not None:
                worker_results[worker_id] = times
            else:
                print(f"Worker {worker_id} failed!")
        except Exception as e:
            print(f"Error getting results: {e}")
    
    # Wait for processes to complete
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    print("-"*60)
    print("RESULTS ANALYSIS")
    print("-"*60)
    
    if len(worker_results) < num_workers:
        print(f"WARNING: Only {len(worker_results)}/{num_workers} workers completed successfully")
        return False
    
    # Analyze cache consistency
    all_times = []
    for worker_id, times in worker_results.items():
        print(f"Worker {worker_id}: {[f'{t:.3f}s' for t in times]}")
        all_times.extend(times)
    
    # Check for consistency (after first cache miss, all should be cache hits)
    cache_hits = [t for t in all_times if t < 0.1]  # Very fast times indicate cache hits
    cache_misses = [t for t in all_times if t >= 0.1]  # Slower times indicate cache misses
    
    print(f"\nCache hits (< 0.1s): {len(cache_hits)}")
    print(f"Cache misses (>= 0.1s): {len(cache_misses)}")
    
    # Get final cache statistics
    try:
        hits, misses = cache._get_stats()
        print(f"\nCache Statistics:")
        print(f"  Total hits: {hits}")
        print(f"  Total misses: {misses}")
        print(f"  Hit rate: {(hits/(hits+misses)*100) if (hits+misses) > 0 else 0:.1f}%")
    except:
        print("Could not retrieve cache statistics")
    
    # Test success criteria
    success = True
    if len(cache_misses) > 3:  # Should be at most 1 miss per worker for first evaluation
        print(f"FAIL: Too many cache misses: {len(cache_misses)} (expected <= 3)")
        success = False
    
    if len(cache_hits) < (len(all_times) - 3):
        print(f"FAIL: Too few cache hits: {len(cache_hits)} (expected >= {len(all_times) - 3})")
        success = False
    
    print("="*60)
    if success:
        print("PASS: MULTIPROCESS CACHE TEST PASSED")
        print("      Cache is working correctly across processes!")
    else:
        print("FAIL: MULTIPROCESS CACHE TEST FAILED")
        print("      Cache inconsistency detected!")
    print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)