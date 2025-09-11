import hashlib
import json
import os
import time
import fcntl
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from ga_constants import GAConstants


class MultiprocessCache:
    """File-based cache that works safely with multiprocessing."""
    
    def __init__(self, cache_dir: str = ".cache", 
                 max_entries: int = GAConstants.DEFAULT_CACHE_MAX_ENTRIES, 
                 max_age_hours: int = GAConstants.DEFAULT_CACHE_MAX_AGE_HOURS):
        # Force absolute path to ensure all processes use the same cache directory
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.abspath(cache_dir)
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.max_age_seconds = max_age_hours * 3600
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(exist_ok=True)
    
    def _record_hit(self):
        """Record a cache hit by creating a simple marker file."""
        try:
            hit_file = self.cache_dir / f"hit_{time.time_ns()}.marker"
            hit_file.write_text("1")
        except Exception:
            pass  # Ignore errors
    
    def _record_miss(self):
        """Record a cache miss by creating a simple marker file."""
        try:
            miss_file = self.cache_dir / f"miss_{time.time_ns()}.marker"
            miss_file.write_text("1")
        except Exception:
            pass  # Ignore errors
    
    def _read_stats(self):
        """Read current statistics by counting marker files."""
        try:
            hits = len(list(self.cache_dir.glob("hit_*.marker")))
            misses = len(list(self.cache_dir.glob("miss_*.marker")))
            return hits, misses
        except Exception:
            return 0, 0
    
    def _generate_cache_key(self, compressor_type: str, params: Dict[str, Any], input_file_path: str) -> str:
        """Generate a consistent cache key."""
        # Include file metadata for cache validity
        try:
            file_size = os.path.getsize(input_file_path)
            file_mtime = os.path.getmtime(input_file_path)
        except OSError:
            # File doesn't exist or can't be accessed
            file_size = 0
            file_mtime = 0
        
        # Create cache key
        params_str = json.dumps(params, sort_keys=True, default=str)
        cache_data = f"{compressor_type}|{params_str}|{input_file_path}|{file_size}|{file_mtime}"
        
        # Hash for consistent, filesystem-safe key
        cache_key = hashlib.sha256(cache_data.encode()).hexdigest()[:32]
        return cache_key
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_entry_valid(self, cache_file: Path) -> bool:
        """Check if a cache entry is still valid (not too old)."""
        try:
            age = time.time() - cache_file.stat().st_mtime
            return age < self.max_age_seconds
        except OSError:
            return False
    
    def get(self, compressor_type: str, params: Dict[str, Any], input_file_path: str) -> Optional[Tuple[float, float, float]]:
        """Retrieve cached compression ratio, time, and RAM usage if available."""
        cache_key = self._generate_cache_key(compressor_type, params, input_file_path)
        cache_file = self._get_cache_file_path(cache_key)
        
        # Try twice with a small delay to handle race conditions
        for attempt in range(2):
            try:
                if cache_file.exists() and self._is_cache_entry_valid(cache_file):
                    with open(cache_file, 'r') as f:
                        # Use file locking for safe concurrent access
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            data = json.load(f)
                            self._record_hit()
                            # Return tuple (fitness, time, ram) with backward compatibility
                            result = data.get('result')
                            compression_time = data.get('compression_time', 0.0)  # Default to 0.0 for old cache entries
                            ram_usage = data.get('ram_usage', 0.0)  # Default to 0.0 for old cache entries
                            return (result, compression_time, ram_usage)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    # If first attempt fails and file might be being written, wait and retry
                    if attempt == 0:
                        time.sleep(0.02)  # 20ms delay before retry
                        continue
                    # This is the final attempt - record miss
                    break  # Exit loop to record miss at the end
            except (OSError, ValueError) as e:
                # If first attempt fails with an error, wait and retry
                if attempt == 0:
                    time.sleep(0.02)  # 20ms delay before retry
                    continue
                # This is the final attempt - record miss
                break  # Exit loop to record miss at the end
        
        # Record miss only once at the end
        self._record_miss()
        return None
    
    def set(self, compressor_type: str, params: Dict[str, Any], input_file_path: str, result: float, compression_time: float = 0.0, ram_usage: float = 0.0) -> None:
        """Cache a compression result with timing and RAM usage."""
        if result is None or result <= 0:
            return  # Don't cache invalid results
        
        cache_key = self._generate_cache_key(compressor_type, params, input_file_path)
        cache_file = self._get_cache_file_path(cache_key)
        
        # Clean up old entries if cache is getting too large
        self._cleanup_old_entries()
        
        try:
            cache_data = {
                'result': result,
                'compression_time': compression_time,
                'ram_usage': ram_usage,
                'timestamp': time.time(),
                'compressor': compressor_type,
                'input_file': input_file_path
            }
            
            # Write atomically to avoid corruption during concurrent access
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                # Use exclusive lock while writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(cache_data, f)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force file system sync
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic move
            temp_file.replace(cache_file)
            
            # Small delay to ensure file system consistency across processes
            time.sleep(0.01)  # 10ms delay for file system sync
            
        except (OSError, ValueError):
            # Silently fail if we can't write to cache
            pass
    
    def _cleanup_old_entries(self) -> None:
        """Remove old cache entries if we have too many."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            
            if len(cache_files) > self.max_entries:
                # Sort by modification time and remove oldest 10%
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                files_to_remove = cache_files[:len(cache_files) // 10]
                
                for cache_file in files_to_remove:
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass  # Ignore errors during cleanup
        except OSError:
            pass  # Ignore errors during cleanup
    
    def get_stats(self) -> Tuple[int, int, float, int]:
        """Return cache statistics: (hits, misses, hit_rate, size)."""
        hits, misses = self._read_stats()
        total_requests = hits + misses
        hit_rate = hits / total_requests if total_requests > 0 else 0.0
        
        try:
            # Count actual cache files (not marker files)
            cache_files = [f for f in self.cache_dir.glob("*.json") if not f.name.endswith('.marker')]
            cache_size = len(cache_files)
        except OSError:
            cache_size = 0
        
        return hits, misses, hit_rate, cache_size
    
    def clear(self) -> None:
        """Clear the entire cache."""
        try:
            # Clear cache files
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            
            # Clear statistics marker files
            for marker_file in self.cache_dir.glob("*.marker"):
                try:
                    marker_file.unlink()
                except OSError:
                    pass
        except OSError:
            pass
    
    def print_stats(self) -> None:
        """Print cache statistics."""
        hits, misses, hit_rate, size = self.get_stats()
        print(f"Compression Cache Stats: {hits} hits, {misses} misses, {hit_rate:.1%} hit rate, {size} entries")


# Global cache instance
_global_cache = None


def get_global_cache() -> MultiprocessCache:
    """Get the global multiprocess-safe compression cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MultiprocessCache()
    return _global_cache