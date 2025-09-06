import hashlib
import json
import os
import time
import fcntl
import threading
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from ga_constants import GAConstants


class MultiprocessCache:
    """File-based cache that works safely with multiprocessing."""
    
    def __init__(self, cache_dir: str = ".cache", 
                 max_entries: int = GAConstants.DEFAULT_CACHE_MAX_ENTRIES, 
                 max_age_hours: int = GAConstants.DEFAULT_CACHE_MAX_AGE_HOURS):
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.max_age_seconds = max_age_hours * 3600
        self._ensure_cache_dir()
        
        # Use file-based statistics for multiprocess safety
        self._stats_file = self.cache_dir / "stats.json"
        self._stats_lock = threading.Lock()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(exist_ok=True)
    
    def _update_stats(self, hits: int = 0, misses: int = 0):
        """Update statistics in a multiprocess-safe way."""
        try:
            with self._stats_lock:
                stats = {'hits': 0, 'misses': 0}
                
                # Read existing stats
                if self._stats_file.exists():
                    try:
                        with open(self._stats_file, 'r') as f:
                            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                            try:
                                stats = json.load(f)
                            finally:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (ValueError, OSError):
                        stats = {'hits': 0, 'misses': 0}
                
                # Update stats
                stats['hits'] = stats.get('hits', 0) + hits
                stats['misses'] = stats.get('misses', 0) + misses
                
                # Write updated stats atomically
                temp_stats = self._stats_file.with_suffix('.tmp')
                with open(temp_stats, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(stats, f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                temp_stats.replace(self._stats_file)
        except (OSError, ValueError):
            # Silently fail if we can't update stats
            pass
    
    def _read_stats(self):
        """Read current statistics."""
        try:
            if self._stats_file.exists():
                with open(self._stats_file, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        stats = json.load(f)
                        return stats.get('hits', 0), stats.get('misses', 0)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (ValueError, OSError):
            pass
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
        return hashlib.sha256(cache_data.encode()).hexdigest()[:32]
    
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
    
    def get(self, compressor_type: str, params: Dict[str, Any], input_file_path: str) -> Optional[float]:
        """Retrieve cached compression ratio if available."""
        cache_key = self._generate_cache_key(compressor_type, params, input_file_path)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            if cache_file.exists() and self._is_cache_entry_valid(cache_file):
                with open(cache_file, 'r') as f:
                    # Use file locking for safe concurrent access
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                        self._update_stats(hits=1)
                        return data.get('result')
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                self._update_stats(misses=1)
                return None
        except (OSError, ValueError):
            self._update_stats(misses=1)
            return None
    
    def set(self, compressor_type: str, params: Dict[str, Any], input_file_path: str, result: float) -> None:
        """Cache a compression result."""
        if result is None or result <= 0:
            return  # Don't cache invalid results
        
        cache_key = self._generate_cache_key(compressor_type, params, input_file_path)
        cache_file = self._get_cache_file_path(cache_key)
        
        # Clean up old entries if cache is getting too large
        self._cleanup_old_entries()
        
        try:
            cache_data = {
                'result': result,
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
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic move
            temp_file.replace(cache_file)
            
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
            # Don't count stats.json as a cache entry
            cache_files = [f for f in self.cache_dir.glob("*.json") if f.name != "stats.json"]
            cache_size = len(cache_files)
        except OSError:
            cache_size = 0
        
        return hits, misses, hit_rate, cache_size
    
    def clear(self) -> None:
        """Clear the entire cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            
            # Reset statistics
            try:
                with open(self._stats_file, 'w') as f:
                    json.dump({'hits': 0, 'misses': 0}, f)
            except (OSError, ValueError):
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