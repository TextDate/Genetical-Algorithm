"""
Optimized Cache Manager for Compression Results

Provides separate caches per compressor type to avoid collisions,
smart LRU management, and enhanced cache key generation for better hit rates.
"""

import hashlib
import json
import os
import time
import threading
from typing import Dict, Any, Optional, Set, List, Tuple
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict
import psutil
from ga_constants import GAConstants, MemoryConstants, bytes_to_mb
from ga_logging import get_logger


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    value: float
    timestamp: float
    access_count: int
    compressor_type: str
    file_hash: str
    compressed_size: int = 0
    compression_time: float = 0.0
    

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


class CompressorCache:
    """
    LRU cache for a specific compressor type with intelligent management.
    """
    
    def __init__(self, compressor_type: str, max_entries: int = 2000):
        self.compressor_type = compressor_type
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_lock = threading.RLock()
        self.stats = CacheStats()
        self.logger = get_logger(f"Cache-{compressor_type}")
        
    def get(self, key: str) -> Optional[Tuple[float, float]]:
        """Get value and compression time from cache, updating access statistics."""
        with self.access_lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats.hits += 1
                return (entry.value, entry.compression_time)
            else:
                self.stats.misses += 1
                return None
    
    def put(self, key: str, entry: CacheEntry) -> None:
        """Store value in cache with LRU eviction."""
        with self.access_lock:
            if key in self.cache:
                # Update existing entry
                self.cache[key] = entry
                self.cache.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.max_entries:
                    self._evict_oldest()
                
                self.cache[key] = entry
                self.stats.total_entries = len(self.cache)
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return
            
        # Evict the oldest (least recently used) entry
        oldest_key, oldest_entry = self.cache.popitem(last=False)
        self.stats.evictions += 1
        
        self.logger.debug("Evicted cache entry",
                         key=oldest_key[:16] + "...",
                         access_count=oldest_entry.access_count,
                         age_hours=f"{(time.time() - oldest_entry.timestamp) / 3600:.1f}")
    
    def clear_expired(self, max_age_seconds: int) -> int:
        """Remove expired entries, return count removed."""
        current_time = time.time()
        expired_keys = []
        
        with self.access_lock:
            for key, entry in self.cache.items():
                if current_time - entry.timestamp > max_age_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            self.stats.total_entries = len(self.cache)
        
        if expired_keys:
            self.logger.debug(f"Removed {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self.access_lock:
            self.stats.total_entries = len(self.cache)
            # Estimate memory usage (rough approximation)
            self.stats.memory_usage_mb = len(self.cache) * 0.5  # ~0.5KB per entry
            return self.stats
    
    def get_top_performers(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed entries."""
        with self.access_lock:
            entries = [(key, entry.access_count) for key, entry in self.cache.items()]
            entries.sort(key=lambda x: x[1], reverse=True)
            return entries[:limit]


class OptimizedCacheManager:
    """
    Multi-compressor cache manager with optimization features.
    
    Maintains separate caches per compressor type to avoid key collisions,
    implements intelligent cache sizing, and provides advanced statistics.
    """
    
    def __init__(self, base_cache_dir: str = ".optimized_cache", 
                 total_max_entries: int = GAConstants.DEFAULT_CACHE_MAX_ENTRIES):
        self.base_cache_dir = Path(base_cache_dir)
        self.total_max_entries = total_max_entries
        self.logger = get_logger("OptimizedCacheManager")
        
        # Separate cache instances per compressor type
        self.caches: Dict[str, CompressorCache] = {}
        self.cache_allocation: Dict[str, int] = {}
        
        # Global statistics
        self.global_stats = CacheStats()
        self.cache_lock = threading.RLock()
        
        # Configuration
        self.max_age_hours = GAConstants.DEFAULT_CACHE_MAX_AGE_HOURS
        self.cleanup_interval_seconds = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Initialize directory structure
        self._ensure_cache_dirs()
        
        # Load existing cache allocations or initialize defaults
        self._initialize_cache_allocation()
        
        self.logger.info("Optimized cache manager initialized",
                        total_max_entries=total_max_entries,
                        cache_dir=str(self.base_cache_dir))
    
    def _ensure_cache_dirs(self):
        """Create cache directory structure."""
        self.base_cache_dir.mkdir(exist_ok=True)
        (self.base_cache_dir / "metadata").mkdir(exist_ok=True)
    
    def _initialize_cache_allocation(self):
        """Initialize cache size allocation per compressor type.
        
        For single-compressor execution, allocate full cache capacity to
        whichever compressor is actively used, maximizing cache efficiency.
        """
        # Single-compressor mode: allocate minimal initial cache, 
        # full capacity will be allocated dynamically when compressor is first used
        initial_cache_size = 100  # Minimal initial allocation
        
        # Pre-allocate small caches for known compressor types
        compressor_types = ['zstd', 'brotli', 'lzma', 'paq8']
        for compressor_type in compressor_types:
            self.cache_allocation[compressor_type] = initial_cache_size
            self.caches[compressor_type] = CompressorCache(compressor_type, initial_cache_size)
        
        self.logger.debug("Cache allocation initialized with single-compressor optimization", 
                         initial_cache_size=initial_cache_size,
                         total_max_entries=self.total_max_entries)
    
    def _get_cache_key(self, compressor_type: str, params: Dict[str, Any], 
                      input_file_path: str) -> str:
        """Generate optimized cache key with better collision avoidance."""
        # Enhanced key generation with more context
        try:
            file_stats = os.stat(input_file_path)
            file_size = file_stats.st_size
            file_mtime = file_stats.st_mtime
            
            # Include file hash for small files (< 10MB) for better accuracy
            file_hash = ""
            if file_size < 10 * 1024 * 1024:  # 10MB threshold
                try:
                    with open(input_file_path, 'rb') as f:
                        content_sample = f.read(8192)  # First 8KB
                        file_hash = hashlib.md5(content_sample).hexdigest()[:8]
                except (OSError, IOError):
                    pass
            
        except OSError:
            file_size = 0
            file_mtime = 0
            file_hash = ""
        
        # Create enhanced cache key with compressor-specific context
        params_str = json.dumps(params, sort_keys=True, default=str)
        
        # Include compressor type in key to absolutely prevent cross-compressor collisions
        cache_components = [
            compressor_type,
            params_str,
            os.path.basename(input_file_path),  # Just filename to reduce key length
            str(file_size),
            str(int(file_mtime)),
            file_hash
        ]
        
        key_data = "|".join(cache_components)
        full_hash = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Use shorter key with compressor prefix for readability
        return f"{compressor_type[:4]}_{full_hash[:24]}"
    
    def get_cache_for_compressor(self, compressor_type: str) -> CompressorCache:
        """Get or create cache for specific compressor type with single-compressor optimization."""
        compressor_key = compressor_type.lower().replace('compressor', '')
        
        if compressor_key not in self.caches:
            # Create new cache with full capacity for single-compressor mode
            self.caches[compressor_key] = CompressorCache(compressor_key, self.total_max_entries)
            self.cache_allocation[compressor_key] = self.total_max_entries
            
            self.logger.info(f"Created new cache for {compressor_key} with full capacity",
                           max_entries=self.total_max_entries)
        else:
            # Check if this is the first real usage - upgrade from initial small cache
            if self.caches[compressor_key].max_entries < self.total_max_entries:
                # This compressor is being used - allocate full capacity
                self.caches[compressor_key].max_entries = self.total_max_entries
                self.cache_allocation[compressor_key] = self.total_max_entries
                
                self.logger.info(f"Upgraded {compressor_key} cache to full capacity",
                               max_entries=self.total_max_entries,
                               current_entries=len(self.caches[compressor_key].cache))
        
        return self.caches[compressor_key]
    
    def get(self, compressor_type: str, params: Dict[str, Any], 
           input_file_path: str) -> Optional[Tuple[float, float]]:
        """Get cached compression result and timing."""
        cache_key = self._get_cache_key(compressor_type, params, input_file_path)
        cache = self.get_cache_for_compressor(compressor_type)
        
        result = cache.get(cache_key)
        
        # Update global statistics
        with self.cache_lock:
            if result is not None:
                self.global_stats.hits += 1
            else:
                self.global_stats.misses += 1
        
        # Periodic cleanup
        self._maybe_cleanup()
        
        return result
    
    def put(self, compressor_type: str, params: Dict[str, Any], 
           input_file_path: str, value: float, 
           compressed_size: int = 0, compression_time: float = 0.0) -> None:
        """Store compression result in cache."""
        cache_key = self._get_cache_key(compressor_type, params, input_file_path)
        cache = self.get_cache_for_compressor(compressor_type)
        
        # Create file hash for the entry
        file_hash = ""
        try:
            file_stats = os.stat(input_file_path)
            if file_stats.st_size < 10 * 1024 * 1024:  # 10MB
                with open(input_file_path, 'rb') as f:
                    content_sample = f.read(4096)  # First 4KB
                    file_hash = hashlib.md5(content_sample).hexdigest()[:8]
        except (OSError, IOError):
            pass
        
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            access_count=1,
            compressor_type=compressor_type,
            file_hash=file_hash,
            compressed_size=compressed_size,
            compression_time=compression_time
        )
        
        cache.put(cache_key, entry)
        
        self.logger.debug("Cached compression result",
                         compressor=compressor_type,
                         value=f"{value:.3f}",
                         key=cache_key[:12] + "...")
    
    def _maybe_cleanup(self):
        """Perform periodic cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval_seconds:
            self._cleanup_expired_entries()
            self.last_cleanup = current_time
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from all caches."""
        max_age_seconds = self.max_age_hours * 3600
        total_removed = 0
        
        for compressor_type, cache in self.caches.items():
            removed = cache.clear_expired(max_age_seconds)
            total_removed += removed
        
        if total_removed > 0:
            self.logger.info(f"Cleaned up {total_removed} expired cache entries")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all caches."""
        stats = {
            'global': {
                'total_hits': self.global_stats.hits,
                'total_misses': self.global_stats.misses,
                'overall_hit_rate': self.global_stats.hit_rate,
                'total_caches': len(self.caches)
            },
            'by_compressor': {}
        }
        
        total_entries = 0
        total_memory_mb = 0.0
        
        for compressor_type, cache in self.caches.items():
            cache_stats = cache.get_stats()
            stats['by_compressor'][compressor_type] = {
                'hits': cache_stats.hits,
                'misses': cache_stats.misses,
                'hit_rate': cache_stats.hit_rate,
                'entries': cache_stats.total_entries,
                'evictions': cache_stats.evictions,
                'memory_mb': cache_stats.memory_usage_mb,
                'max_entries': cache.max_entries
            }
            
            total_entries += cache_stats.total_entries
            total_memory_mb += cache_stats.memory_usage_mb
        
        stats['global']['total_entries'] = total_entries
        stats['global']['memory_usage_mb'] = total_memory_mb
        
        return stats
    
    def optimize_cache_allocation(self) -> None:
        """
        Single-compressor optimization: maintain full capacity for active compressor.
        
        In single-compressor mode, the active compressor already has full capacity,
        so optimization focuses on cache maintenance rather than reallocation.
        """
        stats = self.get_comprehensive_stats()
        
        # Identify the active compressor (the one with actual usage)
        active_compressor = None
        max_requests = 0
        
        for compressor_type, comp_stats in stats['by_compressor'].items():
            requests = comp_stats['hits'] + comp_stats['misses']
            if requests > max_requests:
                max_requests = requests
                active_compressor = compressor_type
        
        if not active_compressor or max_requests == 0:
            self.logger.debug("No active compressor usage detected")
            return
        
        # Ensure active compressor has full capacity
        if active_compressor in self.caches:
            current_max = self.caches[active_compressor].max_entries
            if current_max < self.total_max_entries:
                self.caches[active_compressor].max_entries = self.total_max_entries
                self.cache_allocation[active_compressor] = self.total_max_entries
                
                self.logger.info(f"Ensured full capacity for active compressor {active_compressor}",
                               old_max=current_max, 
                               new_max=self.total_max_entries,
                               hit_rate=f"{stats['by_compressor'][active_compressor]['hit_rate']:.3f}")
        
        # Log single-compressor optimization status
        if active_compressor in stats['by_compressor']:
            comp_stats = stats['by_compressor'][active_compressor]
            self.logger.info("Single-compressor cache optimization complete",
                           compressor=active_compressor,
                           cache_entries=comp_stats['entries'],
                           hit_rate=f"{comp_stats['hit_rate']:.3f}",
                           total_requests=max_requests)
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self.cache_lock:
            for cache in self.caches.values():
                cache.cache.clear()
                cache.stats = CacheStats()
            
            self.global_stats = CacheStats()
        
        self.logger.info("All caches cleared")


# Global optimized cache manager instance
_optimized_cache_manager: Optional[OptimizedCacheManager] = None


def get_optimized_cache_manager() -> OptimizedCacheManager:
    """Get or create the global optimized cache manager."""
    global _optimized_cache_manager
    if _optimized_cache_manager is None:
        _optimized_cache_manager = OptimizedCacheManager()
    return _optimized_cache_manager


def get_cached_result(compressor_type: str, params: Dict[str, Any], 
                     input_file_path: str) -> Optional[float]:
    """Convenience function to get cached compression result."""
    manager = get_optimized_cache_manager()
    return manager.get(compressor_type, params, input_file_path)


def cache_result(compressor_type: str, params: Dict[str, Any], 
                input_file_path: str, value: float,
                compressed_size: int = 0, compression_time: float = 0.0) -> None:
    """Convenience function to cache compression result."""
    manager = get_optimized_cache_manager()
    manager.put(compressor_type, params, input_file_path, value, 
               compressed_size, compression_time)