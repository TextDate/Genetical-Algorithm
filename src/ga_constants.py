"""
Configuration Constants for Genetic Algorithm

Centralizes all magic numbers and hard-coded values for better maintainability.
All constants are organized by category with clear documentation.
"""

from typing import Dict


class GAConstants:
    """Configuration constants for genetic algorithm components."""
    
    # Fitness and Quality Constants
    MIN_FITNESS = 0.1              # Fallback fitness for failed evaluations
    MIN_DIVERSITY_THRESHOLD = 0.1  # Minimum population diversity before intervention
    
    # Performance and Memory Constants  
    MEMORY_WARNING_THRESHOLD_MB = 500      # Memory usage warning threshold (MB)
    MEMORY_CRITICAL_THRESHOLD_MB = 1000    # Memory usage critical threshold (MB)
    CHUNK_SIZE_KB = 64                     # Default chunk size for streaming (KB)
    CHUNK_SIZE_BYTES = 64 * 1024          # Chunk size in bytes
    
    # Cache Configuration
    DEFAULT_CACHE_MAX_ENTRIES = 10000     # Maximum cache entries
    DEFAULT_CACHE_MAX_AGE_HOURS = 24      # Cache entry max age
    
    # Algorithm Tuning Constants
    MUTATION_RATE_MIN = 0.001            # Minimum adaptive mutation rate
    MUTATION_RATE_MAX = 0.1              # Maximum adaptive mutation rate
    CROSSOVER_RATE_BOOST = 0.1           # Boost for crossover rate when needed
    MAX_CROSSOVER_RATE = 0.95            # Maximum crossover rate
    
    # Population Management
    MIN_TOURNAMENT_SIZE = 2              # Minimum tournament selection size
    LATE_PHASE_GENERATION_THRESHOLD = 20  # Generation threshold for late phase
    
    # Duplicate Prevention Thresholds
    DUPLICATE_PREVENTION_THRESHOLDS = {
        'low_duplicate_ratio': 0.1,       # Low duplicate ratio threshold
        'high_density_threshold': 0.1,    # High parameter space density
        'enforcement_adjustment': 0.1,    # Enforcement probability adjustment
        'late_phase_min': 0.6,           # Minimum late phase start
        'late_phase_max': 0.85,          # Maximum late phase start  
        'late_phase_base': 0.75          # Base late phase start
    }


class CompressionTimeouts:
    """Timeout configurations for different compression algorithms."""
    
    # Base timeout settings
    DEFAULT_TIMEOUT_SECONDS = 30         # Default compression timeout
    FILE_WAIT_TIMEOUT_SECONDS = 10       # File existence wait timeout
    
    # ZSTD-specific timeouts
    ZSTD_FAST_TIMEOUT = 15              # For compression levels <= 10
    ZSTD_SLOW_TIMEOUT = 30              # For compression levels > 10
    ZSTD_LEVEL_THRESHOLD = 10           # Threshold for timeout selection
    
    # LZMA-specific timeouts
    LZMA_FAST_TIMEOUT = 20              # Small dictionary, low nice_len
    LZMA_MEDIUM_TIMEOUT = 30            # Medium complexity
    LZMA_SLOW_TIMEOUT = 45              # Large dictionary or high nice_len
    LZMA_DICT_SIZE_THRESHOLD = 1048576  # 1MB dictionary size threshold
    LZMA_NICE_LEN_THRESHOLD = 64        # nice_len threshold
    
    # Brotli-specific timeouts
    BROTLI_FAST_TIMEOUT = 15            # Low quality/window
    BROTLI_MEDIUM_TIMEOUT = 20          # Medium quality/window
    BROTLI_SLOW_TIMEOUT = 25            # High quality/window
    BROTLI_QUALITY_THRESHOLD = 8        # Quality threshold
    BROTLI_WINDOW_THRESHOLD = 20        # Window size threshold
    
    # PAQ8-specific timeouts (very slow compressor)
    PAQ8_TIMEOUT = 7200                 # 2 hours for PAQ8 compression


class TestConstants:
    """Constants used in testing."""
    
    TEST_TIMEOUT_SECONDS = 30           # Default test timeout
    TEST_MIN_FITNESS = 0.1              # Minimum fitness for test validation
    TEST_TIMEOUT_RATE = 0.2             # Rate of timeout simulation in tests
    TEST_SLEEP_TIME = 0.1               # Sleep time for timeout simulation
    
    # Memory test thresholds
    TEST_MEMORY_INCREASE_LIMIT_MB = 200  # Maximum memory increase in tests (MB)
    TEST_MEMORY_INTEGRATION_LIMIT_MB = 100  # Integration test memory limit (MB)
    
    # Performance test values
    TEST_MEMORY_100MB = 100 * 1024 * 1024   # 100MB in bytes
    TEST_MEMORY_50MB = 50 * 1024 * 1024     # 50MB in bytes


class MemoryConstants:
    """Memory-related configuration constants."""
    
    # Unit conversion constants
    BYTES_PER_KB = 1024
    BYTES_PER_MB = 1024 * 1024
    BYTES_PER_GB = 1024 ** 3
    
    # Memory monitoring
    MEMORY_POLLING_INTERVAL = 1          # Memory polling interval (seconds) 
    MEMORY_HISTORY_SIZE = 10             # Number of memory samples to keep
    MEMORY_WARNING_SIZE = 5              # Samples for warning calculation


class AlgorithmConstants:
    """Algorithm-specific configuration constants."""
    
    # Generation tracking
    CONVERGENCE_HISTORY_SIZE = 5         # Generations to track for convergence
    FITNESS_IMPROVEMENT_PRECISION = 4    # Decimal places for fitness comparison
    
    # Adaptive tuning
    INITIAL_MUTATION_RATE = 0.1          # Default initial mutation rate
    INITIAL_CROSSOVER_RATE = 0.8         # Default initial crossover rate
    
    # Performance monitoring intervals
    GENERATION_METRICS_HISTORY = 20      # Generations of metrics to keep


# Convenient access to commonly used constants
MIN_FITNESS = GAConstants.MIN_FITNESS
DEFAULT_TIMEOUT = CompressionTimeouts.DEFAULT_TIMEOUT_SECONDS
CHUNK_SIZE = GAConstants.CHUNK_SIZE_BYTES


def get_compression_timeout(compressor_type: str, params: Dict = None, file_size_mb: float = None) -> int:
    """
    Get appropriate timeout for compression based on type, parameters, and file size.
    
    Args:
        compressor_type: Type of compressor ('zstd', 'lzma', 'brotli', 'paq8')
        params: Compression parameters (optional)
        file_size_mb: Input file size in MB (optional, enables scaling)
        
    Returns:
        Timeout in seconds
    """
    # First, determine base timeout based on compressor type and parameters
    if compressor_type.lower() == 'zstd':
        if params and params.get('level', 0) > CompressionTimeouts.ZSTD_LEVEL_THRESHOLD:
            base_timeout = CompressionTimeouts.ZSTD_SLOW_TIMEOUT
        else:
            base_timeout = CompressionTimeouts.ZSTD_FAST_TIMEOUT
        
    elif compressor_type.lower() == 'lzma':
        if params:
            dict_size = params.get('dict_size', 0)
            nice_len = params.get('nice_len', 0)
            if (dict_size >= CompressionTimeouts.LZMA_DICT_SIZE_THRESHOLD or 
                nice_len >= CompressionTimeouts.LZMA_NICE_LEN_THRESHOLD):
                base_timeout = CompressionTimeouts.LZMA_SLOW_TIMEOUT
            elif dict_size > 0 or nice_len > 0:
                base_timeout = CompressionTimeouts.LZMA_MEDIUM_TIMEOUT
            else:
                base_timeout = CompressionTimeouts.LZMA_FAST_TIMEOUT
        else:
            base_timeout = CompressionTimeouts.LZMA_FAST_TIMEOUT
        
    elif compressor_type.lower() == 'brotli':
        if params:
            quality = params.get('quality', 0)
            window = params.get('window', 0)
            if (quality > CompressionTimeouts.BROTLI_QUALITY_THRESHOLD or 
                window > CompressionTimeouts.BROTLI_WINDOW_THRESHOLD):
                base_timeout = CompressionTimeouts.BROTLI_SLOW_TIMEOUT
            elif quality > 0 or window > 0:
                base_timeout = CompressionTimeouts.BROTLI_MEDIUM_TIMEOUT
            else:
                base_timeout = CompressionTimeouts.BROTLI_FAST_TIMEOUT
        else:
            base_timeout = CompressionTimeouts.BROTLI_FAST_TIMEOUT
        
    elif compressor_type.lower() == 'paq8':
        base_timeout = CompressionTimeouts.PAQ8_TIMEOUT
        
    else:
        base_timeout = CompressionTimeouts.DEFAULT_TIMEOUT_SECONDS
    
    # Apply file-size-aware timeout scaling if file size is provided
    if file_size_mb is not None and file_size_mb > 10:  # Only scale for files > 10MB
        # File size scaling factors:
        # 10-50MB: 1-2x multiplier
        # 50-100MB: 2-4x multiplier  
        # 100-500MB: 4-10x multiplier
        # 500MB+: 10-20x multiplier
        
        if file_size_mb <= 50:
            # Linear scaling from 1x to 2x for 10-50MB
            scale_factor = 1.0 + (file_size_mb - 10) / 40.0
        elif file_size_mb <= 100:
            # Linear scaling from 2x to 4x for 50-100MB  
            scale_factor = 2.0 + 2.0 * (file_size_mb - 50) / 50.0
        elif file_size_mb <= 500:
            # Linear scaling from 4x to 10x for 100-500MB
            scale_factor = 4.0 + 6.0 * (file_size_mb - 100) / 400.0
        else:
            # Linear scaling from 10x to 20x for 500MB+
            scale_factor = 10.0 + 10.0 * min(file_size_mb - 500, 1500) / 1500.0
            scale_factor = min(scale_factor, 20.0)  # Cap at 20x
        
        scaled_timeout = int(base_timeout * scale_factor)
        # Cap maximum timeout at 2 hours for safety
        return min(scaled_timeout, 7200)
    
    return base_timeout


def bytes_to_mb(bytes_value: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_value / MemoryConstants.BYTES_PER_MB


def bytes_to_gb(bytes_value: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_value / MemoryConstants.BYTES_PER_GB


def mb_to_bytes(mb_value: float) -> int:
    """Convert megabytes to bytes."""
    return int(mb_value * MemoryConstants.BYTES_PER_MB)