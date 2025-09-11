import os
import subprocess
import sys
import time
import shutil
import signal
import psutil
from contextlib import contextmanager
from cache import get_global_cache
from ga_constants import CompressionTimeouts, DEFAULT_TIMEOUT
from ga_logging import get_logger
# Always use multiprocess-safe file-based cache for worker processes


class BaseCompressor:
    def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
        self.logger = get_logger("Compressor")
        # Validate input file path
        if not input_file_path:
            raise ValueError("input_file_path cannot be empty")
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file does not exist: {input_file_path}")
        if not os.path.isfile(input_file_path):
            raise ValueError(f"Input path is not a file: {input_file_path}")
        
        # Validate reference file if provided
        if reference_file_path is not None:
            if not os.path.exists(reference_file_path):
                raise FileNotFoundError(f"Reference file does not exist: {reference_file_path}")
            if not os.path.isfile(reference_file_path):
                raise ValueError(f"Reference path is not a file: {reference_file_path}")
        
        # Validate temp directory path
        if not temp or not isinstance(temp, str):
            raise ValueError("temp directory path must be a non-empty string")
            
        self.input_file_path = input_file_path  # Path to the file to be compressed
        self.reference_file_path = reference_file_path  # Reference file path (optional)
        self.temp = temp  # Temporary storage folder
        self.errors = 0
        self.nr_models = 1
        # Cache access is now lazy to avoid pickling issues with multiprocessing

        # Create temp directory if it doesn't exist
        try:
            if not os.path.exists(self.temp):
                os.makedirs(self.temp)
        except OSError as e:
            raise OSError(f"Cannot create temp directory {self.temp}: {e}")

    def get_errors(self):
        """Return the number of errors encountered during compression."""
        return self.errors

    def erase_temp_files(self):
        """Efficiently remove specific temporary files instead of entire directory."""
        if os.path.exists(self.temp):
            try:
                # Remove only files, not the directory itself
                for filename in os.listdir(self.temp):
                    file_path = os.path.join(self.temp, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except OSError as e:
                            self.logger.warning(f"Could not remove temp file", 
                                               file_path=file_path, error=str(e))
                self.logger.info("Temp files successfully cleaned")
            except Exception as e:
                self.logger.error("Error while cleaning temp files", exception=e)
                # Fallback to old method if file listing fails
                try:
                    shutil.rmtree(self.temp)
                    os.makedirs(self.temp)
                    self.logger.info("Temp directory recreated as fallback")
                except Exception as fallback_error:
                    self.logger.error("Fallback temp cleanup also failed", 
                                     exception=fallback_error)

    def run_command(self, command):
        """Execute a system command safely and handle errors."""
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            self.errors += 1
            self.logger.error("Error running command", command=' '.join(command))
            return False
        except subprocess.TimeoutExpired:
            self.logger.warning("Command timed out", command=' '.join(command))
            return False
        return True

    def wait_for_file(self, file_path, timeout=CompressionTimeouts.FILE_WAIT_TIMEOUT_SECONDS):
        """Wait until a file exists (useful for waiting for compression output)."""
        elapsed_time = 0
        while not os.path.exists(file_path) and elapsed_time < timeout:
            self.logger.debug("Waiting for file", file_path=file_path)
            time.sleep(2)
            elapsed_time += 2
        return os.path.exists(file_path)

    def compute_compression_ratio(self, original_file, compressed_file):
        """Compute compression ratio: (original size / compressed size)."""
        if not os.path.exists(original_file) or not os.path.exists(compressed_file):
            self.logger.error("Compression size error: missing file(s)", 
                              original_file=original_file, 
                              compressed_file=compressed_file)
            return None
        original_size = os.path.getsize(original_file)
        compressed_size = os.path.getsize(compressed_file)
        return original_size / compressed_size if compressed_size > 0 else None
    
    @contextmanager
    def timeout(self, seconds: int):
        """Context manager for timing out operations."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Reset alarm and handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    @contextmanager
    def ram_monitor(self):
        """Context manager for monitoring RAM usage during operations."""
        process = psutil.Process()
        initial_ram = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        peak_ram_holder = [initial_ram]  # Use list to allow modification in nested function
        
        def get_current_ram():
            try:
                current_ram = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                peak_ram_holder[0] = max(peak_ram_holder[0], current_ram)
                return current_ram
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return peak_ram_holder[0]
        
        try:
            yield get_current_ram
        finally:
            # Final check for peak RAM
            get_current_ram()
    
    
    def evaluate_with_cache_and_timing(self, compressor_type: str, params: dict, name: str, evaluate_func, timeout_seconds: int = DEFAULT_TIMEOUT):
        """
        Evaluate compression with caching and timeout protection, returning fitness, timing, and RAM usage.
        
        Args:
            compressor_type: Name of the compressor (e.g., 'zstd', 'lzma')
            params: Parameter dictionary for compression
            name: Individual name for logging
            evaluate_func: Function to call if cache miss occurs
            timeout_seconds: Maximum time to wait for compression (default: 30s)
        
        Returns:
            Tuple of (compression_ratio, compression_time, ram_usage_mb) or (None, 0, 0) if evaluation failed
        """
        # Check multiprocess-safe cache first
        cache = get_global_cache()
        cached_result = cache.get(compressor_type, params, self.input_file_path)
        if cached_result is not None:
            self.logger.debug(f"Cache HIT for {name}: {cached_result}")
            # Handle backward compatibility - cached results might be (fitness, time) or (fitness, time, ram)
            if len(cached_result) == 2:
                fitness, compression_time = cached_result
                ram_usage = 0.0  # Default for legacy cache entries
            else:
                fitness, compression_time, ram_usage = cached_result
            return fitness, compression_time, ram_usage
        else:
            self.logger.debug(f"Cache MISS for {name}, parameters: {params}")
        
        # Cache miss - perform actual compression with timeout, timing, and RAM monitoring
        import time
        compression_start_time = time.time()
        ram_usage = 0.0
        
        try:
            with self.timeout(timeout_seconds), self.ram_monitor() as get_ram:
                result = evaluate_func(params, name)
                ram_usage = get_ram()  # Get peak RAM usage
            compression_time = time.time() - compression_start_time
        except TimeoutError:
            compression_time = timeout_seconds  # Record timeout duration
            self.logger.warning("Compression timed out", 
                               timeout_seconds=timeout_seconds,
                               name=name, compressor_type=compressor_type)
            return None, compression_time, ram_usage
        except Exception as e:
            compression_time = time.time() - compression_start_time
            self.logger.error("Unexpected error during compression", 
                             name=name, compressor_type=compressor_type,
                             exception=e)
            return None, compression_time, ram_usage
        
        # Cache the result if valid (now includes RAM usage)
        if result is not None and result > 0:
            cache = get_global_cache()
            cache.set(compressor_type, params, self.input_file_path, result, compression_time, ram_usage)
        
        return result, compression_time, ram_usage
