import zstandard as zstd
import os
import sys
from Compressors.base_compressor import BaseCompressor  # Import BaseCompressor
from ga_constants import get_compression_timeout
from ga_logging import get_logger


class ZstdCompressor(BaseCompressor):
    def __init__(self, input_file_path, temp="temp"):
        super().__init__(input_file_path=input_file_path, temp=temp)
        self.logger = get_logger("ZSTD")

    @staticmethod
    def create_command(params):
        """Create a Zstd compressor with custom parameters."""
        try:
            compression_params = zstd.ZstdCompressionParameters(
                compression_level=params['level'],
                window_log=params['window_log'],
                chain_log=params['chain_log'],
                hash_log=params['hash_log'],
                search_log=params['search_log'],
                min_match=params['min_match'],
                target_length=params['target_length'],
                strategy=params['strategy']
            )
            return zstd.ZstdCompressor(compression_params=compression_params)
        except Exception as e:
            logger = get_logger("ZSTD")
            logger.error("Failed to create ZSTD compressor", exception=e)
            return None

    def evaluate(self, params_list, name):
        """Evaluate the compression efficiency using Zstd with caching and timeout."""
        # Get file size for timeout calculation
        import os
        file_size_mb = os.path.getsize(self.input_file_path) / (1024 * 1024)
        
        # Get appropriate timeout based on compression level and file size
        timeout_seconds = get_compression_timeout('zstd', params_list[0], file_size_mb)
        
        return self.evaluate_with_cache_and_timing(
            compressor_type='zstd',
            params=params_list[0],  # Extract first parameter set
            name=name,
            evaluate_func=lambda params, name: self._run_compression_with_params(params_list, name),
            timeout_seconds=timeout_seconds
        )

    def _run_compression_with_params(self, params, name):
        """Compress a file using Zstd and compute compression ratio with error handling."""
        try:
            cctx = self.create_command(params=params[0])
            if cctx is None:
                self.logger.error("Failed to create Zstd compressor", name=name)
                return None

            # Compress the input file
            compressed_file_path = os.path.join(self.temp, f"{name}.zst")
            
            try:
                with open(self.input_file_path, 'rb') as input_file:
                    with open(compressed_file_path, 'wb') as output_file:
                        cctx.copy_stream(input_file, output_file)
            except IOError as e:
                self.logger.error("IO error during compression", name=name, exception=e)
                return None
            except Exception as e:
                self.logger.error("Compression error", name=name, exception=e)
                return None

            # Verify compressed file was created
            if not os.path.exists(compressed_file_path):
                self.logger.error("Compressed file not created", name=name)
                return None

            # Calculate compression ratio
            compression_ratio = self.compute_compression_ratio(self.input_file_path, compressed_file_path)
            
            if compression_ratio is None:
                self.logger.error("Failed to calculate compression ratio", name=name)
                return None
                
            return compression_ratio
            
        except Exception as e:
            self.logger.error("Unexpected error in compression process", 
                             name=name, exception=e)
            return None
