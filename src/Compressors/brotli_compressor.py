import brotli
import os
from Compressors.base_compressor import BaseCompressor
from ga_constants import get_compression_timeout, CHUNK_SIZE
from ga_logging import get_logger


class BrotliCompressor(BaseCompressor):
    def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
        super().__init__(input_file_path, reference_file_path, temp)
        self.logger = get_logger("Brotli")

    @staticmethod
    def create_command(params):
        """Create a Brotli compressor with tunable parameters."""

        # Map mode strings to Brotli constants
        mode_mapping = {
            "generic": brotli.MODE_GENERIC,
            "text": brotli.MODE_TEXT,
            "font": brotli.MODE_FONT
        }

        # Ensure valid mode selection
        mode = mode_mapping.get(params["mode"], brotli.MODE_GENERIC)

        return brotli.Compressor(
            quality=params["quality"],
            lgwin=params["lgwin"],
            lgblock=params["lgblock"],
            mode=mode  # Use mapped integer value
        )

    def evaluate(self, params_list, name):
        """Evaluate Brotli compression efficiency with caching and timeout."""
        # Get file size for timeout calculation
        import os
        file_size_mb = os.path.getsize(self.input_file_path) / (1024 * 1024)
        
        # Get appropriate timeout based on Brotli complexity and file size
        timeout_seconds = get_compression_timeout('brotli', params_list[0], file_size_mb)
        
        return self.evaluate_with_cache(
            compressor_type='brotli',
            params=params_list[0],  # Extract first parameter set
            name=name,
            evaluate_func=lambda params, name: self._run_compression_with_params(params_list, name),
            timeout_seconds=timeout_seconds
        )

    def _run_compression_with_params(self, params, name):
        """Compress a file using Brotli with streaming to optimize memory usage."""
        compressed_file_path = os.path.join(self.temp, f"{name}.br")
        compressor = self.create_command(params[0])

        try:
            with open(self.input_file_path, 'rb') as input_file, open(compressed_file_path, 'wb') as output_file:
                # Stream compression in chunks to reduce memory usage
                chunk_size = CHUNK_SIZE  # Configurable chunk size
                while True:
                    chunk = input_file.read(chunk_size)
                    if not chunk:
                        break
                    compressed_chunk = compressor.process(chunk)
                    if compressed_chunk:
                        output_file.write(compressed_chunk)
                
                # Write final chunk
                final_chunk = compressor.flush()
                if final_chunk:
                    output_file.write(final_chunk)
                    
        except Exception as e:
            self.logger.error("Brotli compression error", exception=e)
            return None

        return self.compute_compression_ratio(self.input_file_path, compressed_file_path)
