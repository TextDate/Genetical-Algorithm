import brotli
import os
from Compressors.base_compressor import BaseCompressor

class BrotliCompressor(BaseCompressor):
    def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
        super().__init__(input_file_path, reference_file_path, temp)

    def create_command(self, params):
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
        """Evaluate Brotli compression efficiency (Compression Ratio)."""
        compression_ratio = self._run_compression_with_params(params_list, name)
        return compression_ratio

    def _run_compression_with_params(self, params, name):
        """Compress a file using Brotli and compute compression ratio."""
        compressed_file_path = os.path.join(self.temp, f"{name}.br")
        compressor = self.create_command(params[0])

        try:
            with open(self.input_file_path, 'rb') as input_file, open(compressed_file_path, 'wb') as output_file:
                output_file.write(compressor.process(input_file.read()) + compressor.flush())
        except Exception as e:
            print(f"Compression error: {e}")
            return None

        return self.compute_compression_ratio(self.input_file_path, compressed_file_path)
