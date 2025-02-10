import zstandard as zstd
import os
import sys
from Compressors.base_compressor import BaseCompressor  # Import BaseCompressor


class ZstdCompressor(BaseCompressor):
    def __init__(self, input_file_path, reference_file_path, temp="temp"):
        super().__init__(input_file_path, reference_file_path, temp)

    def create_command(self, params):
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
            print("Exception:", e)
            sys.stdout.flush()
            return None

    def evaluate(self, params_list, name):
        """Evaluate the compression efficiency using Zstd."""
        compression_ratio = self._run_compression_with_params(params_list, name)
        return compression_ratio

    def _run_compression_with_params(self, params, name):
        """Compress a file using Zstd and compute compression ratio."""


        cctx = self.create_command(params=params[0])
        if cctx is None:
            return None

        # Compress the input file
        compressed_file_path = os.path.join(self.temp, f"{name}.zst")
        with open(self.input_file_path, 'rb') as input_file, open(compressed_file_path, 'wb') as output_file:
            cctx.copy_stream(input_file, output_file)

        # Calculate compression ratio
        compression_ratio = self.compute_compression_ratio(self.input_file_path, compressed_file_path)

        return compression_ratio
