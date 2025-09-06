import lzma
import os
from Compressors.base_compressor import BaseCompressor
from ga_constants import get_compression_timeout, CHUNK_SIZE
from ga_logging import get_logger


class LzmaCompressor(BaseCompressor):
    def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
        super().__init__(input_file_path, reference_file_path, temp)
        self.logger = get_logger("LZMA")

    @staticmethod
    def create_command(params):
        """Create an LZMA compressor with correctly formatted filters."""
        # Parse combined lc_lp parameter (format: "lc,lp")
        try:
            lc_str, lp_str = params["lc_lp"].split(",")
            lc = int(lc_str)
            lp = int(lp_str)
        except (KeyError, ValueError, AttributeError):
            logger = get_logger("LZMA")
            logger.error("Invalid lc_lp parameter format", 
                        lc_lp=params.get('lc_lp', 'N/A'))
            return None

        # Ensure valid values for `mode`
        mode_mapping = {0: lzma.MODE_FAST, 1: lzma.MODE_NORMAL}
        mode = mode_mapping.get(params["mode"], lzma.MODE_NORMAL)

        # Ensure valid match finder
        match_finder_mapping = {
            "bt2": lzma.MF_BT2, "bt3": lzma.MF_BT3, "bt4": lzma.MF_BT4,
            "hc3": lzma.MF_HC3, "hc4": lzma.MF_HC4
        }
        match_finder = match_finder_mapping.get(params["mf"], lzma.MF_BT4)

        try:
            return lzma.LZMACompressor(filters=[
                {
                    "id": lzma.FILTER_LZMA2,
                    "dict_size": params["dict_size"],  # Must be power of 2
                    "lc": lc,  # Parsed from lc_lp
                    "lp": lp,  # Parsed from lc_lp
                    "pb": params["pb"],  # Must be 0-4
                    "mode": mode,  # Ensure correct mode
                    "mf": match_finder,  # Ensure correct match finder
                    "nice_len": params["nice_len"],  # Must be 2-273
                    "depth": params["depth"]  # Search depth
                }
            ])
        except Exception as e:
            logger = get_logger("LZMA")
            logger.error("Error creating LZMA compressor", 
                        exception=e, params=params)
            return None

    def evaluate(self, params_list, name):
        """Evaluate LZMA compression efficiency with caching and timeout."""
        # Get appropriate timeout based on LZMA complexity
        timeout_seconds = get_compression_timeout('lzma', params_list[0])
        
        return self.evaluate_with_cache(
            compressor_type='lzma',
            params=params_list[0],  # Extract first parameter set
            name=name,
            evaluate_func=lambda params, name: self._run_compression_with_params(params_list, name),
            timeout_seconds=timeout_seconds
        )

    def _run_compression_with_params(self, params, name):
        """Compress a file using LZMA and compute compression ratio."""
        compressed_file_path = os.path.join(self.temp, f"{name}.xz")
        compressor = self.create_command(params[0])

        if compressor is None:
            self.logger.error("Invalid LZMA parameters", 
                             name=name, 
                             lc_lp=params[0].get('lc_lp', 'N/A'))
            return None  # Return None so genetic algorithm assigns min_fitness

        try:
            with open(self.input_file_path, 'rb') as input_file, open(compressed_file_path, 'wb') as output_file:
                # Stream compression in chunks to reduce memory usage
                chunk_size = CHUNK_SIZE  # Configurable chunk size
                while True:
                    chunk = input_file.read(chunk_size)
                    if not chunk:
                        break
                    compressed_chunk = compressor.compress(chunk)
                    if compressed_chunk:
                        output_file.write(compressed_chunk)
                
                # Write final chunk
                final_chunk = compressor.flush()
                if final_chunk:
                    output_file.write(final_chunk)
                    
        except Exception as e:
            self.logger.error("LZMA compression error", name=name, exception=e)
            return None  # Return None so genetic algorithm assigns min_fitness

        return self.compute_compression_ratio(self.input_file_path, compressed_file_path)
