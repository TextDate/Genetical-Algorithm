import lzma
import os
from Compressors.base_compressor import BaseCompressor


class LzmaCompressor(BaseCompressor):
    def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
        super().__init__(input_file_path, reference_file_path, temp)

    @staticmethod
    def create_command(params):
        """Create an LZMA compressor with correctly formatted filters."""
        # Ensure valid values for `mode`
        if params["lc"] + params["lp"] > 4:
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
                    "lc": params["lc"],  # Must be 0-4
                    "lp": params["lp"],  # Must be 0-4
                    "pb": params["pb"],  # Must be 0-4
                    "mode": mode,  # Ensure correct mode
                    "mf": match_finder,  # Ensure correct match finder
                    "nice_len": params["nice_len"],  # Must be 2-273
                    "depth": params["depth"]  # Search depth
                }
            ])
        except Exception as e:
            print(f"Error creating LZMA compressor: {e} with params: {params}")
            return None

    def evaluate(self, params_list, name):
        """Evaluate LZMA compression efficiency (Compression Ratio)."""
        compression_ratio = self._run_compression_with_params(params_list, name)
        return compression_ratio

    def _run_compression_with_params(self, params, name):
        """Compress a file using LZMA and compute compression ratio."""
        compressed_file_path = os.path.join(self.temp, f"{name}.xz")
        compressor = self.create_command(params[0])

        if compressor is None:
            return 0

        try:
            with open(self.input_file_path, 'rb') as input_file, open(compressed_file_path, 'wb') as output_file:
                output_file.write(compressor.compress(input_file.read()) + compressor.flush())
        except Exception as e:
            print(f"Compression error: {e}")
            return 0

        return self.compute_compression_ratio(self.input_file_path, compressed_file_path)
