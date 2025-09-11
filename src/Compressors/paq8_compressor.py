import os
import sys
from Compressors.base_compressor import BaseCompressor  # Import BaseCompressor
from ga_constants import get_compression_timeout

class PAQ8Compressor(BaseCompressor):
    def __init__(self, input_file_path, temp="temp"):
        super().__init__(input_file_path=input_file_path, temp=temp)

    @staticmethod
    def create_command(params, output_file="output.paq8px", input_file="None"):
        """
        Create the PAQ8 compression command.
        Expected params: {'level': int, 'input_file': str, 'temp': str, 'name': str}
        """
        level = params.get('level', 5)

        # Example command for paq8px with level
        command = ["paq8px", f"-{level}", output_file, input_file]
        
        return command
    

    def evaluate(self, params_list, name):
        """
        Evaluate the compression efficiency using PAQ8 with caching and timeout.
        Only the first params set is used since PAQ8 is single-run.
        """
        # Get file size for timeout calculation
        import os
        file_size_mb = os.path.getsize(self.input_file_path) / (1024 * 1024)
        
        # PAQ8 is very slow, use extended timeout
        timeout_seconds = max(7200, get_compression_timeout('paq8', params_list[0], file_size_mb))
        
        result = self.evaluate_with_cache_and_timing(
            compressor_type='paq8',
            params=params_list[0],  # Extract first parameter set
            name=name,
            evaluate_func=lambda params, name: self._run_compression_with_params(params_list[0], name),
            timeout_seconds=timeout_seconds
        )
        
        # Return (fitness, time, ram) tuple
        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            # Backward compatibility: add 0.0 for RAM if not available
            fitness, compression_time = result
            return fitness, compression_time, 0.0
        else:
            # Single value (fitness only)
            return result, 0.0, 0.0

    def _run_compression_with_params(self, params, name):
        """
        Compress the file using PAQ8 and return compression ratio.
        """

        command, output_file = self.create_command(params, output_file=f"{self.temp}/{name}.paq8px", input_file=self.input_file_path)
        success = self.run_command(command)

        # PAQ8 can be very slow, so extend timeout if needed
        if not success or not self.wait_for_file(output_file, timeout=7200):
            print("Compression failed or timed out.")
            return None

        compression_ratio = self.compute_compression_ratio(self.input_file_path, output_file)
        return compression_ratio