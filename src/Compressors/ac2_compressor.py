import os
import shutil
import sys
from Compressors.base_compressor import BaseCompressor
from ga_constants import get_compression_timeout


class AC2Compressor(BaseCompressor):
    def __init__(self, input_file_path, reference_file_path, nr_models, temp="temp"):
        super().__init__(input_file_path, reference_file_path, temp)
        self.nr_models = nr_models  # Number of models

    def create_command(self, params_list, input_file_path):
        """Create AC2 compression command."""
        command = ['ac2/src/AC2', '-v', '-rm']

        for params in params_list:
            command.append(
                f"{params['ctx']}:{params['den']}:{params['hash']}:{params['gamma']}/"
                f"{params['mutations']}:{params['den_mut']}:{params['gamma_mut']}"
            )

        if not os.path.exists(input_file_path):
            print(f"Input file {input_file_path} does not exist.")
            return None

        if not os.path.exists(self.reference_file_path):
            print(f"Reference file {self.reference_file_path} does not exist.")
            return None

        command.append('-t')
        command.append('1')

        command.append('-r')
        command.append(self.reference_file_path)
        command.append(input_file_path)

        return command

    def evaluate(self, params_list, name):
        """Compress the input file using AC2 with caching and timeout."""
        # Get file size for timeout calculation
        file_size_mb = os.path.getsize(self.input_file_path) / (1024 * 1024)
        
        # AC2 can be slow, use extended timeout
        timeout_seconds = max(3600, get_compression_timeout('ac2', params_list[0], file_size_mb))
        
        result = self.evaluate_with_cache_and_timing(
            compressor_type='ac2',
            params=params_list[0],  # Extract first parameter set
            name=name,
            evaluate_func=lambda params, name: self._run_compression_with_ac2_setup(params_list, name),
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
    
    def _run_compression_with_ac2_setup(self, params_list, name):
        """Set up AC2 compression (handles file copying) and run compression."""
        try:
            if not os.path.exists(self.temp):
                os.makedirs(self.temp)

            copy_file_path = os.path.join(self.temp, f"{name}.txt")
            shutil.copyfile(self.input_file_path, copy_file_path)
            
            return self._run_compression_with_params(params_list, copy_file_path)
        except Exception as e:
            print(f"AC2 compression error for {name}: {e}")
            return None

    def _run_compression_with_params(self, params_list, input_file_path):
        """Run AC2 compression and compute compression ratio."""
        command = self.create_command(params_list, input_file_path)
        if command is None:
            return None

        if not self.run_command(command):  # Using BaseCompressor method
            return None

        compressed_file_path = input_file_path + '.co'

        if not self.wait_for_file(compressed_file_path):  # Using BaseCompressor method
            return None

        return self.compute_compression_ratio(self.input_file_path, compressed_file_path)  # Using BaseCompressor
