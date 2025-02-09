import os
import shutil
import sys
from base_compressor import BaseCompressor


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
        """Compress the input file using AC2."""
        if not os.path.exists(self.temp):
            os.makedirs(self.temp)

        copy_file_path = os.path.join(self.temp, f"{name}.txt")
        shutil.copyfile(self.input_file_path, copy_file_path)

        compression_ratio = self._run_compression_with_params(params_list, copy_file_path)
        if compression_ratio is None:
            print(f"Compression failed for {name}.")
            sys.stdout.flush()
            return None

        return compression_ratio

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
