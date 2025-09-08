import os
import sys
from Compressors.base_compressor import BaseCompressor  # Import BaseCompressor

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
        Evaluate the compression efficiency using PAQ8 with timing.
        Only the first params set is used since PAQ8 is single-run.
        """
        import time
        start_time = time.time()
        
        try:
            compression_ratio = self._run_compression_with_params(params_list[0], name)
            compression_time = time.time() - start_time
            return compression_ratio, compression_time
        except Exception as e:
            compression_time = time.time() - start_time
            raise e  # Re-raise the exception after recording time

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