import shutil
import sys
import time
import zstandard as zstd
import os

# DEPRECATED
class Compressor:
    def __init__(self, input_file_path, reference_file_path, temp="temp"):
        self.input_file_path = input_file_path  # Path of the file to be compressed
        self.reference_file_path = reference_file_path  # Path to the Reference File
        self.temp = temp
        self.ref_file_compress_value = self.get_file_size()
        

    def evaluate(self, params_list, name):
        """
        Use Zstd compression with the given parameters.
        Returns a fitness score based on the compression ratio and bits per symbol.
        """
        compression_distance = self._run_compression_with_params(params_list, name)
        return compression_distance
        
        #compression_ratio, compression_distance = self._run_compression_with_params(params_list, name)
        #return compression_ratio, compression_distance
    
    def erase_temp_files(self):
        for filename in os.listdir(self.temp):
            os.remove(self.temp + "/" + filename)
            
    def update_input_file_path(self, new_input_file_path):
        self.input_file_path = new_input_file_path
        
    def get_file_size(self):
        init = time.time()
        cctx = self.create_command({'level': 23, 'window_log': 12, 'chain_log': 12, 'hash_log': 12, 'search_log': 6, 'min_match': 5, 'target_length': 32768, 'strategy': 3})
        ref_compressed_file_path = self.temp + "/" + self.reference_file_path.split("/")[-1] + ".zst"
        with open(self.reference_file_path, 'rb') as input_file, open(ref_compressed_file_path, 'wb') as output_file:
            cctx.copy_stream(input_file, output_file)
        
        print(f"Took {time.time() - init} seconds to compress {self.reference_file_path}")
        sys.stdout.flush()
        
        size = os.path.getsize(ref_compressed_file_path)  
        os.remove(ref_compressed_file_path)
        return size
            
    def create_command(self, params:dict):
        try:
            compression_params = zstd.ZstdCompressionParameters(
                compression_level=params['level'],  # Compression level
                window_log=params['window_log'],  # Window size
                chain_log=params['chain_log'],  # Chain log size
                hash_log=params['hash_log'],  # Hash log size
                search_log=params['search_log'],  # Search log
                min_match=params['min_match'],  # Minimum match length
                target_length=params['target_length'],  # Target length for match optimization
                strategy=params['strategy']  # Compression strategy
            )

            # ZstdCompressor with custom parameters
            cctx = zstd.ZstdCompressor(compression_params=compression_params)
            
        except Exception as e:
            print("Exception:", e)
            sys.stdout.flush()
            

        return cctx

    def concatenate_files(self, output_file_path):
        """
        Concatenate the reference file and the input file to create a new file.
        """
        try:
            if not os.path.exists(self.reference_file_path):
                print(f"Reference file {self.reference_file_path} does not exist.")
                sys.stdout.flush()
                raise FileNotFoundError(f"Reference file {self.reference_file_path} does not exist.")
            if not os.path.exists(self.input_file_path):
                print(f"Input file {self.input_file_path} does not exist.")
                sys.stdout.flush()
                raise FileNotFoundError(f"Input file {self.input_file_path} does not exist.")
            

            with open(self.reference_file_path, 'rb') as ref_file, open(self.input_file_path, 'rb') as input_file, open(output_file_path, 'wb') as output_file:
                shutil.copyfileobj(ref_file, output_file)  # Copy reference file content to the output
                shutil.copyfileobj(input_file, output_file)  # Append input file content to the output

            if not os.path.exists(output_file_path):
                print(f"Output file {output_file_path} was not created.")
                sys.stdout.flush()
                raise FileNotFoundError(f"Output file {output_file_path} was not created.")

        except Exception as e:
            print(f"Error concatenating files: {e}")
            sys.stdout.flush()
            raise
        

    def _run_compression_with_params(self, params, name):
        """Compress a file using Zstd with given parameters and return compression ratio and bits per symbol."""
        
        if not os.path.exists(self.temp):
            os.makedirs(self.temp)

        cctx = self.create_command(params=params[0])

        concatenated_file_path = os.path.join(self.temp, f"{name}_conc.txt")
        
        self.concatenate_files(concatenated_file_path)
        
        # Compress the conc file and write it to an output file
        conc_compressed_file_path = concatenated_file_path + ".zst"
        with open(concatenated_file_path, 'rb') as input_file, open(conc_compressed_file_path, 'wb') as output_file:
            cctx.copy_stream(input_file, output_file)
        
        # x -> file to compress
        # y -> reference file 
        # C(x) -> compress file x size
        # C(yx) -> Compress file that is concatenation of y with x
        # C(yx) - C(y)  

        concatenated_file_size = os.path.getsize(conc_compressed_file_path)
        if concatenated_file_size != 0 and self.ref_file_compress_value != 0 and concatenated_file_size > self.ref_file_compress_value:
            compression_distance = concatenated_file_size - self.ref_file_compress_value
        else:
            return 0        

        os.remove(concatenated_file_path)
        os.remove(conc_compressed_file_path)

        return compression_distance
