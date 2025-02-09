import csv
import os
import shutil
import subprocess
import sys
import time


class Compressor:
    def __init__(self, input_file_path, reference_file_path: str, nr_models, temp="temp"):
        self.input_file_path = input_file_path  # Path of the file to be compressed
        self.reference_file_path = reference_file_path  # Path to the Reference File
        self.nr_models = nr_models  # The Number of models each individual will have
        self.temp = temp  # Path to folder where to store the temporary files
        self.errors = 0
        self.concatenated_file_path = None
        self.ref_file_compress_value = None

        if not os.path.exists(self.temp):
            os.makedirs(self.temp)

        # Compress the reference file during initialization
        self._compress_reference_file()

    def get_errors(self):
        return self.errors

    def update_input_file_path(self, new_input_file_path):
        self.input_file_path = new_input_file_path

    def erase_temp_files(self):
        for filename in os.listdir(self.temp):
            os.remove(self.temp + "/" + filename)

    def create_command(self, params_list: dict, copy_file_path, rm=1):

        # Command creation to run AC2
        command = ['ac2/src/AC2', '-v']

        if rm:
            command.append('-rm')

        else:
            command.append('-tm')

        for params in params_list:
            command.append(
                f"{params['ctx']}:{params['den']}:{params['hash']}:{params['gamma']}/{params['mutations']}:{params['den_mut']}:{params['gamma_mut']}")

        if not os.path.exists(copy_file_path):
            print("Input file does not exist.")
            return 0

        command.append('-t')
        command.append('1')

        if rm:
            if not os.path.exists(self.reference_file_path):
                print("Reference file does not exist.")
                return 0

            command.append('-r')
            command.append(self.reference_file_path)

        command.append(copy_file_path)

        return command

    def _compress_reference_file(self):
        """
        Compress the reference file during initialization and store its compressed size.
        """

        copy_ref_file_path = os.path.join(self.temp, os.path.basename(self.reference_file_path))
        shutil.copyfile(self.reference_file_path, copy_ref_file_path)  # Copying the file to compress

        # Create the compression command
        # REVER ISTO, PARA CONSEGUIR PASSAR PARAMETROS DINAMICAMENTE
        stats_file = "Tester/reference_files_compression_stats.csv"
        params = [{'ctx': 2, 'den': 1, 'hash': 6, 'gamma': 0.01, 'mutations': 0, 'den_mut': 0, 'gamma_mut': 0}]
        params_string = f'{params[0]["ctx"]}:{params[0]["den"]}:{params[0]["hash"]}:{params[0]["gamma"]}/{params[0]["mutations"]}:{params[0]["den_mut"]}:{params[0]["gamma_mut"]}'

        # Read the CSV if it exists
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if the reference file with the same params is already in the CSV
                    if row["file"] == self.reference_file_path.split("/")[-1] and row["params"] == params_string:
                        self.ref_file_compress_value = int(row['size'])
                        print(f"Using cached size for {self.reference_file_path}: {self.ref_file_compress_value}")
                        sys.stdout.flush()
                        return  # Use the cached size

        command = self.create_command(params, copy_ref_file_path, rm=0)

        # Run compression
        try:
            # start_time = time.time()
            # print(f"Compressing {self.reference_file_path}")
            sys.stdout.flush()
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.errors += 1
            print(f"Error compressing reference file: {copy_ref_file_path}")  # , "\n", e.stderr)
            sys.stdout.flush()
        except subprocess.TimeoutExpired:
            print(f"Command timed out while compressing reference file: {command}")
            sys.stdout.flush()

        # print(f"Took: {(time.time() - start_time):0f} seconds to compress {self.reference_file_path}")

        # Wait until the compressed file exists
        while not os.path.exists(copy_ref_file_path + ".co"):
            # print(f"Waiting for compressed reference file {copy_ref_file_path}.co")
            sys.stdout.flush()
            time.sleep(2)

        # Store the compressed size
        self.ref_file_compress_value = os.path.getsize(copy_ref_file_path + ".co")
        # print(f"{self.reference_file_path} compressed: {self.ref_file_compress_value}")
        sys.stdout.flush()

    def concatenate_files(self, output_file_path):
        """
        Concatenate the reference file and the input file to create a new file.
        """
        try:
            if not os.path.exists(self.reference_file_path):
                raise FileNotFoundError(f"Reference file {self.reference_file_path} does not exist.")
            if not os.path.exists(self.input_file_path):
                raise FileNotFoundError(f"Input file {self.input_file_path} does not exist.")

            with open(self.reference_file_path, 'rb') as ref_file, open(self.input_file_path, 'rb') as input_file, open(
                    output_file_path, 'wb') as output_file:
                shutil.copyfileobj(ref_file, output_file)  # Copy reference file content to the output
                shutil.copyfileobj(input_file, output_file)  # Append input file content to the output

            if not os.path.exists(output_file_path):
                raise FileNotFoundError(f"Output file {output_file_path} was not created.")

        except Exception as e:
            print(f"Error concatenating files: {e}")
            sys.stdout.flush()
            raise

    def evaluate(self, params_list, name):
        """
        Use AC2 compressor with the given parameters.
        Returns a fitness score based on the compression ratio.
        """
        if not os.path.exists(self.temp):
            os.makedirs(self.temp)

        # Create temporary copy of the file
        copy_file_path = os.path.join(self.temp, f"{name}.txt")
        shutil.copyfile(self.input_file_path, copy_file_path)

        # Unique concatenated file path for each thread
        self.concatenated_file_path = os.path.join(self.temp, f"{name}_conc.txt")

        try:
            # Check input and reference file existence
            if not os.path.exists(self.input_file_path):
                print(f"Error: Input file {self.input_file_path} does not exist.")
                sys.stdout.flush()
                return None, None
            if not os.path.exists(self.reference_file_path):
                print(f"Error: Reference file {self.reference_file_path} does not exist.")
                sys.stdout.flush()
                return None, None

            # Concatenate files
            self.concatenate_files(self.concatenated_file_path)
            if not os.path.exists(self.concatenated_file_path):
                print(f"Error: Concatenated file {self.concatenated_file_path} was not created.")
                sys.stdout.flush()
                return None, None

        except Exception as e:
            print(f"Error during file concatenation: {e}")
            sys.stdout.flush()
            return None, None

        # Run compression and calculate metrics
        compression_ratio, compression_distance = self._run_compression_with_params(params_list, copy_file_path)

        if compression_ratio is None or compression_distance is None:
            print(f"Compression failed for block {name}.")
            sys.stdout.flush()
            return None, None

        return compression_ratio, compression_distance

    def _run_compression_with_params(self, params_list: dict, copy_file_path):
        """Compress a file using AC2 with given parameters and return compression ratio."""

        command = self.create_command(params_list, copy_file_path, rm=1)

        # print(command)
        # sys.stdout.flush()
        # Run AC2 with locking of the thread until the process is over (with reference file)
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.errors += 1
            print(f"Command failed with error at block:", self.input_file_path, )  # "\n",e.stderr)
            sys.stdout.flush()
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            sys.stdout.flush()

        # Path to the compressed version of the copy file
        compressed_file_path = copy_file_path + '.co'

        # Wait until the compressed file exists
        while not os.path.exists(compressed_file_path):
            print(f"Waiting for {compressed_file_path}")
            sys.stdout.flush()
            time.sleep(2)

        compressed_file_size = os.path.getsize(compressed_file_path)
        input_file_size = os.path.getsize(self.input_file_path)
        if input_file_size != 0 and compressed_file_size != 0:
            compression_ratio = input_file_size / compressed_file_size
        else:
            print("Compression sizes error")
            sys.stdout.flush()
            return 0

        command = self.create_command(params_list, self.concatenated_file_path, rm=0)

        # Run AC2 with locking of the thread until the process is over (without reference file)
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.errors += 1
            print(f"Command failed with error at concatenated file of:", self.reference_file_path, "and",
                  self.input_file_path, )  # "\n",e.stderr)
            sys.stdout.flush()
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            sys.stdout.flush()

        while not os.path.exists(self.concatenated_file_path + ".co"):
            print(f"Waiting for {self.concatenated_file_path}")
            sys.stdout.flush()
            time.sleep(2)

        # x -> file to compress
        # y -> reference file
        # C(x) -> compress file x size
        # C(yx) -> Compress file that is concatenation of y with x
        # C(yx) - C(y)

        concatenated_file_size = os.path.getsize(self.concatenated_file_path + ".co")
        if concatenated_file_size != 0 and self.ref_file_compress_value != 0 and concatenated_file_size > self.ref_file_compress_value:
            compression_distance = concatenated_file_size - self.ref_file_compress_value
        else:
            print("Compression ratio or distance was 0")
            sys.stdout.flush()
            return 0

            # print(compression_ratio, compression_distance)
        # sys.stdout.flush()
        return compression_ratio, compression_distance
