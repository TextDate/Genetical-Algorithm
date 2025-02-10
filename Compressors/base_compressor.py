import os
import subprocess
import sys
import time


class BaseCompressor:
    def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
        self.input_file_path = input_file_path  # Path to the file to be compressed
        self.reference_file_path = reference_file_path  # Reference file path (optional)
        self.temp = temp  # Temporary storage folder
        self.errors = 0
        self.nr_models = 1

        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp):
            os.makedirs(self.temp)

    def get_errors(self):
        """Return the number of errors encountered during compression."""
        return self.errors

    def erase_temp_files(self):
        """Remove all temporary files from the temp directory."""
        for filename in os.listdir(self.temp):
            os.remove(os.path.join(self.temp, filename))

    def run_command(self, command):
        """Execute a system command safely and handle errors."""
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            self.errors += 1
            print(f"Error running command: {' '.join(command)}")
            sys.stdout.flush()
            return False
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {' '.join(command)}")
            sys.stdout.flush()
            return False
        return True

    def wait_for_file(self, file_path, timeout=10):
        """Wait until a file exists (useful for waiting for compression output)."""
        elapsed_time = 0
        while not os.path.exists(file_path) and elapsed_time < timeout:
            print(f"Waiting for {file_path}...")
            sys.stdout.flush()
            time.sleep(2)
            elapsed_time += 2
        return os.path.exists(file_path)

    def compute_compression_ratio(self, original_file, compressed_file):
        """Compute compression ratio: (original size / compressed size)."""
        if not os.path.exists(original_file) or not os.path.exists(compressed_file):
            print("Compression size error: missing file(s).")
            return None
        original_size = os.path.getsize(original_file)
        compressed_size = os.path.getsize(compressed_file)
        return original_size / compressed_size if compressed_size > 0 else None
