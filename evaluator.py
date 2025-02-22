import os
import sys
import pandas as pd
from Compressors.zstd_compressor import ZstdCompressor
from Extractors.zstd_extractor import ZSTDBestParametersExtractor  # Import the extraction class

class CompressionEvaluator:
    def __init__(self, base_directory, text_directory, temp_directory="temp_results"):
        """
        Initializes the evaluator with paths to parameter CSVs and text files.
        """
        self.base_directory = base_directory  # Path to optimized ZSTD results
        self.text_directory = text_directory  # Path to text dataset
        self.temp_directory = temp_directory  # Folder for compressed files

        self.best_parameters = ZSTDBestParametersExtractor(self.base_directory).get_best_parameters()
        os.makedirs(self.temp_directory, exist_ok=True)

    def get_largest_files(self, century, decade, num_files=100):
        """
        Retrieves the 10 largest text files from a given decade.
        """
        decade_path = os.path.join(self.text_directory, f"{century}_century", f"{decade}_decade")
        if not os.path.isdir(decade_path):
            return []

        # Get all text files and their sizes
        files = [(f, os.path.getsize(os.path.join(decade_path, f))) for f in os.listdir(decade_path) if f.endswith(".txt")]
        files.sort(key=lambda x: x[1], reverse=True)  # Sort by size (largest first)
        return [os.path.join(decade_path, f[0]) for f in files[:num_files]]

    def compress_text(self, text_file, compressor_params):
        """
        Compresses a text file using ZSTD with the given parameters and returns the compression ratio.
        """
        compressed_file = os.path.join(self.temp_directory, os.path.basename(text_file) + ".zst")
        
        # Use ZstdCompressor class
        compressor = ZstdCompressor(text_file)

        try:
            compression_ratio = compressor.evaluate([compressor_params], os.path.basename(text_file))
            return compression_ratio
        except Exception as e:
            print(f"Compression error for {text_file} with params {compressor_params}: {e}")
            sys.stdout.flush()
            return None

    def evaluate_compression(self):
        """
        Compresses the 10 largest files from each decade using every decade’s optimized parameters.
        Returns a dictionary storing all compression ratios.
        """
        results = {}

        for century in ["19", "20", "21"]:
            century_path = os.path.join(self.text_directory, f"{century}_century")

            if not os.path.isdir(century_path):
                continue  # Skip if century folder doesn't exist

            for decade_folder in sorted(os.listdir(century_path)):  # Sort decades in ascending order
                if not decade_folder.endswith("_decade"):
                    continue  # Skip unrelated files

                decade = decade_folder.split("_")[0]  # Extract YYYY (e.g., "1930")
                print(f"\n📂 **Processing {century}th century, decade {decade}**")
                sys.stdout.flush()
                
                largest_files = self.get_largest_files(century, decade)
                if not largest_files:
                    print(f"⚠️ No valid files found for {decade}. Skipping...")
                    sys.stdout.flush()
                    continue

                results[decade] = {}

                for param_decade, params in sorted(self.best_parameters.items(), key=lambda x: int(x[0])):  # Sort params by decade
                    print(f"  🔹 Testing {decade} texts with {param_decade} optimized parameters...")
                    sys.stdout.flush()

                    compression_ratios = [
                        self.compress_text(file, params) for file in largest_files
                    ]
                    
                    compression_ratios = [cr for cr in compression_ratios if cr is not None]  # Filter valid results

                    if compression_ratios:
                        results[decade][param_decade] = {
                            "mean_ratio": sum(compression_ratios) / len(compression_ratios),
                            "max_ratio": max(compression_ratios),
                            "min_ratio": min(compression_ratios)
                        }
                    else:
                        results[decade][param_decade] = {"mean_ratio": None, "max_ratio": None, "min_ratio": None}

        return results

    def analyze_results(self, results):
        """
        Analyzes the compression results:
        - Identifies if best parameters match their own decade.
        - Shows best-performing parameters per text decade.
        - Computes accuracy per decade.
        - Displays the ranking position of the real decade's parameters.
        """
        print("\n🔍 **Compression Results Analysis:**")
        sys.stdout.flush()
        match_count = 0
        total_decades = len(results)
        per_decade_accuracy = {}

        for text_decade in sorted(results.keys(), key=lambda x: int(x)):  # Sort results by decade
            compression_data = results[text_decade]
            sorted_decades = sorted(compression_data.items(), key=lambda x: x[1]["mean_ratio"] or 0, reverse=True)
            best_decade, best_data = sorted_decades[0]  # Get the best-performing decade parameters
            best_ratio = best_data["mean_ratio"]

            # Find ranking position of the real decade’s parameters
            real_decade_rank = next((i+1 for i, (decade, _) in enumerate(sorted_decades) if decade == text_decade), len(sorted_decades))

            print(f"\n📜 Texts from {text_decade}:")
            print(f"   ✅ Best compression using {best_decade}'s parameters (Avg Ratio: {best_ratio:.4f})")
            print(f"   🔍 Real decade's parameters ranked **{real_decade_rank}{self.ordinal(real_decade_rank)}** best")
            sys.stdout.flush()

            # Track accuracy per decade
            if text_decade not in per_decade_accuracy:
                per_decade_accuracy[text_decade] = {"matches": 0, "total": 0}

            per_decade_accuracy[text_decade]["total"] += 1
            if best_decade == text_decade:
                per_decade_accuracy[text_decade]["matches"] += 1
                match_count += 1

        # Print accuracy per decade
        print("\n📊 **Accuracy Per Decade:**")
        for decade in sorted(per_decade_accuracy.keys(), key=lambda x: int(x)):  # Sort decades
            acc = per_decade_accuracy[decade]
            accuracy = (acc["matches"] / acc["total"]) * 100
            print(f"🕰 **{decade}**: {accuracy:.2f}% match accuracy")
            sys.stdout.flush()

        overall_accuracy = (match_count / total_decades) * 100
        print(f"\n🎯 **Overall Accuracy: {overall_accuracy:.2f}%**")
        sys.stdout.flush()

    @staticmethod
    def ordinal(n):
        """Returns the ordinal suffix for a number (1st, 2nd, 3rd, etc.)."""
        return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

# Usage
evaluator = CompressionEvaluator(".", "../Dataset/32_char_alphabet/final_texts")
results = evaluator.evaluate_compression()
evaluator.analyze_results(results)
