import pandas as pd
import os

class ZSTDBestParametersExtractor:
    def __init__(self, base_directory):
        """
        Initialize the extractor with the base directory containing all decade folders.
        """
        self.base_directory = base_directory

    def get_best_parameters(self):
        """
        Finds the best ZSTD compression parameters from the latest generation for each decade.
        Returns a dictionary where keys are decades and values are parameter dictionaries.
        """
        best_parameters = {}

        for decade_folder in os.listdir(self.base_directory):
            decade_path = os.path.join(self.base_directory, decade_folder)

            if not os.path.isdir(decade_path) or not decade_folder.endswith("_ga_results"):
                continue  # Skip non-decade folders

            zstd_path = os.path.join(decade_path, "ZSTD_ga_results")
            if not os.path.isdir(zstd_path):
                continue  # Skip if no ZSTD results

            # Get all generation CSVs
            generation_files = [f for f in os.listdir(zstd_path) if f.startswith("generation_") and f.endswith(".csv")]
            if not generation_files:
                continue  # Skip if no generations found

            # Identify the latest generation
            latest_gen_file = max(generation_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            latest_gen_path = os.path.join(zstd_path, latest_gen_file)

            # Read CSV and find the best individual (Rank = 1)
            df = pd.read_csv(latest_gen_path)
            best_row = df[df["Rank"] == 1].iloc[0]  # Get the first row where Rank = 1

            # Extract parameters, removing "_Gene1" suffix from column names
            params = {col.replace("_Gene1", ""): best_row[col] for col in df.columns if "_Gene1" in col}

            # Store parameters under the decade key
            best_parameters[decade_folder.split("_")[0]] = params

        return best_parameters

    def print_best_parameters(self, best_parameters):
        """
        Nicely prints the best compression parameters for each decade in ascending order.
        """
        print("\n" + "-" * 50)
        print("📌 Optimized ZSTD Compression Parameters by Decade")
        print("-" * 50)

        # Sort decades in ascending order
        for decade in sorted(best_parameters.keys(), key=lambda x: int(x)):
            params = best_parameters[decade]

            print(f"\n🕰 **Decade: {decade}**")
            print("-" * 50)
            for param, value in params.items():
                print(f"  🔹 {param.capitalize()}: {value}")

        print("\n" + "-" * 50)


