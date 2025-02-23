import sys
import pandas as pd
import matplotlib.pyplot as plt
import os


class EvolutionVisualizer:
    def __init__(self, csv_directory):
        """
        Initialize the visualizer with the directory containing the CSV files for each generation.
        """
        self.csv_directory = csv_directory
        self.data = self._load_all_csv()
        self.available_params = self._identify_parameters()  # Dynamically detect parameters
        self.compressor = csv_directory.split("_")[0]
        
        if not os.path.exists(self.compressor):
            os.makedirs(self.compressor)

    def _load_all_csv(self):
        """
        Load all CSV files from the specified directory and store them in a dictionary.
        The key is the generation number, and the value is the corresponding DataFrame.
        """
        data = {}

        # Get all generation CSV files and extract generation numbers
        csv_files = [f for f in os.listdir(self.csv_directory) if f.startswith('generation') and f.endswith('.csv')]

        for file in csv_files:
            file_path = os.path.join(self.csv_directory, file)

            # Extract generation number from the filename
            generation_number = int(file.split('_')[1].split('.')[0])

            # Load CSV and store in dictionary
            data[generation_number] = pd.read_csv(file_path)

        # Ensure dictionary is sorted by generation number
        return dict(sorted(data.items()))

    def _identify_parameters(self):
        """
        Dynamically detect available parameters in the first generation's DataFrame.
        Works for any compressor (Zstd, LZMA, Brotli, etc.).
        """
        if not self.data:
            print("No CSV files found to visualize.")
            return []

        df = next(iter(self.data.values()))  # Get the first DataFrame
        excluded_columns = ["Generation", "Rank", "Fitness", "Individual"]  # Exclude non-parameter columns
        return [col for col in df.columns if col not in excluded_columns]

    def plot_best_individuals(self):
        """
        Plot the best individual's fitness across generations.
        Ensures that each generation has a valid best individual.
        """
        best_fitness = []
        best_individuals = []
        generations = list(self.data.keys())

        for gen, df in self.data.items():
            df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")  # Convert to numeric
            df = df.dropna(subset=["Fitness"]).sort_values(by="Fitness", ascending=False)

            best_fitness.append(df.iloc[0]["Fitness"])
            best_individuals.append(df.iloc[0]["Individual"])

            print(f"Gen {gen}: Best Individual = {df.iloc[0]['Individual']}, Fitness = {df.iloc[0]['Fitness']}")

        # Save log of best individuals
        log_file = os.path.join(self.csv_directory, "best_individuals_log.txt")
        with open(log_file, "w") as f:
            for gen, ind, fit in zip(generations, best_individuals, best_fitness):
                f.write(f"Generation {gen}: {ind} - Fitness: {fit}\n")

        valid_generations = [gen for gen, fit in zip(generations, best_fitness) if fit is not None]
        valid_fitness = [fit for fit in best_fitness if fit is not None]

        plt.figure(figsize=(10, 6))
        plt.plot(valid_generations, valid_fitness, marker="o", linestyle="-", color="b", label="Best Fitness")
        plt.title("Best Fitness Across Generations")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.savefig(os.path.join(self.compressor, "best_fitness.png"))
        plt.close()



    def plot_average_fitness(self):
        """
        Plot the average fitness across generations.
        """
        avg_fitness = []
        generations = list(self.data.keys())

        for gen, df in self.data.items():

            df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")
            avg_fitness.append(df["Fitness"].mean())

        valid_generations = [gen for gen, fit in zip(generations, avg_fitness) if fit is not None]
        valid_fitness = [fit for fit in avg_fitness if fit is not None]

        plt.figure(figsize=(10, 6))
        plt.plot(valid_generations, valid_fitness, marker="o", linestyle="-", color="g", label="Average Fitness")
        plt.title("Average Fitness Across Generations")
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.grid(True)
        plt.savefig(os.path.join(self.compressor, "average_fitness.png"))
        plt.close()


# Example usage
visualiser = EvolutionVisualizer("2010_53750_ga_results/AC2_ga_results")  # Change to your actual results directory
visualiser.plot_best_individuals()
visualiser.plot_average_fitness()
