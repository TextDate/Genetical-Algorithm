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

    def _load_all_csv(self):
        """
        Load all CSV files from the specified directory and store them in a list of DataFrames.
        Each DataFrame corresponds to a generation.
        """
        data = []
        csv_files = sorted([f for f in os.listdir(self.csv_directory) if f.endswith('.csv')])
        for file in csv_files:
            if file.startswith('generation'):
                file_path = os.path.join(self.csv_directory, file)
                df = pd.read_csv(file_path)
                data.append(df)
        return data

    def plot_parameter_evolution(self):
        """
        Plot the evolution of each Zstd parameter across generations.
        """
        if not self.data:
            print("No CSV files found to visualize.")
            return

        # Extract parameter names that correspond to Zstd parameters from the first file
        zstd_params = ['level', 'window_log', 'chain_log', 'hash_log', 'search_log', 'min_match', 'target_length',
                       'strategy']

        # Plot evolution of each parameter
        for param in zstd_params:
            plt.figure(figsize=(10, 6))
            for i, df in enumerate(self.data):
                plt.scatter([i] * len(df), df[param], alpha=0.5, label=f"Gen {i + 1}" if i == 0 else "")

            plt.title(f'Evolution of {param} Across Generations')
            plt.xlabel('Generation')
            plt.ylabel(f'{param} Value')
            plt.grid(True)
            plt.show()

    def plot_best_individuals(self):
        """
        Plot the best individual's fitness across generations.
        """
        best_fitness = []
        generations = list(range(1, len(self.data) + 1))

        for df in self.data:
            best_fitness.append(df['Fitness'].max())  # Assuming higher fitness is better

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, marker='o', linestyle='-', color='b', label='Best Fitness')
        plt.title('Best Fitness Across Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

    def plot_average_fitness(self):
        """
        Plot the average fitness across generations.
        """
        avg_fitness = []
        generations = list(range(1, len(self.data) + 1))

        for df in self.data:
            avg_fitness.append(df['Fitness'].mean())  # Calculate the mean fitness for the generation

        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_fitness, marker='o', linestyle='-', color='g', label='Average Fitness')
        plt.title('Average Fitness Across Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.grid(True)
        plt.show()

    def plot_mutation_crossover(self):
        """Plot the number of mutations and crossovers per generation."""
        # Load mutation and crossover counts from CSV
        filename = os.path.join(self.csv_directory, "mutation_crossover_counts.csv")
        if not os.path.isfile(filename):
            print(f"No mutation and crossover CSV found at {filename}")
            return

        df = pd.read_csv(filename)
        generations = df['Generation']
        mutations = df['Mutations']
        crossovers = df['Crossovers']

        plt.figure(figsize=(10, 6))
        plt.plot(generations, mutations, marker='o', label="Mutations", color='red')
        plt.plot(generations, crossovers, marker='o', label="Crossovers", color='blue')
        plt.xlabel("Generation")
        plt.ylabel("Count")
        plt.title("Mutations and Crossovers per Generation")
        plt.legend()
        plt.grid(True)
        plt.show()


#visualiser = EvolutionVisualizer("ga_results_all_files")
#visualiser.plot_parameter_evolution()
#visualiser.plot_best_individuals()
#visualiser.plot_average_fitness()
#visualiser.plot_mutation_crossover()
