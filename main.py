import os
import argparse
import cProfile
import pstats

from genetic_algorithm import GeneticAlgorithm
from zstd_compressor import Compressor as zstdCompressor
from ac2_compressor import Compressor as ac2Compressor
from plot_maker import EvolutionVisualizer


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a genetic algorithm for Zstd or AC2 compression.')
    parser.add_argument('--compressor', type=str, choices=['zstd', 'ac2'], required=True, help="Choose the compressor: 'zstd' or 'ac2'")
    parser.add_argument('--models', type=int, default=1, help="Number of models to use when using AC2 compressor (default: 1)")
    parser.add_argument('--reference', '-r', type=str, required=False, help="Reference file (required for AC2 compressor)")
    parser.add_argument('--input', '-i', type=str, required=True, help="File to compress")
    parser.add_argument('--output_dir', '-o', type=str, default="ga_results_all_files", help="Folder to store the CSV results (default: 'ga_results_all_files')")
    parser.add_argument('--temp_dir', '-t', type=str, required=False, help="Folder to store the")
    parser.add_argument('--max_threads', '-mt', type=int, default=16, help="Maximum number of threads to use.")

    args = parser.parse_args()

    zstd_param_ranges = {
        'level': tuple(range(-50, 23)),  # Compression levels (-50 to 22)
        'window_log': tuple(range(10, 31)),  # Window log range (10 to 30)
        'chain_log': tuple(range(6, 31)),  # Chain log (6 to 30)
        'hash_log': tuple(range(6, 31)),  # Hash log (6 to 30)
        'search_log': tuple(range(1, 11)),  # Search log (1 to 10)
        'min_match': tuple(range(3, 7)),  # Minimum match length (3 to 6)
        'target_length': tuple(2 ** x for x in range(12, 18)),  # Target match length in power of 2 (up to 128 KB)
        'strategy': tuple(range(0, 6))  # Compression strategies (0 to 5)
    }

    ac2_param_ranges = {
        'ctx': tuple(range(1, 6)),  # Context size (1 to 5) 6 goes segmentation fault, does not even start
        'den': (1, 2, 5, 10, 20, 50, 100, 200, 500),  # Denominator values
        'hash': (0, 16, 32, 64, 128, 192, 255),  # Hash size values
        'gamma': (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),  # Gamma values
        'mutations': tuple(range(1, 6)),  # Number of mutations (1 to 5)
        'den_mut': tuple(range(1, 11)),  # Denominator with mutation
        'gamma_mut': (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),  # Gamma with mutation
    }

    # genetic_algorithm parameters
    generations = 100  # How many generations will be done
    population_size = 100  # How many samples will be created/stored per generation
    mutation_rate = 0.01  # Probability of a bit from a gene to mutate
    crossover_rate = 0.5  # Probability of a crossover to happen

    # Select compressor based on argument
    if args.compressor == 'zstd':
        param_ranges = zstd_param_ranges
        compressor = zstdCompressor(args.input)  # Initialize zstd compressor with the input file
    else:
        if not args.reference:
            raise ValueError("A reference file must be provided when using the AC2 compressor.")
        param_ranges = ac2_param_ranges
        compressor = ac2Compressor(args.input, args.reference, nr_models=args.models, temp=args.temp_dir)  # Initialize AC2 compressor

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize genetic_algorithm
    ga = GeneticAlgorithm(param_ranges, compressor, population_size, generations, mutation_rate, crossover_rate, output_dir=args.output_dir, max_threads=args.max_threads)
    # Run the genetic_algorithm and get the best solution
    best_solution, best_fitness = ga.run()
    print(f"Best parameters found: {best_solution} with fitness {best_fitness}")

    # Initialize the visualizer
    # csv_directory = args.output_dir # Directory containing CSV files for each generation
    # visualizer = EvolutionVisualizer(csv_directory)

    # Plot the evolution of parameters and best fitness values
    # visualizer.plot_parameter_evolution()
    # visualizer.plot_best_individuals()
    # visualizer.plot_average_fitness()
    # visualizer.plot_mutation_crossover()


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    main()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(236)  # Print the top 10 time-consuming functions
