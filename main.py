import os
import argparse
import json
from genetic_algorithm import GeneticAlgorithm
from Compressors.zstd_compressor import ZstdCompressor
from Compressors.ac2_compressor import AC2Compressor


def load_parameters(file_path):
    """Load parameters from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter file '{file_path}' not found.")

    with open(file_path, 'r') as file:
        return json.load(file)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a genetic algorithm for Zstd or AC2 compression.')

    # Compressor Choice
    parser.add_argument('--compressor', type=str, choices=['zstd', 'ac2'], required=True,
                        help="Choose the compressor: 'zstd' or 'ac2'")

    # Parameters file path
    parser.add_argument('--param_file', '-p', type=str, required=True, help="JSON file containing compressor parameter ranges")

    # AC2 specific parameters
    parser.add_argument('--models', type=int, default=1, help="Number of models to use when using AC2 compressor (default: 1)")
    parser.add_argument('--reference', '-r', type=str, required=False, help="Reference file (required for AC2 compressor)")

    # File to compress
    parser.add_argument('--input', '-i', type=str, required=True, help="File to compress")

    # Directories to be created
    parser.add_argument('--output_dir', '-o', type=str, default="ga_results_all_files", help="Folder to store the CSV results (default: 'ga_results_all_files')")
    parser.add_argument('--temp_dir', '-t', type=str, required=False, help="Folder to store temporary files")

    # Multithreading
    parser.add_argument('--max_threads', '-mt', type=int, default=16, help="Maximum number of threads to use.")

    # GA parameters
    parser.add_argument('--generations', type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument('--population_size', type=int, default=100, help="Population size (default: 100)")
    parser.add_argument('--mutation_rate', type=float, default=0.01, help="Mutation rate (default: 0.01)")
    parser.add_argument('--crossover_rate', type=float, default=0.5, help="Crossover rate (default: 0.5)")

    args = parser.parse_args()

    # Load parameters from the file
    parameters = load_parameters(args.param_file)

    # Select the appropriate parameter set
    if args.compressor == 'zstd':
        param_ranges = parameters.get('zstd', {})
        compressor = ZstdCompressor(args.input, args.reference, temp=args.temp_dir)
    else:
        if not args.reference:
            raise ValueError("A reference file must be provided when using the AC2 compressor.")
        param_ranges = parameters.get('ac2', {})
        compressor = AC2Compressor(args.input, args.reference, nr_models=args.models, temp=args.temp_dir)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        param_ranges, compressor,
        args.population_size, args.generations,
        args.mutation_rate, args.crossover_rate,
        output_dir=args.output_dir, max_threads=args.max_threads
    )

    # Run the genetic algorithm and get the best solution
    best_solution, best_fitness = ga.run()
    print(f"Best parameters found: {best_solution} with fitness {best_fitness}")


if __name__ == "__main__":
    main()
