"""
Genetic Algorithm for Data Compression Parameter Optimization

This module provides a command-line interface for running genetic algorithms
to optimize compression parameters for various compression algorithms including
ZSTD, Brotli, LZMA, AC2, and PAQ8.

Features:
- Multi-compressor support (ZSTD, Brotli, LZMA, AC2, PAQ8)
- Advanced genetic algorithm with optimization techniques
- Configurable population size, generations, and parallel processing
- Comprehensive logging and result reporting
- Parameter space definition through JSON configuration

Usage:
    python main.py --compressor zstd --param_file config/params.json --input data.txt
"""

import os
import argparse
import json
import gzip
import shutil
from typing import Dict, Any, Optional

# Core GA system
from genetic_algorithm import GeneticAlgorithm
from ga_config import GAConfig
from ga_logging import setup_logging, get_logger

# Compressor implementations
from Compressors.zstd_compressor import ZstdCompressor
from Compressors.ac2_compressor import AC2Compressor
from Compressors.lzma_compressor import LzmaCompressor
from Compressors.brotli_compressor import BrotliCompressor
from Compressors.paq8_compressor import PAQ8Compressor

# Utilities
from cache import get_global_cache


def load_parameters(file_path: str) -> Dict[str, Any]:
    """
    Load compression parameters from a JSON configuration file.
    
    The JSON file should contain parameter ranges for each compressor type:
    {
        "zstd": {
            "level": [1, 3, 6, 9, 12],
            "window_log": [10, 12, 14, 16, 18]
        },
        "brotli": {
            "quality": [4, 6, 8, 10, 11]
        }
    }
    
    Args:
        file_path: Path to the JSON parameter configuration file
        
    Returns:
        Dictionary containing parameter configurations
        
    Raises:
        FileNotFoundError: If the parameter file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter file '{file_path}' not found.")

    try:
        with open(file_path, 'r') as file:
            parameters = json.load(file)
            return parameters
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in parameter file '{file_path}': {e}")


def main() -> None:
    """
    Main entry point for the genetic algorithm compression optimization system.
    
    Parses command-line arguments, initializes the appropriate compressor,
    configures the genetic algorithm, and runs the optimization process.
    Results are saved to the specified output directory.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a genetic algorithm for Zstd or AC2 compression.')

    # Compressor Choice
    parser.add_argument('--compressor',  '-c',type=str, choices=['zstd', 'ac2', 'lzma', 'brotli', 'paq8'], required=True,
                        help="Choose the compressor: 'zstd', 'ac2', 'lzma', 'brotli', or 'paq8'")

    # Parameters file path
    parser.add_argument('--param_file', '-p', type=str, required=True, help="JSON file containing compressor parameter ranges")

    # AC2 specific parameters
    parser.add_argument('--models', '-m', type=int, default=1, help="Number of models to use when using AC2 compressor (default: 1)")
    parser.add_argument('--reference', '-r', type=str, required=False, help="Reference file (required for AC2 compressor)")

    # File to compress
    parser.add_argument('--input', '-i', type=str, required=True, help="File to compress")

    # Directories to be created
    parser.add_argument('--output_dir', '-o', type=str, default="ga_results", help="Folder to store the CSV results (default: 'ga_results_all_files')")
    parser.add_argument('--temp_dir', '-t', type=str, default="temp", help="Folder to store temporary files")

    # Multithreading
    parser.add_argument('--max_threads', '-mt', type=int, default=16, help="Maximum number of threads to use.")

    # GA parameters
    parser.add_argument('--generations', '-g',type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument('--population_size', '-ps',type=int, default=100, help="Population size (default: 100)")
    parser.add_argument('--mutation_rate', '-mr',type=float, default=0.01, help="Mutation rate (default: 0.01)")
    parser.add_argument('--crossover_rate', '-mcr',type=float, default=0.8, help="Crossover rate (default: 0.5)")

    args = parser.parse_args()

    # Check if input file is Gzip-compressed (by magic number, not just .gz)
    input_file = args.input
    is_gzip = False
    try:
        with open(input_file, "rb") as f:
            magic = f.read(2)
            if magic == b'\x1f\x8b':  # Gzip magic number
                is_gzip = True
    except Exception as e:
        print(f"Could not read input file to check gzip: {e}")

    if is_gzip:
        # Create a dedicated folder for decompressed input files
        os.makedirs("input_files", exist_ok=True)

        filename = os.path.basename(input_file)[:-3]  # remove .gz
        decompressed_file = os.path.join("input_files", filename)

        try:
            with gzip.open(input_file, 'rb') as f_in:
                with open(decompressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Wait until the file exists and has a non-zero size
            import time
            for _ in range(10):
                if os.path.exists(decompressed_file) and os.path.getsize(decompressed_file) > 0:
                    break
                time.sleep(1)

            if os.path.exists(decompressed_file) and os.path.getsize(decompressed_file) > 0:
                logger.info("Input file decompressed", 
                           decompressed_file=os.path.abspath(decompressed_file))
                args.input = decompressed_file
            else:
                logger.warning("Decompressed file not found or empty", 
                               file=os.path.abspath(decompressed_file))
        except Exception as e:
            logger.error("Failed to decompress gzip file", exception=e)

    # Load parameters from the file
    parameters = load_parameters(args.param_file)

    # Select the appropriate parameter set
    if args.compressor == 'zstd':
        param_ranges = parameters.get('zstd', {})
        compressor = ZstdCompressor(args.input, temp=args.temp_dir)
    elif args.compressor == 'lzma':
        param_ranges = parameters.get('lzma', {})
        compressor = LzmaCompressor(args.input, temp=args.temp_dir)
    elif args.compressor == 'brotli':
        param_ranges = parameters.get('brotli', {})
        compressor = BrotliCompressor(args.input, temp=args.temp_dir)
    elif args.compressor == 'paq8':
        param_ranges = parameters.get('paq8', {})
        compressor = PAQ8Compressor(args.input, temp=args.temp_dir)


    else:
        if not args.reference:
            raise ValueError("A reference file must be provided when using the AC2 compressor.")
        param_ranges = parameters.get('ac2', {})
        compressor = AC2Compressor(args.input, args.reference, nr_models=args.models, temp=args.temp_dir)

    # Create configuration from CLI arguments
    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_threads=args.max_threads,
        output_dir=args.output_dir
    )
    
    # Setup logging
    logger = setup_logging(
        level="INFO",
        log_to_file=True,
        output_dir=config.output_dir,
        console_colors=True
    )
    
    # Log configuration
    logger.log_config_summary(config)

    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize genetic algorithm with config object
    ga = GeneticAlgorithm(param_ranges, compressor, config)

    # Clear cache and log initial stats
    cache = get_global_cache()
    cache.clear()
    logger.info("Starting genetic algorithm with enhanced error handling, logging, and parallel processing")

    # Run the genetic algorithm and get the best solution
    try:
        best_solution, best_fitness = ga.run()
        logger.info(f"GA completed successfully - Best fitness: {best_fitness:.4f}")
        logger.info(f"Best parameters: {best_solution}")
    except Exception as e:
        logger.critical("GA execution failed", exception=e)
        raise

    # Clean up the decompressed input file and folder if it was created
    if os.path.exists("input_files"):
        try: 
            os.rmdir("input_files")
        except OSError:
            pass
        except Exception as e:
            logger.warning("Could not delete decompressed file", exception=e)


if __name__ == "__main__":
    main()
