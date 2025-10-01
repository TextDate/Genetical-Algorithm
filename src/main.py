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
from pathlib import Path

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

# Multi-domain analysis components
from ga_components.file_analyzer import FileAnalyzer
from ga_components.compressor_recommender import CompressorRecommender, RecommendationContext

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
    parser = argparse.ArgumentParser(description='Run a genetic algorithm for compression optimization.')

    # Compressor Choice (with auto-recommendation option)
    parser.add_argument('--compressor',  '-c',type=str, choices=['zstd', 'ac2', 'lzma', 'brotli', 'paq8', 'auto'], 
                        help="Choose the compressor: 'zstd', 'ac2', 'lzma', 'brotli', 'paq8', or 'auto' for intelligent selection")
    
    # Intelligent compressor selection options
    parser.add_argument('--analyze-file', action='store_true',
                        help="Analyze file characteristics before compression")
    parser.add_argument('--recommend-compressor', action='store_true',
                        help="Get intelligent compressor recommendations")
    parser.add_argument('--optimization-target', choices=['ratio', 'speed', 'memory', 'balanced'], default='balanced',
                        help="Optimization target for compressor recommendation (default: balanced)")

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
    
    # Multi-objective evaluation parameters
    parser.add_argument('--disable_multi_objective', action='store_true', 
                        help="Disable multi-objective evaluation (use fitness only)")
    parser.add_argument('--fitness_weight', type=float, default=0.5,
                        help="Weight for compression ratio in multi-objective evaluation (default: 0.5)")
    parser.add_argument('--time_weight', type=float, default=0.3,
                        help="Weight for compression time in multi-objective evaluation (default: 0.3)")
    parser.add_argument('--ram_weight', type=float, default=0.2,
                        help="Weight for RAM usage in multi-objective evaluation (default: 0.2)")
    parser.add_argument('--disable_time_penalty', action='store_true',
                        help="Disable time penalty system (use linear time weighting only)")
    parser.add_argument('--time_penalty_threshold', type=float, default=10.0,
                        help="Time threshold in seconds for applying penalties (default: 10.0)")

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
                print(f"Input file decompressed: {os.path.abspath(decompressed_file)}")
                args.input = decompressed_file
            else:
                print(f"Warning: Decompressed file not found or empty: {os.path.abspath(decompressed_file)}")
        except Exception as e:
            print(f"Error: Failed to decompress gzip file: {e}")

    # Load parameters from the file
    parameters = load_parameters(args.param_file)

    # Handle file analysis and intelligent compressor selection
    if args.analyze_file or args.recommend_compressor or args.compressor == 'auto':
        analyzer = FileAnalyzer()
        try:
            characteristics = analyzer.analyze_file(args.input)
            
            if args.analyze_file:
                print(f"\nFile Analysis Results:")
                print(f"File Size: {characteristics.file_size:,} bytes")
                print(f"Data Type: {characteristics.data_type.value}")
                print(f"Data Domain: {characteristics.data_domain.value}")
                print(f"Entropy: {characteristics.entropy:.3f}")
                print(f"Predicted Compressibility: {characteristics.predicted_compressibility}")
                print()
            
            if args.recommend_compressor or args.compressor == 'auto':
                recommender = CompressorRecommender()
                context = RecommendationContext(optimization_target=args.optimization_target)
                recommendations = recommender.recommend_compressor(args.input, context)
                
                if recommendations:
                    if args.recommend_compressor:
                        print(f"Compressor Recommendations (target: {args.optimization_target}):")
                        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                            print(f"{i}. {rec.compressor_name.upper()} (confidence: {rec.confidence_score:.2f})")
                            print(f"   Expected ratio: {rec.expected_compression_ratio:.2f}x")
                            print(f"   Reasoning: {', '.join(rec.reasoning[:2])}")
                        print()
                    
                    if args.compressor == 'auto':
                        # Use the top recommendation
                        selected_compressor = recommendations[0].compressor_name
                        print(f"Auto-selected compressor: {selected_compressor.upper()} (confidence: {recommendations[0].confidence_score:.2f})")
                        print(f"Reasoning: {', '.join(recommendations[0].reasoning[:3])}")
                        print()
                        args.compressor = selected_compressor
                
        except Exception as e:
            logger = get_logger("Main")
            logger.warning(f"File analysis failed: {e}")
            if args.compressor == 'auto':
                print("Auto-selection failed, defaulting to ZSTD")
                args.compressor = 'zstd'

    # Validate compressor selection
    if not args.compressor or args.compressor == 'auto':
        raise ValueError("No compressor selected. Use --compressor or enable auto-selection.")

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
    elif args.compressor == 'ac2':
        if not args.reference:
            raise ValueError("A reference file must be provided when using the AC2 compressor.")
        param_ranges = parameters.get('ac2', {})
        compressor = AC2Compressor(args.input, args.reference, nr_models=args.models, temp=args.temp_dir)
    else:
        raise ValueError(f"Unsupported compressor: {args.compressor}")

    # Create configuration from CLI arguments
    config = GAConfig.from_args(args)
    
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
