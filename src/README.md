# src/ Directory

This directory contains the core source code for the genetic algorithm optimization system. It houses the main execution logic, genetic algorithm implementation, and all supporting modules for compression parameter optimization.

## Directory Structure

```
src/
├── Compressors/         # Compression algorithm implementations
├── ga_components/       # Modular genetic algorithm components
├── cache.py            # Basic caching utilities
├── ga_config.py        # Configuration management and validation
├── ga_constants.py     # System constants and timeout scaling
├── ga_exceptions.py    # Custom exception hierarchy
├── ga_logging.py       # Structured logging system
├── genetic_algorithm.py # Main GA orchestrator
├── main.py             # CLI entry point and argument parsing
└── multiprocess_cache.py # Advanced multiprocess-safe caching
```

## Key Components

### Core Files

- **main.py**: Primary entry point for the application. Handles command-line argument parsing, configuration loading, and orchestrates the genetic algorithm execution.

- **genetic_algorithm.py**: Main genetic algorithm orchestrator that coordinates all GA operations including population initialization, evolution cycles, evaluation, selection, crossover, and mutation.

- **ga_config.py**: Centralized configuration management with parameter validation, type safety, and computed properties. Replaces scattered parameter passing with a unified configuration object.

### Configuration and Constants

- **ga_constants.py**: System-wide constants including file size-aware timeout scaling for compression operations, memory thresholds, and performance parameters.

- **ga_exceptions.py**: Custom exception hierarchy for structured error handling throughout the system.

### Caching System

- **cache.py**: Basic caching utilities for compression results.

- **multiprocess_cache.py**: Advanced multiprocess-safe caching system with LRU eviction and file-based persistence for sharing cache across worker processes.

### Logging

- **ga_logging.py**: Structured logging system with configurable levels, file and console output, and performance tracking capabilities.

## Usage

The genetic algorithm can be executed through the main entry point:

```bash
# Basic usage
python src/main.py --compressor zstd --param_file config/params.json --input data.txt

# With custom GA parameters
python src/main.py --compressor lzma --param_file config/params.json --input data.txt \
                   --generations 50 --population_size 30 --max_threads 4

# AC2 compressor with reference file
python src/main.py --compressor ac2 --param_file config/params.json --input data.txt \
                   --reference reference.txt --models 2
```

## Design Patterns

The source code follows several key design patterns:

- **Component-based Architecture**: Each GA operation is implemented as a separate, testable component in the ga_components/ directory.

- **Factory Pattern**: The CompressorRegistry provides dynamic compressor selection based on runtime parameters.

- **Configuration Object Pattern**: GAConfig centralizes all configuration management, replacing scattered parameter passing.

- **Strategy Pattern**: Multiple algorithms for selection, crossover, and mutation operations.

## Key Features

- **Multi-threaded Evaluation**: Dynamic thread scaling based on system resources
- **Adaptive Parameters**: Mutation and crossover rates adjust during evolution
- **Smart Caching**: Multiprocess-safe caching with LRU eviction
- **Resource Monitoring**: Memory usage tracking and timeout protection
- **Convergence Detection**: Early stopping with multiple criteria
- **Structured Logging**: Comprehensive logging for debugging and analysis

## Configuration

The system is configured through:
- Command-line arguments parsed in main.py
- JSON configuration files (config/params.json)
- GAConfig object for runtime parameter management
- Environment variables for system-level settings

## Error Handling

Robust error handling is implemented through:
- Custom exception hierarchy (ga_exceptions.py)
- Structured logging of errors and warnings
- Graceful degradation for non-critical failures
- Resource cleanup on errors

## Performance Considerations

- File size-aware timeout scaling prevents hangs on large datasets
- Memory monitoring with warnings at 500MB and critical alerts at 1GB
- Dynamic thread management based on available CPU cores
- Efficient caching reduces redundant compression operations