# config/ Directory

This directory contains all configuration files for the genetic algorithm optimization system. These files define system parameters, compression algorithm settings, batch processing configurations, and dependency requirements.

## Directory Structure

```
config/
├── params.json                  # Main GA and compressor parameter configuration
├── requirements.txt            # Python dependencies and package versions
├── batch_config_example.json   # Example batch processing configuration
├── batch_config_hpc.json      # HPC cluster batch configuration
├── batch_config_minimal.json  # Minimal batch configuration template
└── README.md                  # This file
```

## Configuration Files

### 1. params.json - Main Parameter Configuration

The primary configuration file defining genetic algorithm parameters and compression algorithm parameter ranges.

#### GA Configuration Section:
```json
{
    "ga_config": {
        "population_size": 50,        // Number of individuals in each generation
        "generations": 100,           // Maximum number of generations
        "mutation_rate": 0.1,         // Probability of mutation (0.0-1.0)
        "crossover_rate": 0.8,        // Probability of crossover (0.0-1.0)
        "offspring_ratio": 0.9,       // Ratio of offspring to population size
        "elite_ratio": 0.1,           // Ratio of elite individuals preserved
        "tournament_size": 3,         // Size of tournament selection
        "convergence_generations": 20, // Generations for convergence detection
        "convergence_threshold": 0.001, // Fitness improvement threshold
        "max_threads": 50             // Maximum number of worker threads
    }
}
```

#### Compressor Parameter Ranges:

**ZSTD Parameters:**
- `level`: Compression level (-50 to 22, negative values for ultra-fast mode)
- `window_log`: Window size logarithm (10-30, affects memory usage)
- `chain_log`: Hash chain depth (6-30, affects compression ratio)
- `hash_log`: Hash table size (6-30, affects speed vs. memory)
- `search_log`: Search algorithm intensity (1-10)
- `min_match`: Minimum match length (3-6)
- `target_length`: Target match length (4096-131072)
- `strategy`: Compression strategy (0-5)

**LZMA Parameters:**
- `preset`: Compression preset level (0-9)
- `dict_size`: Dictionary size in bytes (4KB-128MB)
- `lc`: Literal context bits (0-4)
- `lp`: Literal position bits (0-4)
- `pb`: Position bits (0-4)
- `mode`: Compression mode (0-2)
- `nice_len`: Nice length parameter (5-273)
- `mf`: Match finder algorithm (0-3)
- `depth`: Search depth (0-1000)

**Brotli Parameters:**
- `quality`: Compression quality (0-11)
- `window`: Window size (10-24)
- `block`: Block size (16-24)
- `mode`: Compression mode (0-2: generic, text, font)

**PAQ8 Parameters:**
- `memory`: Memory usage level (1-9)
- `mode`: Compression mode selection

**AC2 Parameters:**
- `ctx`: Context size (1-5)
- `den`: Denominator parameter (1-500)
- `hash`: Hash parameter (0-255)
- `gamma`: Learning rate (0.01-0.99)
- `mutations`: Number of mutations (1-5)
- `den_mut`: Denominator mutation parameter (1-10)

### 2. requirements.txt - Python Dependencies

Defines all Python packages and their versions required for the system:

```
numpy>=1.21.0            # Numerical computing
scipy>=1.7.0             # Scientific computing
matplotlib>=3.5.0        # Basic plotting
seaborn==0.13.2          # Advanced statistical visualization
pandas>=1.3.0            # Data manipulation
psutil>=5.8.0            # System monitoring
lzma                     # LZMA compression (built-in)
brotli>=1.0.9            # Brotli compression
zstandard>=0.15.0        # ZSTD compression
scikit-learn>=1.0.0      # Machine learning for recommendations
```

**Key Dependencies:**
- **seaborn**: Statistical visualization for analysis plots
- **psutil**: System resource monitoring
- **zstandard**: High-performance ZSTD compression
- **scikit-learn**: ML models for compressor recommendations

### 3. Batch Configuration Files

#### batch_config_example.json - Comprehensive Example
Complete example showing all available batch processing options:
- Multiple dataset configurations
- Compressor selection per dataset
- Custom GA parameters per job
- Resource allocation settings
- Advanced scheduling options

#### batch_config_hpc.json - HPC Cluster Configuration
Optimized for high-performance computing environments:
- SLURM integration settings
- Resource-aware job scheduling
- Large-scale concurrent processing
- Memory and CPU optimization

#### batch_config_minimal.json - Minimal Template
Basic configuration template for simple batch jobs:
- Single dataset processing
- Default GA parameters
- Minimal resource requirements
- Quick start configuration

## Configuration Usage

### Loading Configuration:
```python
from ga_config import GAConfig

# Load from default params.json
config = GAConfig.from_file("config/params.json")

# Access GA parameters
population_size = config.population_size
max_threads = config.max_threads

# Access compressor parameters
zstd_params = config.get_compressor_params("zstd")
```

### Command Line Usage:
```bash
# Use default configuration
python src/main.py --compressor zstd --param_file config/params.json --input data.txt

# Override specific parameters
python src/main.py --compressor lzma --param_file config/params.json --input data.txt \
               --generations 200 --population_size 100

# Batch processing
python multi_domain_cli.py batch --config config/batch_config_example.json
```

## Parameter Optimization Guidelines

### GA Parameter Tuning:

**Population Size:**
- Small datasets: 20-50 individuals
- Medium datasets: 50-100 individuals
- Large datasets: 100+ individuals

**Generations:**
- Quick optimization: 20-50 generations
- Standard optimization: 50-100 generations
- Thorough optimization: 100+ generations

**Mutation Rate:**
- High diversity needed: 0.15-0.3
- Standard optimization: 0.05-0.15
- Fine-tuning: 0.01-0.05

**Crossover Rate:**
- Aggressive recombination: 0.8-0.95
- Balanced approach: 0.6-0.8
- Conservative approach: 0.4-0.6

### Thread Configuration:

**max_threads Recommendations:**
- Small files (< 10MB): 4-8 threads
- Medium files (10-100MB): 8-16 threads
- Large files (> 100MB): 16+ threads
- HPC clusters: Match allocated CPU cores

## Environment-Specific Configurations

### Development Environment:
```json
{
    "ga_config": {
        "population_size": 20,
        "generations": 30,
        "max_threads": 4
    }
}
```

### Production Environment:
```json
{
    "ga_config": {
        "population_size": 100,
        "generations": 200,
        "max_threads": 32
    }
}
```

### HPC Cluster Environment:
```json
{
    "ga_config": {
        "population_size": 200,
        "generations": 500,
        "max_threads": 64
    }
}
```

## Configuration Validation

The system includes comprehensive configuration validation:

### Parameter Range Checking:
- All numeric parameters validated against defined ranges
- Invalid values replaced with defaults or cause errors
- Warning messages for suboptimal configurations

### Dependency Validation:
- Required packages checked at startup
- Version compatibility verified
- Missing dependencies reported with installation instructions

### Resource Validation:
- Available memory checked against requirements
- CPU core count validation for thread settings
- Disk space verification for temporary files

## Best Practices

### Configuration Management:
1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for dev/test/prod
3. **Documentation**: Document custom parameter choices
4. **Backup**: Maintain backup configurations for critical setups

### Parameter Selection:
1. **Start Conservative**: Begin with default parameters
2. **Incremental Changes**: Modify parameters gradually
3. **Benchmark**: Test parameter changes on representative data
4. **Monitor**: Track performance impact of parameter changes

### Security Considerations:
1. **No Secrets**: Never store passwords or keys in configuration files
2. **Path Validation**: Validate all file paths in configurations
3. **Input Sanitization**: Sanitize all configuration inputs
4. **Access Control**: Restrict configuration file access appropriately

## Troubleshooting

### Common Configuration Issues:

**Invalid Parameter Ranges:**
- Check parameter values against documented ranges
- Verify JSON syntax is correct
- Ensure all required parameters are present

**Memory Issues:**
- Reduce population_size for large files
- Decrease max_threads if memory limited
- Monitor memory usage during execution

**Performance Problems:**
- Increase max_threads on multi-core systems
- Adjust convergence parameters for faster termination
- Use smaller parameter ranges for faster convergence

**Dependency Problems:**
- Verify all packages in requirements.txt are installed
- Check package versions for compatibility
- Use virtual environments to isolate dependencies