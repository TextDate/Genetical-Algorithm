# Compressors/ Directory

This directory contains the compression algorithm implementations used by the genetic algorithm for parameter optimization. Each compressor implements a common interface defined by the BaseCompressor class, enabling seamless integration with the GA system.

## Directory Structure

```
Compressors/
├── To_Store/              # Storage for archived compressor implementations
├── base_compressor.py     # Abstract base class for all compressors
├── zstd_compressor.py     # ZSTD compression implementation
├── lzma_compressor.py     # LZMA compression implementation  
├── brotli_compressor.py   # Brotli compression implementation
├── paq8_compressor.py     # PAQ8 compression implementation
├── ac2_compressor.py      # AC2 compression implementation
└── README.md             # This file
```

## Supported Compression Algorithms

### 1. ZSTD Compressor (zstd_compressor.py)
- **Algorithm**: Zstandard compression developed by Meta
- **Parameters**: Compression level (1-22), window log size, hash log, chain log, search log, min match length, strategy
- **Strengths**: Excellent balance of speed and compression ratio
- **Use Cases**: General-purpose compression, real-time applications

### 2. LZMA Compressor (lzma_compressor.py)
- **Algorithm**: Lempel-Ziv-Markov chain algorithm
- **Parameters**: Compression level (0-9), format (XZ, LZMA, RAW), dictionary size, LC/LP/PB parameters
- **Strengths**: High compression ratios
- **Use Cases**: Archival, storage optimization, when compression ratio is priority

### 3. Brotli Compressor (brotli_compressor.py)
- **Algorithm**: Google's Brotli compression
- **Parameters**: Compression level (0-11), window size, block size, mode
- **Strengths**: Excellent for text compression, web optimization
- **Use Cases**: Web assets, text files, HTTP compression

### 4. PAQ8 Compressor (paq8_compressor.py)
- **Algorithm**: Context mixing compression
- **Parameters**: Memory usage levels, context mixing parameters
- **Strengths**: Extremely high compression ratios
- **Use Cases**: Maximum compression scenarios, research applications

### 5. AC2 Compressor (ac2_compressor.py)
- **Algorithm**: Arithmetic coding with context modeling
- **Parameters**: Model count, compression mode, reference file support
- **Strengths**: Adaptive compression, reference-based compression
- **Use Cases**: DNA sequences, repetitive data with known patterns

## Base Compressor Architecture

### BaseCompressor Class (base_compressor.py)

The `BaseCompressor` class provides the foundation for all compression implementations:

#### Key Features:
- **Unified Interface**: Common methods across all compressors
- **Resource Management**: Automatic cleanup of temporary files
- **Timeout Protection**: File size-aware timeout scaling
- **Error Handling**: Robust error management with structured logging
- **Caching Integration**: Built-in support for compression result caching
- **Memory Monitoring**: Process memory tracking and warnings

#### Core Methods:
```python
def compress(self, parameters)           # Compress file with given parameters
def get_compression_ratio(self)          # Calculate compression ratio
def get_compression_time(self)           # Get compression execution time
def get_memory_usage(self)               # Get peak memory usage
def erase_temp_files(self)               # Clean up temporary files
```

#### Resource Management:
- Automatic temporary file cleanup
- Process monitoring and timeout enforcement
- Memory usage tracking with configurable limits
- Signal handling for graceful termination

## Implementation Requirements

When implementing a new compressor, you must:

1. **Inherit from BaseCompressor**:
   ```python
   from base_compressor import BaseCompressor
   
   class NewCompressor(BaseCompressor):
       def __init__(self, input_file_path, reference_file_path=None, temp="temp"):
           super().__init__(input_file_path, reference_file_path, temp)
   ```

2. **Implement Required Methods**:
   - `compress(parameters)`: Execute compression with GA-provided parameters
   - Parameter validation and encoding/decoding
   - Error handling and logging

3. **Handle Parameter Encoding**:
   - Convert GA bit strings to compressor-specific parameters
   - Validate parameter ranges and constraints
   - Provide meaningful error messages for invalid parameters

4. **Integration Points**:
   - Add parameter definitions to `config/params.json`
   - Register in the main compressor selection logic
   - Include in test suites and documentation

## Parameter Optimization

Each compressor optimizes different parameter sets:

### ZSTD Parameters:
- Compression level (1-22)
- Window log size (10-27)
- Hash log (6-26)
- Chain log (6-28)
- Search log (1-26)
- Min match length (3-7)
- Strategy (1-9)

### LZMA Parameters:
- Compression level (0-9)
- Dictionary size (4KB-128MB)
- LC (literal context bits, 0-4)
- LP (literal position bits, 0-4)
- PB (position bits, 0-4)

### Brotli Parameters:
- Quality level (0-11)
- Window size (10-24)
- Block size (16-24)
- Mode (generic, text, font)

## Performance Considerations

### Timeout Management:
- File size-aware timeout scaling implemented in base class
- Small files (< 10MB): Base timeout
- Medium files (10-100MB): 2-4x base timeout
- Large files (100MB+): 4-20x base timeout

### Memory Management:
- Process memory monitoring with configurable limits
- Warning threshold: 500MB
- Critical threshold: 1GB
- Automatic cleanup on memory pressure

### Caching:
- Compression results cached by input file hash and parameters
- Multiprocess-safe file-based cache
- LRU eviction for memory management

## Error Handling

### Common Error Scenarios:
- **File Not Found**: Input or reference files missing
- **Invalid Parameters**: Out-of-range or incompatible parameter values
- **Compression Failures**: Algorithm-specific errors
- **Timeout Exceeded**: Long-running compressions
- **Memory Exhaustion**: Resource limit exceeded

### Error Recovery:
- Graceful degradation for non-critical errors
- Automatic retry with adjusted parameters
- Comprehensive logging for debugging
- Resource cleanup on failures

## Testing and Validation

### Integration Testing:
- Parameter encoding/decoding validation
- Compression ratio and time measurement
- Memory usage tracking
- Error condition handling

### Performance Testing:
- Compression speed benchmarks
- Memory usage profiling
- Timeout behavior validation
- Cache efficiency measurement

## Adding New Compressors

To add a new compression algorithm:

1. **Create Implementation File**: `new_compressor.py`
2. **Inherit from BaseCompressor**: Follow established patterns
3. **Define Parameters**: Add to `config/params.json`
4. **Register Compressor**: Update selection logic in main.py
5. **Add Tests**: Include in test suite
6. **Update Documentation**: Add to this README and main documentation

## Dependencies

Each compressor may have specific external dependencies:
- **ZSTD**: python-zstandard package
- **LZMA**: Built into Python standard library
- **Brotli**: brotli package
- **PAQ8**: External PAQ8 binary
- **AC2**: External AC2 binary

Dependencies are managed through `config/requirements.txt` and validated during system initialization.