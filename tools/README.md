# tools/ Directory

This directory contains analysis, visualization, and utility tools for the genetic algorithm optimization system. These tools provide comprehensive analysis capabilities, data collection utilities, and visualization systems for understanding GA performance and results.

## Directory Structure

```
tools/
├── ga_performance_analyzer.py      # GA performance analysis and visualization
├── multi_domain_visualizer.py     # Multi-domain analysis and reporting
├── dataset_collector.py           # Dataset generation and collection utilities
├── chemistry_ga_analyzer.py       # Chemistry-specific GA analysis
├── chemistry_data_generator.py    # Chemistry dataset generation
├── code_quality_analyzer.py       # Code quality assessment and metrics
└── README.md                      # This file
```

## Analysis and Visualization Tools

### 1. GA Performance Analyzer (ga_performance_analyzer.py)

Comprehensive genetic algorithm performance analysis and visualization tool.

**Primary Functions:**
- **Fitness Evolution Analysis**: Track best fitness improvements across generations
- **Time Performance Analysis**: Monitor compression time evolution during optimization
- **Comparative Analysis**: Compare performance across different compressors
- **Statistical Reporting**: Generate detailed performance statistics

**Key Features:**
- Multi-compressor comparison plots
- Generation-by-generation analysis
- Time and fitness correlation analysis
- Statistical summary generation
- Multiple output formats (PNG, PDF, SVG)

**Usage:**
```bash
# Analyze all GA results in current directory
python tools/ga_performance_analyzer.py --input_pattern "ga_results_*" --output_dir analysis_output/

# Analyze specific compressor results
python tools/ga_performance_analyzer.py --results_dir ga_results_zstd/ --output_dir zstd_analysis/

# Generate comprehensive report
python tools/ga_performance_analyzer.py --input_pattern "*_ga_results" --create_report
```

**Generated Outputs:**
- `fitness_evolution.png`: Best fitness progression across generations
- `time_evolution.png`: Compression time trends during optimization
- `comparative_analysis.png`: Multi-compressor performance comparison
- `analysis_report.md`: Detailed text report with statistics
- `summary_statistics.csv`: Numerical summary of all metrics

### 2. Multi-Domain Visualizer (multi_domain_visualizer.py)

Advanced visualization system for multi-domain analysis and cross-domain performance evaluation.

**Primary Functions:**
- **Cross-Domain Analysis**: Performance comparison across different data types
- **Compressor Effectiveness**: Visualize compressor performance by domain
- **Statistical Analysis**: Advanced statistical visualization of results
- **Executive Reporting**: High-level summary visualizations

**Key Features:**
- Domain-specific performance analysis
- Compressor recommendation visualization
- Cross-domain correlation analysis
- Executive summary generation
- Interactive plot generation (if supported)

**Usage:**
```bash
# Generate multi-domain analysis from benchmark results
python tools/multi_domain_visualizer.py --results-file benchmark_results.json --output-dir visualizations/

# Create executive summary
python tools/multi_domain_visualizer.py --results-file results.json --executive-summary

# Custom domain analysis
python tools/multi_domain_visualizer.py --results-file results.json --domains text,scientific,multimedia
```

**Generated Outputs:**
- `domain_performance_heatmap.png`: Cross-domain performance matrix
- `compressor_effectiveness.png`: Compressor performance by domain
- `recommendation_analysis.png`: Compressor recommendation accuracy
- `executive_summary.png`: High-level performance overview
- `statistical_analysis.png`: Advanced statistical visualizations

## Data Collection and Generation Tools

### 3. Dataset Collector (dataset_collector.py)

Comprehensive dataset generation and collection utility for testing and benchmarking.

**Primary Functions:**
- **Multi-Domain Dataset Creation**: Generate datasets across different data types
- **Synthetic Data Generation**: Create test data with specific characteristics
- **Real Dataset Collection**: Organize and structure real-world datasets
- **Metadata Generation**: Create dataset metadata and descriptions

**Key Features:**
- 20+ data type generation (text, scientific, multimedia, etc.)
- Configurable dataset sizes (1MB to 1GB+)
- Realistic data pattern generation
- Metadata extraction and organization
- Dataset validation and quality checks

**Usage:**
```bash
# Create comprehensive multi-domain dataset collection
python tools/dataset_collector.py --create-collection --output-dir datasets/

# Generate specific data types
python tools/dataset_collector.py --types text,scientific --sizes 10MB,100MB --output-dir test_data/

# Organize existing datasets
python tools/dataset_collector.py --organize-existing --input-dir raw_data/ --output-dir organized_datasets/
```

**Generated Dataset Types:**
- **Text Data**: Literature, logs, documentation, code
- **Scientific Data**: DNA sequences, protein structures, chemical compounds
- **Multimedia Data**: Image files, audio samples, video streams
- **Structured Data**: CSV files, JSON data, XML documents
- **Binary Data**: Executables, compressed archives, random data

### 4. Chemistry GA Analyzer (chemistry_ga_analyzer.py)

Specialized analysis tool for chemistry-related genetic algorithm optimization.

**Primary Functions:**
- **Chemical Data Analysis**: Specialized metrics for chemical compound data
- **Molecular Compression**: Analysis of compression efficiency for molecular data
- **Chemical Pattern Recognition**: Identify patterns in chemical data compression
- **Domain-Specific Reporting**: Chemistry-focused performance reports

**Key Features:**
- SDF (Structure Data Format) file analysis
- Molecular complexity metrics
- Chemical property correlation with compression
- Specialized visualization for chemical data

### 5. Chemistry Data Generator (chemistry_data_generator.py)

Generator for chemistry-specific test datasets and molecular data.

**Primary Functions:**
- **Molecular Structure Generation**: Create realistic molecular data
- **Chemical Format Support**: Generate SDF, MOL, PDB formats
- **Property-Based Generation**: Create molecules with specific properties
- **Scalable Dataset Creation**: Generate datasets from MB to GB sizes

**Key Features:**
- Realistic molecular structure generation
- Multiple chemical file format support
- Property-driven dataset creation
- Chemical database integration capabilities

## Code Quality and Maintenance Tools

### 6. Code Quality Analyzer (code_quality_analyzer.py)

Comprehensive code quality assessment and metrics generation tool.

**Primary Functions:**
- **Code Metrics Analysis**: Lines of code, complexity, maintainability
- **Dependency Analysis**: Module dependencies and coupling analysis
- **Documentation Coverage**: Assess documentation completeness
- **Quality Reporting**: Generate detailed quality reports

**Key Features:**
- Multiple code quality metrics
- Dependency graph generation
- Documentation coverage analysis
- Technical debt assessment
- Quality trend tracking

**Usage:**
```bash
# Analyze entire codebase
python tools/code_quality_analyzer.py --analyze-all --output-dir quality_reports/

# Specific module analysis
python tools/code_quality_analyzer.py --modules src/ga_components/ --detailed-report

# Generate quality metrics
python tools/code_quality_analyzer.py --metrics-only --output quality_metrics.json
```

## Tool Integration

### Workflow Integration:
The tools are designed to work together in analysis workflows:

1. **Data Generation**: Use `dataset_collector.py` to create test datasets
2. **GA Execution**: Run genetic algorithm optimization
3. **Performance Analysis**: Use `ga_performance_analyzer.py` for single-domain analysis
4. **Multi-Domain Analysis**: Use `multi_domain_visualizer.py` for cross-domain insights
5. **Quality Assessment**: Use `code_quality_analyzer.py` for system maintenance

### SLURM Integration:
Tools are compatible with HPC cluster environments:
```bash
# SLURM job for analysis
sbatch GA_plotter.sh

# Which runs:
python tools/ga_performance_analyzer.py --input_pattern "ga_results_*" --output_dir analysis_output/
```

## Output Formats and Visualization

### Supported Output Formats:
- **Images**: PNG, PDF, SVG for plots and visualizations
- **Data**: CSV, JSON for numerical results
- **Reports**: Markdown, HTML for comprehensive reports
- **Interactive**: HTML with embedded JavaScript (when supported)

### Visualization Libraries:
- **Matplotlib**: Core plotting functionality
- **Seaborn**: Statistical visualization and styling
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support

### Plot Types Generated:
- **Line Plots**: Evolution trends over generations
- **Heatmaps**: Cross-domain performance matrices
- **Bar Charts**: Compressor comparison metrics
- **Scatter Plots**: Correlation and relationship analysis
- **Box Plots**: Statistical distribution analysis

## Configuration and Customization

### Tool Configuration:
Most tools support configuration through:
- Command-line arguments
- Configuration files (JSON/YAML)
- Environment variables
- Runtime parameter adjustment

### Customization Options:
- **Plot Styling**: Color schemes, fonts, sizes
- **Output Formats**: Multiple format support
- **Analysis Metrics**: Configurable metrics and thresholds
- **Report Templates**: Customizable report formats

## Performance Considerations

### Scalability:
- **Large Datasets**: Efficient processing of GB-scale results
- **Memory Management**: Streaming processing for large files
- **Parallel Processing**: Multi-core utilization where applicable
- **Caching**: Result caching for repeated analysis

### Resource Requirements:
- **Memory**: 1-4GB for typical analysis tasks
- **CPU**: Single-core for most tasks, multi-core for large datasets
- **Storage**: Temporary space for intermediate results
- **Dependencies**: Python scientific stack (NumPy, Pandas, Matplotlib)

## Best Practices

### Analysis Workflow:
1. **Data Validation**: Verify input data integrity before analysis
2. **Incremental Analysis**: Start with small datasets for testing
3. **Result Verification**: Cross-check results with multiple tools
4. **Documentation**: Document analysis parameters and findings

### Visualization Guidelines:
1. **Clear Labels**: Ensure all plots have descriptive labels
2. **Consistent Styling**: Use consistent color schemes and fonts
3. **Appropriate Scale**: Choose appropriate axis scales and ranges
4. **Export Quality**: Use high-resolution exports for reports

### Error Handling:
1. **Input Validation**: Validate all input files and parameters
2. **Graceful Degradation**: Handle missing data gracefully
3. **Error Reporting**: Provide clear error messages and suggestions
4. **Recovery Options**: Offer alternative processing methods when possible

## Troubleshooting

### Common Issues:

**Missing Dependencies:**
```bash
pip install matplotlib seaborn pandas numpy
```

**Memory Issues:**
- Reduce dataset size for analysis
- Use streaming processing options
- Increase system memory allocation

**File Format Issues:**
- Verify CSV file format and encoding
- Check for corrupted result files
- Ensure proper file permissions

**Visualization Problems:**
- Check matplotlib backend configuration
- Verify display environment for GUI plots
- Use file output for headless environments

### Performance Optimization:

**Large Dataset Handling:**
- Use data chunking for very large files
- Enable parallel processing where available
- Consider data sampling for initial analysis

**Memory Optimization:**
- Process data in chunks
- Clear intermediate results
- Use memory-efficient data structures

**Speed Optimization:**
- Cache intermediate results
- Use vectorized operations
- Enable multi-core processing