"""
Multi-Domain Compression Analysis CLI

Command-line interface for advanced multi-domain compression analysis operations.
Provides access to file analysis, compressor recommendations, batch processing,
and comprehensive benchmarking capabilities.

Usage Examples:
    # Analyze a single file
    python multi_domain_cli.py analyze-file --input data.txt
    
    # Get compressor recommendations
    python multi_domain_cli.py recommend --input data.txt --target ratio
    
    # Run multi-domain benchmark
    python multi_domain_cli.py benchmark --dataset-dir datasets/ --compressors zstd lzma brotli
    
    # Batch process multiple datasets  
    python multi_domain_cli.py batch --config batch_config.json
    
    # Create dataset collection
    python multi_domain_cli.py create-datasets --output-dir datasets/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Multi-domain components
from ga_components.file_analyzer import FileAnalyzer, DataType, DataDomain
from ga_components.compressor_recommender import CompressorRecommender, RecommendationContext
from ga_components.multi_domain_benchmarker import MultiDomainBenchmarker, MultiDomainBenchmarkConfig
from ga_components.batch_processor import BatchProcessor, BatchProcessingConfig
from ga_components.cross_domain_evaluator import CrossDomainEvaluator
from ga_config import GAConfig
from ga_logging import setup_logging, get_logger

# Dataset collector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from dataset_collector import MultiDomainDatasetCollector


def setup_cli_logging(level: str = "INFO", output_dir: Optional[str] = None) -> None:
    """Setup logging for CLI operations."""
    setup_logging(
        level=level,
        log_to_file=bool(output_dir),
        output_dir=output_dir or "cli_logs",
        console_colors=True
    )


def cmd_analyze_file(args) -> None:
    """Analyze a single file and display characteristics."""
    logger = get_logger("CLI-AnalyzeFile")
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    logger.info(f"Analyzing file: {args.input}")
    
    analyzer = FileAnalyzer()
    try:
        characteristics = analyzer.analyze_file(args.input)
        
        # Display results
        print(f"\nFile Analysis Results for: {args.input}")
        print("=" * 60)
        print(f"File Size: {characteristics.file_size:,} bytes ({characteristics.file_size / (1024*1024):.2f} MB)")
        print(f"Data Type: {characteristics.data_type.value}")
        print(f"Data Domain: {characteristics.data_domain.value}")
        print(f"Entropy: {characteristics.entropy:.3f}")
        print(f"Text Ratio: {characteristics.text_ratio:.3f}")
        print(f"Repetition Factor: {characteristics.repetition_factor:.3f}")
        print(f"Predicted Compressibility: {characteristics.predicted_compressibility}")
        
        if characteristics.mime_type:
            print(f"MIME Type: {characteristics.mime_type}")
        
        if characteristics.detected_encoding:
            print(f"Detected Encoding: {characteristics.detected_encoding}")
        
        print(f"\nRecommended Compressors:")
        for i, compressor in enumerate(characteristics.recommended_compressors, 1):
            print(f"  {i}. {compressor}")
        
        # Save detailed analysis if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(characteristics.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Detailed analysis saved to: {output_path}")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def cmd_recommend_compressor(args) -> None:
    """Generate compressor recommendations for a file."""
    logger = get_logger("CLI-Recommend")
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    logger.info(f"Generating compressor recommendations for: {args.input}")
    
    # Create recommendation context
    context = RecommendationContext(
        optimization_target=args.target,
        max_time_budget=args.max_time,
        memory_constraints=args.max_memory,
        quality_requirements=args.quality
    )
    
    recommender = CompressorRecommender()
    try:
        recommendations = recommender.recommend_compressor(args.input, context)
        
        if not recommendations:
            print("No recommendations could be generated for this file.")
            return
        
        print(f"\nCompressor Recommendations for: {args.input}")
        print("=" * 60)
        print(f"Optimization Target: {context.optimization_target}")
        
        if context.max_time_budget:
            print(f"Time Budget: {context.max_time_budget}s")
        if context.memory_constraints:
            print(f"Memory Limit: {context.memory_constraints}MB")
        
        print("\nRecommendations (ranked by confidence):")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.compressor_name.upper()}")
            print(f"   Confidence: {rec.confidence_score:.2f}")
            print(f"   Expected Compression Ratio: {rec.expected_compression_ratio:.2f}x")
            print(f"   Expected Time: {rec.expected_compression_time:.2f}s")
            print(f"   Expected Memory: {rec.expected_memory_usage:.0f}MB")
            print(f"   Reasoning:")
            for reason in rec.reasoning:
                print(f"     - {reason}")
            print()
        
        # Save detailed recommendations if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            recommendations_data = {
                'file_path': args.input,
                'context': {
                    'optimization_target': context.optimization_target,
                    'max_time_budget': context.max_time_budget,
                    'memory_constraints': context.memory_constraints,
                    'quality_requirements': context.quality_requirements
                },
                'recommendations': [rec.to_dict() for rec in recommendations],
                'summary': recommender.get_recommendation_summary(recommendations)
            }
            
            with open(output_path, 'w') as f:
                json.dump(recommendations_data, f, indent=2, default=str)
            
            logger.info(f"Detailed recommendations saved to: {output_path}")
    
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        sys.exit(1)


def cmd_multi_domain_benchmark(args) -> None:
    """Run comprehensive multi-domain benchmark."""
    logger = get_logger("CLI-Benchmark")
    
    if not os.path.exists(args.dataset_dir):
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    logger.info(f"Starting multi-domain benchmark on: {args.dataset_dir}")
    
    # Create GA config
    ga_config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        max_threads=args.max_threads
    )
    
    # Create benchmark configuration
    benchmark_config = MultiDomainBenchmarkConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        compressors=args.compressors,
        file_patterns=args.file_patterns,
        max_concurrent_benchmarks=args.max_concurrent,
        ga_config=ga_config
    )
    
    benchmarker = MultiDomainBenchmarker(benchmark_config)
    
    try:
        logger.info("Running benchmark...")
        results = benchmarker.run_benchmark()
        
        print(f"\nBenchmark Results Summary")
        print("=" * 60)
        print(f"Total benchmarks: {results['benchmark_summary']['total_benchmarks']}")
        print(f"Files processed: {results['benchmark_summary']['files_processed']}")
        print(f"Compressors tested: {', '.join(results['benchmark_summary']['compressors_tested'])}")
        print(f"Data types covered: {', '.join(results['benchmark_summary']['data_types_covered'])}")
        
        print(f"\nCompressor Performance:")
        print("-" * 30)
        for compressor, perf in results['compressor_performance'].items():
            print(f"{compressor.upper()}:")
            print(f"  Benchmarks: {perf['benchmarks_run']}")
            print(f"  Avg Compression Ratio: {perf['avg_compression_ratio']:.2f}x")
            print(f"  Success Rate: {perf['success_rate']:.1%}")
        
        print(f"\nCross-Domain Recommendations:")
        print("-" * 30)
        for domain, compressor in results['cross_domain_recommendations'].items():
            print(f"  {domain}: {compressor}")
        
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


def cmd_batch_process(args) -> None:
    """Run batch processing on multiple datasets."""
    logger = get_logger("CLI-Batch")
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    logger.info(f"Starting batch processing with config: {args.config}")
    
    # Load batch configuration
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    # Create batch processing configuration
    batch_config = BatchProcessingConfig(
        max_concurrent_jobs=config_data.get('max_concurrent_jobs', 4),
        max_memory_per_job_mb=config_data.get('max_memory_per_job_mb', 2048),
        enable_smart_scheduling=config_data.get('enable_smart_scheduling', True),
        retry_failed_jobs=config_data.get('retry_failed_jobs', True),
        auto_recommend_compressors=config_data.get('auto_recommend_compressors', True)
    )
    
    processor = BatchProcessor(batch_config)
    
    try:
        # Create jobs from dataset configurations
        dataset_configs = config_data.get('datasets', [])
        jobs = processor.create_batch_jobs(dataset_configs)
        
        logger.info(f"Created {len(jobs)} batch jobs")
        
        # Process batch
        results = processor.process_batch(jobs, args.output_dir)
        
        print(f"\nBatch Processing Results")
        print("=" * 60)
        print(f"Total jobs: {results['batch_summary']['total_jobs']}")
        print(f"Successful: {results['batch_summary']['successful_jobs']}")
        print(f"Failed: {results['batch_summary']['failed_jobs']}")
        print(f"Success rate: {results['batch_summary']['success_rate']:.1%}")
        print(f"Total runtime: {results['batch_summary']['total_runtime']:.2f}s")
        
        if results['performance_summary']['best_overall_fitness']:
            print(f"\nBest Result:")
            print(f"  Job: {results['performance_summary']['best_overall_job']}")
            print(f"  Compressor: {results['performance_summary']['best_overall_compressor']}")
            print(f"  Fitness: {results['performance_summary']['best_overall_fitness']:.3f}")
        
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


def cmd_create_datasets(args) -> None:
    """Create comprehensive dataset collection."""
    logger = get_logger("CLI-CreateDatasets")
    
    logger.info(f"Creating dataset collection in: {args.output_dir}")
    
    collector = MultiDomainDatasetCollector(args.output_dir)
    
    try:
        metadata = collector.create_comprehensive_dataset_collection()
        
        print(f"\nDataset Collection Created")
        print("=" * 60)
        print(f"Total files: {metadata['collection_metadata']['total_files']:,}")
        print(f"Total size: {metadata['collection_metadata']['total_size_mb']:.2f} MB")
        print(f"Domains: {metadata['collection_metadata']['domains_count']}")
        print(f"Output directory: {metadata['collection_metadata']['base_directory']}")
        
        print(f"\nDomain Breakdown:")
        print("-" * 30)
        for domain, summary in metadata['domain_summary'].items():
            print(f"{domain.replace('_', ' ').title()}:")
            print(f"  Files: {summary['file_count']:,}")
            print(f"  Size: {summary['total_size_bytes'] / (1024*1024):.2f} MB")
        
        logger.info("Dataset collection created successfully")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        sys.exit(1)


def cmd_cross_domain_analysis(args) -> None:
    """Perform cross-domain performance analysis on benchmark results."""
    logger = get_logger("CLI-CrossDomain")
    
    if not os.path.exists(args.results_file):
        logger.error(f"Results file not found: {args.results_file}")
        sys.exit(1)
    
    logger.info(f"Analyzing cross-domain performance from: {args.results_file}")
    
    # Load benchmark results
    with open(args.results_file, 'r') as f:
        results_data = json.load(f)
    
    raw_results = results_data.get('raw_results', [])
    if not raw_results:
        logger.error("No raw results found in the results file")
        sys.exit(1)
    
    evaluator = CrossDomainEvaluator()
    
    try:
        # Perform cross-domain evaluation
        compressor_metrics = evaluator.evaluate_cross_domain_performance(raw_results)
        
        # Generate performance report
        report = evaluator.generate_performance_report(compressor_metrics)
        
        print(f"\nCross-Domain Performance Analysis")
        print("=" * 60)
        print(f"Compressors analyzed: {len(compressor_metrics)}")
        print(f"Total benchmarks: {len(raw_results)}")
        
        print(f"\nOverall Ranking:")
        print("-" * 30)
        for i, (compressor, score) in enumerate(report['overall_ranking'], 1):
            print(f"{i}. {compressor.upper()}: {score:.3f}")
        
        print(f"\nCategory Leaders:")
        print("-" * 30)
        for category, (compressor, score) in report['category_leaders'].items():
            print(f"{category.replace('_', ' ').title()}: {compressor.upper()} ({score:.3f})")
        
        # Save detailed analysis
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            analysis_data = {
                'input_file': args.results_file,
                'compressor_metrics': {name: asdict(metrics) for name, metrics in compressor_metrics.items()},
                'performance_report': report
            }
            
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            logger.info(f"Detailed analysis saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Cross-domain analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Domain Compression Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python multi_domain_cli.py analyze-file --input data.txt
  
  # Get recommendations with speed optimization
  python multi_domain_cli.py recommend --input data.txt --target speed
  
  # Run benchmark on dataset
  python multi_domain_cli.py benchmark --dataset-dir datasets/ --compressors zstd lzma brotli
  
  # Create test datasets
  python multi_domain_cli.py create-datasets --output-dir test_datasets/
        """
    )
    
    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-dir', help='Directory for log files')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # File analysis command
    analyze_parser = subparsers.add_parser('analyze-file', help='Analyze file characteristics')
    analyze_parser.add_argument('--input', '-i', required=True, help='Input file to analyze')
    analyze_parser.add_argument('--output', '-o', help='Output file for detailed analysis (JSON)')
    analyze_parser.set_defaults(func=cmd_analyze_file)
    
    # Compressor recommendation command
    recommend_parser = subparsers.add_parser('recommend', help='Get compressor recommendations')
    recommend_parser.add_argument('--input', '-i', required=True, help='Input file')
    recommend_parser.add_argument('--target', choices=['ratio', 'speed', 'memory', 'balanced'], 
                                 default='balanced', help='Optimization target')
    recommend_parser.add_argument('--max-time', type=float, help='Maximum compression time (seconds)')
    recommend_parser.add_argument('--max-memory', type=int, help='Maximum memory usage (MB)')
    recommend_parser.add_argument('--quality', choices=['high', 'medium', 'low'], 
                                 help='Quality requirements')
    recommend_parser.add_argument('--output', '-o', help='Output file for recommendations (JSON)')
    recommend_parser.set_defaults(func=cmd_recommend_compressor)
    
    # Multi-domain benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run multi-domain benchmark')
    benchmark_parser.add_argument('--dataset-dir', required=True, help='Dataset directory')
    benchmark_parser.add_argument('--compressors', nargs='+', 
                                 choices=['zstd', 'lzma', 'brotli', 'paq8'],
                                 default=['zstd', 'lzma', 'brotli'], 
                                 help='Compressors to benchmark')
    benchmark_parser.add_argument('--output-dir', default='benchmark_results', 
                                 help='Output directory')
    benchmark_parser.add_argument('--file-patterns', nargs='+', default=['*'],
                                 help='File patterns to include')
    benchmark_parser.add_argument('--population-size', type=int, default=20,
                                 help='GA population size')
    benchmark_parser.add_argument('--generations', type=int, default=15,
                                 help='GA generations')  
    benchmark_parser.add_argument('--max-threads', type=int, default=2,
                                 help='Maximum threads per GA')
    benchmark_parser.add_argument('--max-concurrent', type=int, default=4,
                                 help='Maximum concurrent benchmarks')
    benchmark_parser.set_defaults(func=cmd_multi_domain_benchmark)
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Run batch processing')
    batch_parser.add_argument('--config', required=True, help='Batch configuration file (JSON)')
    batch_parser.add_argument('--output-dir', default='batch_results', help='Output directory')
    batch_parser.set_defaults(func=cmd_batch_process)
    
    # Dataset creation command
    datasets_parser = subparsers.add_parser('create-datasets', help='Create dataset collection')
    datasets_parser.add_argument('--output-dir', default='datasets', help='Output directory')
    datasets_parser.set_defaults(func=cmd_create_datasets)
    
    # Cross-domain analysis command
    analysis_parser = subparsers.add_parser('cross-domain-analysis', 
                                          help='Analyze cross-domain performance')
    analysis_parser.add_argument('--results-file', required=True, 
                                help='Benchmark results file (JSON)')
    analysis_parser.add_argument('--output', '-o', help='Output file for analysis (JSON)')
    analysis_parser.set_defaults(func=cmd_cross_domain_analysis)
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    # Setup logging
    setup_cli_logging(args.log_level, args.log_dir)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()