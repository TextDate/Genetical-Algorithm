"""
Multi-Domain Benchmarking Framework

Provides comprehensive benchmarking capabilities across multiple data types,
compressors, and domains. Implements the core T2 functionality for automated
multi-domain application testing and evaluation.
"""

import os
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ga_logging import get_logger
from ga_config import GAConfig
from genetic_algorithm import GeneticAlgorithm
from ga_components.file_analyzer import FileAnalyzer, FileCharacteristics, DataType, DataDomain
from ga_components.compressor_registry import get_compressor_factory

# Import compressors
from Compressors.zstd_compressor import ZstdCompressor
from Compressors.lzma_compressor import LzmaCompressor  
from Compressors.brotli_compressor import BrotliCompressor
from Compressors.paq8_compressor import PAQ8Compressor
from Compressors.ac2_compressor import AC2Compressor


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    file_path: str
    compressor_name: str
    file_characteristics: Dict[str, Any]
    
    # GA Results
    best_fitness: float
    best_parameters: Dict[str, Any]
    generations_run: int
    total_runtime: float
    
    # Performance metrics
    best_compression_ratio: float
    best_compression_time: float
    best_ram_usage: float
    
    # Multi-objective scores
    multi_objective_score: Optional[float] = None
    fitness_component: Optional[float] = None
    time_component: Optional[float] = None
    ram_component: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class MultiDomainBenchmarkConfig:
    """Configuration for multi-domain benchmarking."""
    dataset_dir: str
    output_dir: str
    compressors: List[str]
    file_patterns: List[str] = None  # e.g., ["*.txt", "*.sdf", "*.log"]
    max_file_size: int = 1024 * 1024 * 1024  # 1GB default
    min_file_size: int = 1024  # 1KB default
    max_concurrent_benchmarks: int = 4
    ga_config: Optional[GAConfig] = None
    enable_cross_domain_analysis: bool = True
    save_individual_results: bool = True


class MultiDomainBenchmarker:
    """
    Multi-domain benchmarking system for comprehensive compressor evaluation.
    
    Automatically discovers files, classifies them by domain, runs GA optimization
    for each compressor-file combination, and provides cross-domain analysis.
    """
    
    def __init__(self, config: MultiDomainBenchmarkConfig):
        self.config = config
        self.logger = get_logger("MultiDomainBenchmarker")
        self.file_analyzer = FileAnalyzer()
        self.compressor_factory = get_compressor_factory()
        self.results: List[BenchmarkResult] = []
        self._lock = threading.Lock()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Default GA config if not provided
        if self.config.ga_config is None:
            self.config.ga_config = GAConfig(
                population_size=30,
                generations=25,  # Reduced for benchmarking
                max_threads=2,   # Leave resources for concurrent benchmarks
                output_dir=os.path.join(self.config.output_dir, "individual_runs")
            )
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive multi-domain benchmark.
        
        Returns:
            Dictionary containing benchmark results and analysis
        """
        self.logger.info("Starting multi-domain benchmark")
        start_time = time.time()
        
        # Discover and analyze files
        files_to_benchmark = self._discover_files()
        self.logger.info(f"Discovered {len(files_to_benchmark)} files for benchmarking")
        
        if not files_to_benchmark:
            raise ValueError("No suitable files found for benchmarking")
        
        # Load parameters for all compressors
        param_configs = self._load_compressor_parameters()
        
        # Create benchmark tasks
        benchmark_tasks = self._create_benchmark_tasks(files_to_benchmark, param_configs)
        self.logger.info(f"Created {len(benchmark_tasks)} benchmark tasks")
        
        # Run benchmarks
        self._execute_benchmarks(benchmark_tasks)
        
        # Generate analysis
        analysis = self._generate_analysis()
        
        # Save results
        self._save_results(analysis)
        
        total_time = time.time() - start_time
        self.logger.info(f"Multi-domain benchmark completed in {total_time:.2f} seconds")
        
        return analysis
    
    def _discover_files(self) -> List[Tuple[str, FileCharacteristics]]:
        """Discover and analyze files in the dataset directory."""
        files_with_characteristics = []
        
        # File pattern matching
        patterns = self.config.file_patterns or ['*']
        dataset_path = Path(self.config.dataset_dir)
        
        for pattern in patterns:
            for file_path in dataset_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        file_size = file_path.stat().st_size
                        
                        # Size filtering
                        if not (self.config.min_file_size <= file_size <= self.config.max_file_size):
                            continue
                        
                        # Analyze file
                        characteristics = self.file_analyzer.analyze_file(str(file_path))
                        files_with_characteristics.append((str(file_path), characteristics))
                        
                        self.logger.debug(f"Discovered {file_path.name}: "
                                        f"{characteristics.data_type.value}/{characteristics.data_domain.value}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze {file_path}: {e}")
                        continue
        
        return files_with_characteristics
    
    def _load_compressor_parameters(self) -> Dict[str, Dict]:
        """Load parameter configurations for all compressors."""
        # Look for params.json in config directory
        param_file_paths = [
            "config/params.json",
            "../config/params.json", 
            "params.json"
        ]
        
        param_config = None
        for path in param_file_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    param_config = json.load(f)
                break
        
        if param_config is None:
            raise FileNotFoundError("Could not find params.json configuration file")
        
        # Extract configurations for requested compressors
        configs = {}
        for compressor in self.config.compressors:
            if compressor in param_config:
                configs[compressor] = param_config[compressor]
            else:
                self.logger.warning(f"No parameters found for compressor: {compressor}")
        
        return configs
    
    def _create_benchmark_tasks(self, files_with_characteristics: List[Tuple[str, FileCharacteristics]],
                               param_configs: Dict[str, Dict]) -> List[Dict]:
        """Create individual benchmark tasks."""
        tasks = []
        
        for file_path, characteristics in files_with_characteristics:
            # Use recommended compressors if available, otherwise all configured
            compressors_to_test = (
                [c for c in characteristics.recommended_compressors if c in self.config.compressors]
                or self.config.compressors
            )
            
            for compressor_name in compressors_to_test:
                if compressor_name in param_configs:
                    task = {
                        'file_path': file_path,
                        'file_characteristics': characteristics,
                        'compressor_name': compressor_name,
                        'param_ranges': param_configs[compressor_name]
                    }
                    tasks.append(task)
        
        return tasks
    
    def _execute_benchmarks(self, tasks: List[Dict]):
        """Execute benchmark tasks with controlled concurrency."""
        completed_count = 0
        total_tasks = len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_benchmarks) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_benchmark, task): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        with self._lock:
                            self.results.append(result)
                        completed_count += 1
                        self.logger.info(f"Completed benchmark {completed_count}/{total_tasks}: "
                                       f"{task['compressor_name']} on {Path(task['file_path']).name}")
                    
                except Exception as e:
                    self.logger.error(f"Benchmark task failed: {task['compressor_name']} "
                                    f"on {Path(task['file_path']).name}: {e}")
        
        self.logger.info(f"Completed {completed_count}/{total_tasks} benchmarks")
    
    def _run_single_benchmark(self, task: Dict) -> Optional[BenchmarkResult]:
        """Run a single benchmark task."""
        file_path = task['file_path']
        compressor_name = task['compressor_name']
        param_ranges = task['param_ranges']
        characteristics = task['file_characteristics']
        
        try:
            # Create compressor instance
            compressor = self._create_compressor(compressor_name, file_path)
            if compressor is None:
                return None
            
            # Create GA config for this run
            run_config = GAConfig(
                population_size=self.config.ga_config.population_size,
                generations=self.config.ga_config.generations,
                mutation_rate=self.config.ga_config.mutation_rate,
                crossover_rate=self.config.ga_config.crossover_rate,
                max_threads=self.config.ga_config.max_threads,
                output_dir=os.path.join(self.config.output_dir, "individual_runs", 
                                      f"{compressor_name}_{Path(file_path).stem}"),
                enable_multi_objective=self.config.ga_config.enable_multi_objective,
                fitness_weight=self.config.ga_config.fitness_weight,
                time_weight=self.config.ga_config.time_weight,
                ram_weight=self.config.ga_config.ram_weight
            )
            
            # Run GA optimization
            start_time = time.time()
            ga = GeneticAlgorithm(param_ranges, compressor, run_config)
            
            best_solution, best_fitness = ga.run()
            total_runtime = time.time() - start_time
            
            # Extract additional metrics from GA run
            best_individual = ga.population[0]  # Best individual should be first after final sort
            compression_ratio = best_individual[1]  # Fitness value
            compression_time = getattr(best_individual, 'compression_time', 0.0)
            ram_usage = getattr(best_individual, 'ram_usage', 0.0)
            
            # Multi-objective components if available
            multi_obj_score = None
            fitness_comp = None
            time_comp = None
            ram_comp = None
            
            if hasattr(best_individual, 'multi_objective_score'):
                multi_obj_score = best_individual.multi_objective_score
                fitness_comp = best_individual.fitness_component
                time_comp = best_individual.time_component  
                ram_comp = best_individual.ram_component
            
            # Create result
            result = BenchmarkResult(
                file_path=file_path,
                compressor_name=compressor_name,
                file_characteristics=characteristics.to_dict(),
                best_fitness=best_fitness,
                best_parameters=best_solution,
                generations_run=run_config.generations,
                total_runtime=total_runtime,
                best_compression_ratio=compression_ratio,
                best_compression_time=compression_time,
                best_ram_usage=ram_usage,
                multi_objective_score=multi_obj_score,
                fitness_component=fitness_comp,
                time_component=time_comp,
                ram_component=ram_comp
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to run benchmark {compressor_name} on {file_path}: {e}")
            return None
        
        finally:
            # Cleanup compressor temporary files
            if 'compressor' in locals():
                try:
                    compressor.erase_temp_files()
                except:
                    pass
    
    def _create_compressor(self, compressor_name: str, file_path: str):
        """Create compressor instance."""
        temp_dir = os.path.join(self.config.output_dir, "temp", compressor_name)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            if compressor_name == 'zstd':
                return ZstdCompressor(file_path, temp=temp_dir)
            elif compressor_name == 'lzma':
                return LzmaCompressor(file_path, temp=temp_dir)
            elif compressor_name == 'brotli':
                return BrotliCompressor(file_path, temp=temp_dir)
            elif compressor_name == 'paq8':
                return PAQ8Compressor(file_path, temp=temp_dir)
            elif compressor_name == 'ac2':
                # AC2 requires reference file - skip if not available
                self.logger.warning("AC2 compressor requires reference file - skipping")
                return None
            else:
                self.logger.error(f"Unknown compressor: {compressor_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create {compressor_name} compressor: {e}")
            return None
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of benchmark results."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by various dimensions
        by_compressor = {}
        by_data_type = {}
        by_data_domain = {}
        by_file = {}
        
        for result in self.results:
            # By compressor
            comp = result.compressor_name
            if comp not in by_compressor:
                by_compressor[comp] = []
            by_compressor[comp].append(result)
            
            # By data type
            data_type = result.file_characteristics['data_type']
            if data_type not in by_data_type:
                by_data_type[data_type] = []
            by_data_type[data_type].append(result)
            
            # By data domain
            domain = result.file_characteristics['data_domain']
            if domain not in by_data_domain:
                by_data_domain[domain] = []
            by_data_domain[domain].append(result)
            
            # By file
            file_name = Path(result.file_path).name
            if file_name not in by_file:
                by_file[file_name] = []
            by_file[file_name].append(result)
        
        # Generate summary statistics
        compressor_performance = self._analyze_compressor_performance(by_compressor)
        domain_analysis = self._analyze_by_domain(by_data_domain)
        file_analysis = self._analyze_by_file(by_file)
        cross_domain_recommendations = self._generate_cross_domain_recommendations()
        
        analysis = {
            'benchmark_summary': {
                'total_benchmarks': len(self.results),
                'compressors_tested': list(by_compressor.keys()),
                'data_types_covered': list(by_data_type.keys()),
                'data_domains_covered': list(by_data_domain.keys()),
                'files_processed': len(by_file)
            },
            'compressor_performance': compressor_performance,
            'domain_analysis': domain_analysis,
            'file_analysis': file_analysis,
            'cross_domain_recommendations': cross_domain_recommendations,
            'raw_results': [result.to_dict() for result in self.results]
        }
        
        return analysis
    
    def _analyze_compressor_performance(self, by_compressor: Dict[str, List[BenchmarkResult]]) -> Dict:
        """Analyze overall compressor performance."""
        performance = {}
        
        for compressor, results in by_compressor.items():
            if not results:
                continue
            
            # Calculate statistics
            compression_ratios = [r.best_compression_ratio for r in results]
            compression_times = [r.best_compression_time for r in results if r.best_compression_time > 0]
            runtimes = [r.total_runtime for r in results]
            
            performance[compressor] = {
                'benchmarks_run': len(results),
                'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios),
                'best_compression_ratio': max(compression_ratios),
                'worst_compression_ratio': min(compression_ratios),
                'avg_compression_time': sum(compression_times) / len(compression_times) if compression_times else 0,
                'avg_total_runtime': sum(runtimes) / len(runtimes),
                'success_rate': len([r for r in results if r.best_fitness > 1.0]) / len(results)
            }
        
        return performance
    
    def _analyze_by_domain(self, by_domain: Dict[str, List[BenchmarkResult]]) -> Dict:
        """Analyze performance by data domain."""
        domain_analysis = {}
        
        for domain, results in by_domain.items():
            if not results:
                continue
            
            # Find best compressor for this domain
            compressor_scores = {}
            for result in results:
                comp = result.compressor_name
                if comp not in compressor_scores:
                    compressor_scores[comp] = []
                compressor_scores[comp].append(result.best_compression_ratio)
            
            # Calculate average performance per compressor
            avg_scores = {}
            for comp, scores in compressor_scores.items():
                avg_scores[comp] = sum(scores) / len(scores)
            
            best_compressor = max(avg_scores.items(), key=lambda x: x[1]) if avg_scores else None
            
            domain_analysis[domain] = {
                'files_processed': len(set(r.file_path for r in results)),
                'compressors_tested': list(compressor_scores.keys()),
                'best_compressor': best_compressor[0] if best_compressor else None,
                'best_avg_ratio': best_compressor[1] if best_compressor else 0.0,
                'compressor_performance': avg_scores
            }
        
        return domain_analysis
    
    def _analyze_by_file(self, by_file: Dict[str, List[BenchmarkResult]]) -> Dict:
        """Analyze results by individual file."""
        file_analysis = {}
        
        for file_name, results in by_file.items():
            if not results:
                continue
            
            # Find best result for this file
            best_result = max(results, key=lambda r: r.best_compression_ratio)
            
            file_analysis[file_name] = {
                'file_path': best_result.file_path,
                'data_type': best_result.file_characteristics['data_type'],
                'data_domain': best_result.file_characteristics['data_domain'],
                'file_size': best_result.file_characteristics['file_size'],
                'compressors_tested': len(results),
                'best_compressor': best_result.compressor_name,
                'best_compression_ratio': best_result.best_compression_ratio,
                'predicted_compressibility': best_result.file_characteristics['predicted_compressibility'],
                'prediction_accuracy': self._assess_prediction_accuracy(best_result)
            }
        
        return file_analysis
    
    def _assess_prediction_accuracy(self, result: BenchmarkResult) -> str:
        """Assess accuracy of compressibility prediction."""
        predicted = result.file_characteristics['predicted_compressibility']
        actual_ratio = result.best_compression_ratio
        
        # Simple assessment based on compression ratio
        if actual_ratio >= 2.0:
            actual = "high"
        elif actual_ratio >= 1.5:
            actual = "medium"
        else:
            actual = "low"
        
        return "accurate" if predicted == actual else "inaccurate"
    
    def _generate_cross_domain_recommendations(self) -> Dict[str, str]:
        """Generate compressor recommendations by data domain."""
        recommendations = {}
        
        # Group by domain and find best average performer
        domain_performance = {}
        for result in self.results:
            domain = result.file_characteristics['data_domain']
            if domain not in domain_performance:
                domain_performance[domain] = {}
            
            comp = result.compressor_name
            if comp not in domain_performance[domain]:
                domain_performance[domain][comp] = []
            
            domain_performance[domain][comp].append(result.best_compression_ratio)
        
        # Calculate recommendations
        for domain, comp_results in domain_performance.items():
            best_comp = None
            best_score = 0.0
            
            for comp, ratios in comp_results.items():
                avg_ratio = sum(ratios) / len(ratios)
                if avg_ratio > best_score:
                    best_score = avg_ratio
                    best_comp = comp
            
            if best_comp:
                recommendations[domain] = best_comp
        
        return recommendations
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Save benchmark results and analysis."""
        # Save complete analysis as JSON
        analysis_file = os.path.join(self.config.output_dir, "multi_domain_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save CSV summary of results
        csv_file = os.path.join(self.config.output_dir, "benchmark_results_summary.csv")
        with open(csv_file, 'w', newline='') as f:
            if self.results:
                fieldnames = [
                    'file_name', 'file_size', 'data_type', 'data_domain', 
                    'compressor', 'compression_ratio', 'compression_time', 
                    'total_runtime', 'predicted_compressibility'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    writer.writerow({
                        'file_name': Path(result.file_path).name,
                        'file_size': result.file_characteristics['file_size'],
                        'data_type': result.file_characteristics['data_type'],
                        'data_domain': result.file_characteristics['data_domain'],
                        'compressor': result.compressor_name,
                        'compression_ratio': result.best_compression_ratio,
                        'compression_time': result.best_compression_time,
                        'total_runtime': result.total_runtime,
                        'predicted_compressibility': result.file_characteristics['predicted_compressibility']
                    })
        
        # Save compressor recommendations
        rec_file = os.path.join(self.config.output_dir, "compressor_recommendations.json")
        with open(rec_file, 'w') as f:
            json.dump(analysis['cross_domain_recommendations'], f, indent=2)
        
        self.logger.info(f"Results saved to {self.config.output_dir}")
        self.logger.info(f"Analysis: {analysis_file}")
        self.logger.info(f"CSV Summary: {csv_file}")
        self.logger.info(f"Recommendations: {rec_file}")